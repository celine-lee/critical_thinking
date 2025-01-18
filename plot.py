import torch
import math
import json
import re
import sys
import ipdb
import traceback
import os
import ast
import shutil
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import random
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from collections import defaultdict
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

import argparse

global compute_random
global temperature
global num_beams
global num_gens
global foldername
global n_buckets

def get_args():
    global compute_random
    global temperature
    global num_beams
    global num_gens
    global foldername
    global n_buckets
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--n_buckets", type=int, default=10)
    parser.add_argument("--num_gens", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--get_isolated", nargs='+', default=['k', 't', 'N'])
    parser.add_argument("--k_vals", nargs='+', default=None)
    parser.add_argument("--t_vals", nargs='+', default=None)
    parser.add_argument("--N_vals", nargs='+', default=None)
    parser.add_argument("--models", nargs='+', default=['3.1-8B'])
    parser.add_argument("--delete_old", action="store_true")
    parser.add_argument("--all_methods", action="store_true")
    args = parser.parse_args()
    foldername = os.path.join(f"{args.output_folder.rstrip('/')}_graphs{'_noinsn' if not args.all_methods else ''}_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}")
    temperature = args.temperature
    num_beams = args.num_beams
    num_gens = args.num_gens
    n_buckets = args.n_buckets

    if "even_odd" in args.output_folder: compute_random = lambda k: 0.5
    elif "array_index" in args.output_folder: compute_random = lambda k: 1/int(k)
    return args


model_colors = {
    "3.1-8B": "purple",
    "Qwen2.5-32B": "blue",
    "Qwen2.5-14B": "brown",
    "Qwen2.5-7B": "yellow",
    # "2-13b-chat-hf": "blue",
    "Mistral-7B": "red",
    # "OLMo-2-1124-13B": "green",
    "OLMo-2-1124-7B": "black",
    # "CodeQwen1.5-7B": "green",
    "Ministral-8B": "orange",
    # "deepseek-coder-6.7b": "yellow",
    # "deepseek-coder-7b-instruct-v1.5": "brown",
    # "2-13b-hf": "blue",
    # "3.2-3B": "blue",
    # "3.2-1B": "red",
    "gemma-2-9b": "pink",
    # "codegemma-7b": "yellow",
}

colormap = get_cmap("tab10")  # Use a colormap with distinct colors
method_markers = {
    "request_descriptor_brief": "d",
    "request_descriptor_none": "*",
    "request_descriptor_detail": "^",
    "request_descriptor_states_long": "o",
    "request_descriptor_states_short": ".",
    "_none_": "x"
}
method_colors = {methodname: colormap(i) for i, methodname in enumerate(method_markers.keys())}
method_linestyle = {
    "request_descriptor_brief": ":",
    "request_descriptor_none": "--", 
    "request_descriptor_detail": "-.",
    "request_descriptor_states_long": "-",
    "request_descriptor_states_short": "--",
    "_none_": "-"
}

def load_data(data_folder, k_vals, t_vals, N_vals, target_methodname=None):
    ks = []
    Ns = []
    ts = []
    models = []
    methods = []
    gen_toks = []
    correct = []

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(data_folder, f"k*")):
        parsed_experimentname = re.search(rf"k(\d+)_N(\d+)_t(\d+)", subfolder)
        if parsed_experimentname is None:
            continue
        k = parsed_experimentname.group(1)
        N = parsed_experimentname.group(2)
        t = parsed_experimentname.group(3)

        if k not in k_vals: continue
        if t not in t_vals: continue
        if N not in N_vals: continue
        for experiment_file in glob.glob(os.path.join(subfolder, "*")):
            if f"T{temperature}" not in experiment_file:
                continue
            if re.search(r'_B\d+_S\d+', experiment_file):
                if f"_B{num_beams}_S{num_gens}.json" not in experiment_file: 
                    continue
            elif temperature > 0.0: 
                if num_beams != 1 or num_gens != 6: 
                    continue
            modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
            if not any(model_str in modelname for model_str in model_colors):
                continue
            
            results = json.load(open(experiment_file))
            for methodname in method_markers:
                if methodname in experiment_file:
                    break
            if methodname not in experiment_file:
                methodname = "_none_"
            if target_methodname is not None and methodname != target_methodname: 
                continue

            ks.extend([k for _ in results])
            Ns.extend([N for _ in results])
            ts.extend([t for _ in results])
            models.extend([modelname for _ in results])
            methods.extend([methodname for _ in results])
            gen_toks.extend([ex["generated_tokens"] for ex in results])
            correct.extend([ex["correct"] for ex in results])

    data = {
        "k": ks,
        "N": Ns,
        "t": ts,
        "Model": models,
        "Method": methods,
        "No gen toks": gen_toks,
        "Correct?": correct,
    }
    all_var_vals = [set(ks), set(ts), set(N)]

    # Create a DataFrame for the new data
    df = pd.DataFrame(data)
    # Create a column for grouping
    df["Group"] = list(zip(df["Method"], df["Model"], df["k"], df["N"], df["t"]))

    return df, all_var_vals


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

# to check that different prompting methods to produce differnt-lenght CoTs
def plot_requested_vs_generated(df):
    # Separate data by model size
    model_data = {
        model_name: df[df["Model"].str.contains(model_name)].sort_values(by="No gen toks")
        for model_name in model_colors
    }

    # Plot box plots for each model size and requested length category
    plt.figure(figsize=(12, 6))

    # Get sorted unique requested lengths for x-axis labels
    unique_lengths = sorted(df["Method"].unique())
    tick_positions = None

    plotted_something = False
    # Plot box plots for each requested length category for each model
    for i, (model, model_df) in enumerate(model_data.items(), start=1):
        grouped = model_df.groupby("Method", observed=True)["No gen toks"].apply(list)
        if len(grouped) == 0:
            continue
        positions = [
            unique_lengths.index(req_len) + (i - 2) * 0.2 for req_len in grouped.index
        ]
        if not tick_positions or len(tick_positions) < len(positions):
            tick_positions = positions
        plt.boxplot(
            grouped,
            positions=positions,
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor=model_colors[model], color=model_colors[model]),
            medianprops=dict(color="black"),
            showfliers=False,
        )
        plotted_something = True
    if not plotted_something:
        return

    # Set x-axis labels
    plt.xticks(ticks=tick_positions, labels=unique_lengths, rotation=45)
    plt.xlabel("Method")
    plt.ylabel("Actual length (no. tokens)")
    plt.title(f"Box Plot of Generated Tokens by Method and Model")

    # Legend for the models
    legend_elements = [
        Line2D([0], [0], color=model_colors[model], lw=2, label=model)
        for model in model_colors
    ]
    plt.legend(handles=legend_elements, loc="best", fancybox=True, shadow=True)

    plt.savefig(
        os.path.join(foldername, f"lengthvsrequested_boxplot.png")
    )
    plt.clf()

def calculate_buckets_samerange(sub_df, bins=None, groupby_key="Model"):
    if len(sub_df) == 0:
        return None, None
    if bins is None:
        min_len = df["No gen toks"].min()
        max_len = df["No gen toks"].max()
        bins = np.linspace(min_len, max_len, n_buckets + 1)

    sub_df["Length Bucket"] = pd.cut(
        sub_df["No gen toks"], bins, include_lowest=True
    )

    bucket_avg = (
        sub_df.groupby([groupby_key, "Length Bucket"], observed=True)["Correct?"].mean().reset_index()
    )
    sub_df = sub_df.merge(bucket_avg, on=groupby_key, suffixes=('', '_mean'))
    bucket_avg["Bucket Center"] = bucket_avg["Length Bucket"].apply(lambda x: x.mid)
    return bucket_avg, sub_df

def calculate_buckets_samesize(sub_df, groupby_key="Model"):
    if len(sub_df) == 0: return None, None
    # Use qcut to create equal-sized buckets by count
    if len(sub_df["No gen toks"].unique()) < n_buckets:    
        sub_df.loc[:, "Length Bucket"] = pd.qcut(
            sub_df["No gen toks"], q=len(sub_df["No gen toks"].unique()), duplicates="drop"
        )
    else:
        sub_df.loc[:, "Length Bucket"] = pd.qcut(
            sub_df["No gen toks"], q=n_buckets, duplicates="drop"
        )

    bucket_avg = (
        sub_df.groupby([groupby_key, "Length Bucket"], observed=True)["Correct?"].mean().reset_index()
    )

    # Calculate the bucket center by averaging bin edges
    bucket_avg["Bucket Center"] = bucket_avg["Length Bucket"].apply(
        lambda x: (x.left + x.right) / 2
    )
    sub_df = sub_df.merge(bucket_avg, on=groupby_key, suffixes=('', '_mean'))

    return bucket_avg, sub_df

def plot_correctness_by_ttoks_isolate_factor(df, k, t, N, modelname, stats_dict, factor_vals):
    assert sum((factor is None for factor in (k, t, N, modelname))) == 1, f"{(k, t, N, modelname)} one must be None"
    # Filter the data for the specific model, k, t, N, modelname
    filtered_data = df[
        (df["Model"].str.contains(modelname) if modelname is not None else True)
        & ((df["k"] == k) if k is not None else True)
        & ((df["t"] == t) if t is not None else True)
        & ((df["N"] == N) if N is not None else True)
    ]


    if t is None:
        isolated_factor = "t"
    elif k is None:
        isolated_factor = "k"
    elif N is None:
        isolated_factor = "N"
    elif modelname is None: 
        isolated_factor = "Model"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: {(k, t, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))
    if isolated_factor == "Model":
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=str)
    else:
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=int)
        max_factor = int(factor_values[-1])

    ktn_intrep = (int(k) if k is not None else None, int(t) if t is not None else None, int(N) if N is not None else None)
    stats_dict[ktn_intrep] = {}

    used_factor_values = []
    last_max = None
    # Iterate over unique t values
    for factor_value in factor_values:
        if isolated_factor == "Model":
            plot_this_one = False
            for factor_val in factor_vals:
                if factor_val in factor_value: 
                    plot_this_one = True
                    factor_value = factor_val
                    break
            if not plot_this_one: continue
        else:
            if factor_value not in factor_vals: continue
        used_factor_values.append(factor_value)
        if modelname is None:
            filtered_data_factor = filtered_data[filtered_data[isolated_factor].str.contains(factor_value)]
        else:
            filtered_data_factor = filtered_data[filtered_data[isolated_factor] == factor_value]
        bucket_avg, filtered_data_factor = calculate_buckets_samesize(filtered_data_factor)
        if filtered_data_factor is None: continue
        if len(bucket_avg) == 0: continue

        if modelname:
            factor_value = int(factor_value)
            # Normalize the intensity of the color based on t
            color_intensity = factor_value / (max_factor + 1) 
            base_color = model_colors.get(modelname, "blue")
        else:
            color_intensity = 0.8
            base_color = model_colors.get(factor_value, "blue")
        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Find the index of the maximum value
        max_index = np.argmax(bucket_avg["Correct?"])
        this_max = bucket_avg["Bucket Center"][max_index]
        stats_dict[ktn_intrep][factor_value] = {"peak": this_max}

        # Data up to the first peak
        peak_range = bucket_avg["Bucket Center"][:max_index + 1]
        correctness_range = bucket_avg["Correct?"][:max_index + 1]

        # Remove NaN values from peak_range and correctness_range
        valid_indices = ~correctness_range.isna()
        filtered_peak_range = peak_range[valid_indices]
        filtered_correctness_range = correctness_range[valid_indices]

        # Fit a linear regression model
        if len(filtered_peak_range) > 1:  # Ensure there are enough points to fit
            reg = LinearRegression()
            reg.fit(filtered_peak_range.values.reshape(-1, 1), filtered_correctness_range)
            predictions = reg.predict(filtered_peak_range.values.reshape(-1, 1))

            # Calculate regression statistics
            mse = mean_squared_error(filtered_correctness_range, predictions)
            pearson_corr, _ = pearsonr(filtered_peak_range, filtered_correctness_range)

            # Store stats
            stats_dict[ktn_intrep][factor_value]["regression"] = {
                "weights": reg.coef_[0],  # Slope
                "intercept": reg.intercept_,
                "mse": mse,
                "pearson_corr": pearson_corr,
            }

            # Overlay the regression line
            plt.plot(
                filtered_peak_range, predictions, linestyle="--", color=rgba_color
            )
        
        # Plot the average correctness for each model size and method
        plt.plot(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"],
            color=rgba_color,
            label=f"{isolated_factor}={factor_value} ({int(this_max - last_max)})" if last_max is not None else f"{isolated_factor}={factor_value}",
        )

        last_max = this_max
        sem_values = filtered_data_factor.groupby("Bucket Center", observed=True)["Correct?"].apply(sem)
        # Calculate confidence intervals
        ci = sem_values * 1.96  # For 95% confidence
        plt.fill_between(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"] - ci.values,
            bucket_avg["Correct?"] + ci.values,
            alpha=color_intensity,
            color=rgba_color,
        )

        # Place a dot at the maximum value
        plt.scatter(this_max, bucket_avg["Correct?"][max_index], color='red', alpha=color_intensity)
    
    if len(used_factor_values) == 0: return
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Average Correctness")
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.title(
        f"Average Correctness vs. No. of Generated Tokens ({k, t, N, modelname} Buckets={n_buckets})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"{isolated_factor}{''.join(used_factor_values)}_"
    if k is None:
        filename += f"t{t}_N{N}_{modelname}.png"
    elif t is None:
        filename += f"k{k}_N{N}_{modelname}.png"
    elif N is None:
        filename += f"k{k}_t{t}_{modelname}.png"
    elif modelname is None:
        filename = f"byModel_k{k}_t{t}_N{N}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_ttoks_across_method(df, k, t, N, modelname):
    # Filter the data for the specific model, k, t, N
    filtered_data = df[
        (df["Model"].str.contains(modelname))
        & ((df["k"] == k) if k is not None else True)
        & ((df["t"] == t) if t is not None else True)
        & ((df["N"] == N) if N is not None else True)
    ]

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No correct examples found for: {(k, t, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))
    
    for methodname in method_markers:
        filtered_data_method = filtered_data[filtered_data['Method'].str.contains(methodname)]
        bucket_avg, filtered_data_method = calculate_buckets_samesize(filtered_data_method)
        if filtered_data_method is None:
            continue

        base_color = method_colors.get(methodname, "blue")
        rgba_color = mcolors.to_rgba(base_color)
        # Plot the average correctness for each model size and method
        plt.plot(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"],
            color=rgba_color,
            label=methodname.lstrip("_"),
            marker=method_markers[methodname],
            linestyle=method_linestyle[methodname]
        )
        sem_values = filtered_data_method.groupby("Bucket Center", observed=True)["Correct?"].apply(sem)
        # Calculate confidence intervals
        ci = sem_values * 1.96  # For 95% confidence
        plt.fill_between(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"] - ci.values,
            bucket_avg["Correct?"] + ci.values,
            alpha=0.8,
            color=rgba_color,
        )

    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Average Correctness")
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.title(
        f"Average Correctness vs. No. of Generated Tokens ({k, t, N, modelname} Buckets={n_buckets})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"k{k}_t{t}_N{N}_{modelname}_bymethod.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_N_per_compute_budget(df, modelname, k, t):
    
    filtered_data = df[
        (df["Model"].str.contains(modelname))
        & (df["k"] == k)
        & (df["t"] == t)
    ]
    
    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No data found for Model: {modelname}.")
        return
    
    # Create buckets for "No gen toks" (compute budget)
    filtered_data = calculate_buckets_samesize(
        filtered_data, groupby_key="N"
    )
    
    # Ensure bucket calculation succeeded
    if filtered_data is None:
        print("Bucket calculation failed. Ensure there is sufficient data for bucketing.")
        return

    # Cast N to integers for proper sorting
    filtered_data["N"] = filtered_data["N"].astype(int)
    # Group by bucket center and N, and calculate average correctness
    grouped_data = (
        filtered_data.groupby(["Bucket Center", "N"], observed=True)["Correct?"]
        .mean()
        .reset_index()
        .sort_values(by=["Bucket Center", "N"])
    )
    # Plot the data
    plt.figure(figsize=(12, 6))
    
    compute_values = sorted(grouped_data["Bucket Center"].unique())
    max_compute = compute_values[-1]

    # Plot a line for each bucket
    for bucket_center in compute_values:

        # Normalize the intensity of the color based on bucket center
        color_intensity = bucket_center / max_compute
        base_color = model_colors.get(modelname, "blue")
        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)

        bucket_data = grouped_data[grouped_data["Bucket Center"] == bucket_center]
        plt.plot(
            bucket_data["N"],  # x-axis: N values
            bucket_data["Correct?_mean"],  # y-axis: average correctness
            marker="o",
            label=f"Bucket: {bucket_center:.1f}",
            color=rgba_color,
        )
    
    # Customize the plot
    plt.xlabel("N")
    plt.ylabel("Average Correctness")
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.title(f"Average Correctness vs. N (Model: {modelname}, k={k} t={t}, Buckets: {n_buckets})")
    plt.legend(title="Compute Budget Buckets", loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save and clear the figure
    plt.savefig(
        os.path.join(foldername, f"{modelname}_k{k}_t{t}_correctness_by_N_buckets{n_buckets}.png")
    )
    plt.clf()

def scatter_ttoks_by_complexity(df, modelname, N, t):
    # Filter the data for the specific model, N, and t
    filtered_data = df[
        (df["Model"].str.contains(modelname))
        & (df["N"] == N)
        & (df["t"] == t)
    ]

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No correct examples found for Model: {modelname}, N={N}, t={t}.")
        return

    plt.figure(figsize=(12, 6))

    filtered_data["k"] = filtered_data["k"].astype(int)
    filtered_data["No gen toks"] = filtered_data["No gen toks"].astype(int)
    bucket_size = math.ceil(filtered_data["No gen toks"].max() / n_buckets)

    # Plot the average correctness for each model size and method
    data = np.zeros((n_buckets * bucket_size, filtered_data["k"].max()+1)) # 
    totals = np.zeros((n_buckets * bucket_size, filtered_data["k"].max()+1))
    for k_val, no_gen_val, correct in zip(filtered_data["k"], filtered_data["No gen toks"], filtered_data["Correct?"]):
        data[(no_gen_val // bucket_size) * bucket_size:((no_gen_val // bucket_size) + 1) * bucket_size, k_val] += int(correct)
        totals[(no_gen_val // bucket_size) * bucket_size:((no_gen_val // bucket_size) + 1) * bucket_size, k_val] += 1
    data = np.where(totals == 0, np.nan, data / totals)

    # Identify columns to keep (those with at least one non-NaN value)
    columns_with_data = ~np.isnan(data).all(axis=0)

    # Filter the data and totals arrays
    data = data[:, columns_with_data]
    filtered_k_values = np.arange(filtered_data["k"].max() + 1)[columns_with_data]

    plt.imshow(data, interpolation="nearest", aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar()

    # Customize plot labels and legend
    plt.ylabel("No gen toks")
    plt.xlabel("k")
    plt.xticks(ticks=np.arange(len(filtered_k_values)), labels=filtered_k_values)

    plt.title(
        f"No. of Generated Tokens vs. DFA complexity (N={N}, t={t}, {modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)

    # Save and clear the figure
    os.makedirs(os.path.join(foldername, "scatter_plots"), exist_ok=True)
    plt.savefig(
        os.path.join(
            foldername,
            "scatter_plots",
            f"N{N}_t{t}_{modelname}_ttok_by_k.png",
        )
    )
    plt.clf()

def scatter_ttoks_by_inst_complexity(df, modelname, k, t):
    # Filter the data for the specific model, N, and t
    filtered_data = df[
        (df["Model"].str.contains(modelname))
        & (df["k"] == k)
        & (df["t"] == t)
    ]

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No correct examples found for Model: {modelname}, k={k}, t={t}.")
        return

    plt.figure(figsize=(12, 6))

    filtered_data["N"] = filtered_data["N"].astype(int)
    filtered_data["No gen toks"] = filtered_data["No gen toks"].astype(int)
    bucket_size = math.ceil(filtered_data["No gen toks"].max() / n_buckets)

    # Plot the average correctness for each model size and method
    data = np.zeros((n_buckets * bucket_size, filtered_data["N"].max()+1)) # 
    totals = np.zeros((n_buckets * bucket_size, filtered_data["N"].max()+1))
    for N_val, no_gen_val, correct in zip(filtered_data["N"], filtered_data["No gen toks"], filtered_data["Correct?"]):
        data[(no_gen_val // bucket_size) * bucket_size:((no_gen_val // bucket_size) + 1) * bucket_size, N_val] += int(correct)
        totals[(no_gen_val // bucket_size) * bucket_size:((no_gen_val // bucket_size) + 1) * bucket_size, N_val] += 1
    data = np.where(totals == 0, np.nan, data / totals)
    data = data[:, ~np.all(np.isnan(data), axis=0)]


    plt.imshow(data, interpolation="nearest", aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar()

    # Customize plot labels and legend
    plt.ylabel("No gen toks")
    plt.xlabel("N")
    plt.xticks(list(range(len(set(filtered_data["N"])))), sorted(list(set(filtered_data["N"]))))

    plt.title(
        f"No. of Generated Tokens vs. instance complexity (k={k}, t={t}, {modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(
        os.path.join(
            foldername,
            "scatter_plots",
            f"k{k}_t{t}_{modelname}_ttok_by_N.png",
        )
    )
    plt.clf()

def meta_plot_corr(stats_dict, modelname, all_var_vals):
    idx_to_var = [(0, "k"), (1, "t"), (2, "N")]
    colormap = get_cmap("tab10")  # Use a colormap with distinct colors

    for tuple_idx, var_changing in idx_to_var:
        for hold_idx, hold_var in idx_to_var:
            if hold_idx == tuple_idx: continue
            other_var_idx = len(idx_to_var) - hold_idx - tuple_idx

            for hold_var_val in all_var_vals[hold_idx]:
                dfas_to_include = [dfa_detail for dfa_detail in stats_dict.keys() if (dfa_detail[hold_idx] == int(hold_var_val)) and (dfa_detail[tuple_idx] is None)]
                dfas_to_include = sorted(dfas_to_include, key=lambda ent: ent[other_var_idx])
                color_map = {dfa_detail: colormap(i) for i, dfa_detail in enumerate(dfas_to_include)}
                
                # GRAPH 1: DFA Complexity vs. Pearson Correlation of ttoks & performance regression line
                plt.figure(figsize=(10, 6))
                for dfa_detail in dfas_to_include:
                    experiment_results = stats_dict[dfa_detail]
                    data = {
                        var_changing: [],
                        "pearson_corr": [],
                    }

                    for var_value, var_experiment_results in experiment_results.items():
                        if "regression" not in var_experiment_results: continue # not enough points for regression
                        data[var_changing].append(var_value)
                        data["pearson_corr"].append(var_experiment_results["regression"]["pearson_corr"])

                        plt.scatter(
                            var_value,
                            var_experiment_results["regression"]["pearson_corr"],
                            color=color_map[dfa_detail],
                            alpha=0.8
                        )
                    
                    # Convert lists into a Pandas DataFrame for easier handling
                    df_data = pd.DataFrame(data)

                    # Plot scatter with unique color for `dfa_detail`
                    label_str = f"{idx_to_var[other_var_idx][1]}={dfa_detail[other_var_idx]}"
                    plt.plot(
                        df_data[var_changing],
                        df_data["pearson_corr"],
                        color=color_map[dfa_detail],
                        label=label_str,
                        alpha=0.8
                    )

                # Finalize and save the plot
                plt.ylim((0,1))
                plt.xlabel(f"DFA Complexity ({var_changing})")
                plt.ylabel("Pearson correlation (Performance vs ttoks)")
                plt.title(f"Pearson correlation (Performance vs ttoks) vs. {var_changing} (hold: {hold_var}={hold_var_val})")
                plt.legend(loc="best", fancybox=True, shadow=True, title="DFA (k, t, N)")
                plt.grid(True, linestyle="--", alpha=0.6)
                os.makedirs(os.path.join(foldername, "meta_plots"), exist_ok=True)
                plt.savefig(os.path.join(foldername, "meta_plots", f"corr_{var_changing}_hold{hold_var}{hold_var_val}_{modelname}.png"))
                plt.clf()

def meta_plot_diminish(stats_dict, modelname, all_var_vals):
    idx_to_var = [(0, "k"), (1, "t"), (2, "N")]
    colormap = get_cmap("tab10")  # Use a colormap with distinct colors

    for tuple_idx, var_changing in idx_to_var:
        for hold_idx, hold_var in idx_to_var:
            if hold_idx == tuple_idx: continue
            other_var_idx = len(idx_to_var) - hold_idx - tuple_idx

            for hold_var_val in all_var_vals[hold_idx]:
                dfas_to_include = [dfa_detail for dfa_detail in stats_dict.keys() if (dfa_detail[hold_idx] == int(hold_var_val)) and (dfa_detail[tuple_idx] is None)]
                dfas_to_include = sorted(dfas_to_include, key=lambda ent: ent[other_var_idx])
                color_map = {dfa_detail: colormap(i) for i, dfa_detail in enumerate(dfas_to_include)}

                # GRAPH 2: DFA Complexity vs. No. of Tokens Start Getting Diminishing Returns
                plt.figure(figsize=(10, 6))
                used_factor_values = set()
                for dfa_detail in dfas_to_include:
                    experiment_results = stats_dict[dfa_detail]
                    data = {
                        var_changing: [],
                        "peak_ttoks": [],
                    }
                    k = int(dfa_detail[0]) if dfa_detail[0] is not None else None
                    for var_value, var_experiment_results in experiment_results.items():
                        if k is None: k = var_value
                        if var_experiment_results["peak"] <= 1./k: continue
                        data[var_changing].append(var_value)
                        used_factor_values.add(var_value)
                        data["peak_ttoks"].append(var_experiment_results["peak"])

                        plt.scatter(
                            var_value,
                            var_experiment_results["peak"],
                            color=color_map[dfa_detail],
                            alpha=0.8
                        )
                    
                    # Convert lists into a Pandas DataFrame for easier handling
                    df_data = pd.DataFrame(data)

                    # Plot scatter with unique color for `dfa_detail`
                    label_str = f"{idx_to_var[other_var_idx][1]}={dfa_detail[other_var_idx]}"
                    plt.plot(
                        df_data[var_changing],
                        df_data["peak_ttoks"],
                        color=color_map[dfa_detail],
                        label=label_str,
                        alpha=0.8
                    )
                if len(used_factor_values) == 0: continue

                # Finalize and save the plot
                plt.ylim(bottom=0)
                plt.xlabel(f"DFA Complexity ({var_changing})")
                plt.ylabel("Tokens Where Returns Diminish (Peak Tokens)")
                plt.title(f"Diminishing Returns vs. {var_changing} (hold: {hold_var}={hold_var_val}) ({modelname})")
                plt.legend(loc="best", fancybox=True, shadow=True, title="DFA (k, t, N)")
                plt.grid(True, linestyle="--", alpha=0.6)
                os.makedirs(os.path.join(foldername, "meta_plots"), exist_ok=True)
                used_factor_values = ''.join(f"{fv}" for fv in sorted(list(used_factor_values)))
                plt.savefig(os.path.join(foldername, "meta_plots", f"diminish_{var_changing}{used_factor_values}_hold{hold_var}{hold_var_val}_{modelname}.png"))
                plt.clf()

def meta_plot_slopes(stats_dict, modelname, all_var_vals):
    idx_to_var = [(0, "k"), (1, "t"), (2, "N")]

    for tuple_idx, var_changing in idx_to_var:
        for hold_idx, hold_var in idx_to_var:
            if hold_idx == tuple_idx: continue
            other_var_idx = len(idx_to_var) - hold_idx - tuple_idx

            for hold_var_val in all_var_vals[hold_idx]:
                dfas_to_include = [dfa_detail for dfa_detail in stats_dict.keys() if (dfa_detail[hold_idx] == int(hold_var_val)) and (dfa_detail[tuple_idx] is None)]
                dfas_to_include = sorted(dfas_to_include, key=lambda ent: ent[other_var_idx])
                color_map = {dfa_detail: colormap(i) for i, dfa_detail in enumerate(dfas_to_include)}
                
                # GRAPH 1: DFA Complexity vs. slope of ttoks & performance regression line
                plt.figure(figsize=(10, 6))
                used_factor_values = set()
                for dfa_detail in dfas_to_include:
                    experiment_results = stats_dict[dfa_detail]
                    data = {
                        var_changing: [],
                        "slope": [],
                    }

                    for var_value, var_experiment_results in experiment_results.items():
                        if "regression" not in var_experiment_results: continue # not enough points for regression
                        data[var_changing].append(var_value)
                        used_factor_values.add(var_value)
                        data["slope"].append(var_experiment_results["regression"]["weights"])

                        plt.scatter(
                            var_value,
                            var_experiment_results["regression"]["weights"],
                            color=color_map[dfa_detail],
                            alpha=0.8
                        )
                    
                    # Convert lists into a Pandas DataFrame for easier handling
                    df_data = pd.DataFrame(data)

                    # Plot scatter with unique color for `dfa_detail`
                    label_str = f"{idx_to_var[other_var_idx][1]}={dfa_detail[other_var_idx]}"
                    plt.plot(
                        df_data[var_changing],
                        df_data["slope"],
                        color=color_map[dfa_detail],
                        label=label_str,
                        alpha=0.8
                    )
                if len(used_factor_values) == 0: continue
                # Finalize and save the plot
                plt.xlabel(f"DFA Complexity ({var_changing})")
                plt.ylabel("LR Slope (Performance vs ttoks)")
                plt.title(f"LR Slope (Performance vs ttoks) vs. {var_changing} (hold: {hold_var}={hold_var_val})")
                plt.legend(loc="best", fancybox=True, shadow=True, title="DFA (k, t, N)")
                plt.grid(True, linestyle="--", alpha=0.6)
                os.makedirs(os.path.join(foldername, "meta_plots"), exist_ok=True)
                used_factor_values = ''.join(f"{fv}" for fv in sorted(list(used_factor_values)))
                plt.savefig(os.path.join(foldername, "meta_plots", f"lrslope_{var_changing}{used_factor_values}_hold{hold_var}{hold_var_val}_{modelname}.png"))
                plt.clf()

def plot_correctness_by_N_isolate_factor(df, k, t, modelname):
    assert sum((factor is None for factor in (k, t))) == 1, f"{(k, t)} one must be None"
    # Filter the data for the specific model, k, t, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["k"] == k) if k is not None else True)
        & ((df["t"] == t) if t is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if t is None:
        isolated_factor = "t"
    elif k is None:
        isolated_factor = "k"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: {(k, t, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    def exponential_decay(x, a, b): 
       return a * np.exp(-b * x)

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for factor_value in sorted(filtered_data[isolated_factor].unique().astype(int)):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("N")
            ["Correct?"].mean()
        )
        if len(performance.values) == 0: continue
        used_vals.append(factor_value)
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            label=f"{isolated_factor}={factor_value}",
        )
        initial_a_guess = 1.0
        initial_b_guess = performance.values[0]

        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for N in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["N"] == str(N)]["Correct?"]
            if sample.empty:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                ci = np.percentile(np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), [2.5, 97.5])
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

        # Plot confidence intervals as a shaded region
        plt.fill_between(
            performance.index.astype(int),
            ci_lower,
            ci_upper,
            color=rgba_color,
            alpha=color_intensity,
        )

        # curve fit
        popt, pcov = curve_fit(exponential_decay, performance.index.astype(int).tolist(), performance.values, p0=[initial_a_guess, initial_b_guess])

        # overlay fitted exponential decay on plot.
        fitted_values = exponential_decay(performance.index.astype(int), *popt)
        plt.plot(
            performance.index.astype(int),
            fitted_values,
            linestyle="--",
            color="black",
            label=f"Fitted Curve ({isolated_factor}={factor_value}): d={popt[1]:.2f}",
            alpha=color_intensity
        )

    if not len(used_vals): return
    # Add random guessing baseline (1/k)
    plt.axhline(
        y = compute_random(k),
        color='gray',
        linestyle='--',
        label=f'Random Guessing (1/{k})',
    )
        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("N")
    plt.title(
        f"Correctness vs. N ({k, t, modelname} Buckets={n_buckets})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_N_"
    if k is None:
        filename += f"t{t}_{modelname}.png"
    elif t is None:
        filename += f"k{k}_{modelname}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_k_isolate_factor(df, t, N, modelname):
    assert sum((factor is None for factor in (t, N))) == 1, f"{(t, N)} one must be None"
    # Filter the data for the specific model, k, t, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N"] == N) if N is not None else True)
        & ((df["t"] == t) if t is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if t is None:
        isolated_factor = "t"
    elif N is None:
        isolated_factor = "N"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: (t,N,M) {(t, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for factor_value in sorted(filtered_data[isolated_factor].unique().astype(int)):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]
        filtered_data_factor["k"] = filtered_data_factor["k"].astype(int)

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("k")
            ["Correct?"].mean()
        )
        if len(performance.values) == 0: continue
        performance = performance.sort_index()
        used_vals.append(factor_value)
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            label=f"{isolated_factor}={factor_value}",
        )

        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for k in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["k"] == k]["Correct?"]
            if sample.empty:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                ci = np.percentile(np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), [2.5, 97.5])
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

        # Plot confidence intervals as a shaded region
        plt.fill_between(
            performance.index.astype(int),
            ci_lower,
            ci_upper,
            color=rgba_color,
            alpha=color_intensity,
        )

    if not len(used_vals): return

    # Add random guessing baseline (1/k)
    if not filtered_data["k"].empty:
        max_k = filtered_data["k"].astype(int).max()
        k_values = np.arange(1, max_k + 1)
        baseline = [compute_random(k) for k in k_values]
        plt.plot(
            k_values,
            baseline,
            color='gray',
            linestyle='--',
            label='Random Guessing (1/k)',
        )
        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("k")
    plt.title(
        f"Correctness vs. k ({t, N, modelname} Buckets={n_buckets})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_k_"
    if N is None:
        filename += f"t{t}_{modelname}.png"
    elif t is None:
        filename += f"N{N}_{modelname}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

if __name__ == "__main__":
    args = get_args()
    if args.delete_old and os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername, exist_ok=True)
    method_constraint = "_none_"
    if args.all_methods: method_constraint = None
    df, all_var_vals = load_data(args.output_folder, args.k_vals, args.t_vals, args.N_vals, target_methodname=method_constraint)

    plot_requested_vs_generated(df)

    k_vals = args.k_vals if args.k_vals is not None else set(df["k"])
    t_vals = args.t_vals if args.t_vals is not None else set(df["t"])
    N_vals = args.N_vals if args.N_vals is not None else set(df["N"])
    for modelname in args.models:
        numeric_stats = {}
        for t in t_vals:
            for k in k_vals:
                if int(t) > int(k):
                    continue
                # scatter_ttoks_by_inst_complexity(modelname, k, t)
                if "N" in args.get_isolated:
                    plot_correctness_by_ttoks_isolate_factor(df, k, t, None, modelname, numeric_stats, N_vals)
                
                if args.all_methods: 
                    for N in N_vals:
                        plot_correctness_by_ttoks_across_method(df, k, t, N, modelname)

            # plot_correctness_by_N_isolate_factor(df, None, t, modelname)
            # plot_correctness_by_k_isolate_factor(df, t, None, modelname)
            
            for N in N_vals:
                if "k" in args.get_isolated:
                    plot_correctness_by_ttoks_isolate_factor(df, None, t, N, modelname, numeric_stats, k_vals)
                # scatter_ttoks_by_complexity(modelname, N, t)
            
        for k in k_vals:
            plot_correctness_by_N_isolate_factor(df, k, None, modelname)

            for N in N_vals:
                if "t" in args.get_isolated:
                    plot_correctness_by_ttoks_isolate_factor(df, k, None, N, modelname, numeric_stats, t_vals)

        meta_plot_diminish(numeric_stats, modelname, all_var_vals)

        # meta_plot_corr(numeric_stats, modelname, all_var_vals)
        # meta_plot_slopes(numeric_stats, modelname, all_var_vals)
        # numeric_stats = {str(key): value for key, value in numeric_stats.items()}
        # with open(f"numeric_stats_{modelname}_{os.path.basename(os.path.dirname(foldername.rstrip('/')))}.json", "w") as wf: json.dump(numeric_stats, wf, indent=4)

        for N in N_vals:
            plot_correctness_by_k_isolate_factor(df, None, N, modelname)

    if "Model" in args.get_isolated:
        for k in k_vals:
            for N in N_vals:
                for t in t_vals:
                    plot_correctness_by_ttoks_isolate_factor(df, k, t, N, None, {}, args.models)
