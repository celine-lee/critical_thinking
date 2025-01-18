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

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_buckets", type=int, default=3)
    parser.add_argument("--num_gens", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--get_isolated", nargs='+', default=['k', 'm', 'N'])
    parser.add_argument("--k_vals", nargs='+', default=None)
    parser.add_argument("--m_vals", nargs='+', default=None)
    parser.add_argument("--N_vals", nargs='+', default=None)
    parser.add_argument("--models", nargs='+', default=['3.1-8B'])
    parser.add_argument("--delete_old", action="store_true")
    args = parser.parse_args()
    foldername = os.path.join(f"{args.output_folder.rstrip('/')}_graphs_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}")
    temperature = args.temperature
    num_beams = args.num_beams
    num_gens = args.num_gens
    n_buckets = args.n_buckets

    if "even_odd" in args.output_folder: compute_random = lambda k: 0.5
    elif "array_i" in args.output_folder: compute_random = lambda k: 1/int(k)
    return args


model_colors = {
    "3.1-8B": "purple",
    "Qwen2.5-32B": "blue",
    "Qwen2.5-14B": "brown",
    "Qwen2.5-7B": "yellow",
    "Mistral-7B": "red",
    "OLMo-2-1124-13B": "green",
    "OLMo-2-1124-7B": "black",
    "Ministral-8B": "orange",
    "gemma-2-9b": "pink",
}

colormap = get_cmap("tab10")  # Use a colormap with distinct colors

def load_data(data_folder, k_vals, m_vals, N_vals, skip_nulls=True):
    ks = []
    ms = []
    Ns = []
    models = []
    gen_toks = []
    len_gen_no_digits = []
    correct = []
    num_wraps = []

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(data_folder, f"k*")):
        parsed_experimentname = re.search(rf"k(\d+)_m(\d+)_N(\d+)", subfolder)
        if parsed_experimentname is None:
            continue
        k = parsed_experimentname.group(1)
        m = parsed_experimentname.group(2)
        N = parsed_experimentname.group(3)

        if k not in k_vals: continue
        if m not in m_vals: continue
        if N not in N_vals: continue
        for experiment_file in glob.glob(os.path.join(subfolder, "*")):
            if f"T{temperature}" not in experiment_file:
                continue
            if re.search(r'_B\d+_S\d+', experiment_file):
                if f"_B{num_beams}_S{num_gens}.json" not in experiment_file: 
                    continue
            elif temperature == 0.0: 
                assert num_beams == 1 and num_gens == 1
            modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
            if not any(model_str in modelname for model_str in model_colors):
                continue
            
            results = json.load(open(experiment_file))
            results = [res for res in results if res["pred_answer"]]

            ks.extend([k for _ in results])
            ms.extend([m for _ in results])
            Ns.extend([N for _ in results])
            models.extend([modelname for _ in results])
            gen_toks.extend([ex["generated_tokens"] for ex in results])
            len_gen_no_digits.extend([len(re.sub(r'\d', '', ex["model_generation"])) for ex in results])
            correct.extend([ex["correct"] for ex in results])
            num_wraps.extend(ex["num_wraps"] for ex in results)

    data = {
        "k": ks,
        "m": ms,
        "N": Ns,
        "Model": models,
        "No gen toks": gen_toks,
        "Len gen no digits": len_gen_no_digits,
        "Correct?": correct,
        "No wraparound": num_wraps,
    }
    all_var_vals = [set(ks), set(ms), set(Ns)]

    # Create a DataFrame for the new data
    df = pd.DataFrame(data)

    return df, all_var_vals

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

def plot_gen_by_factor(df, k, m, N, gen_factor="No gen toks"):
    assert sum((factor is None for factor in (k, m, N))) >= 1, f"{(k, m, N)} at least one must be None"

    if k is None:
        isolated_factor = "k"
        plot_multiple_of = "m"
    elif m is None:
        isolated_factor = "m"
        plot_multiple_of = "k"
    elif N is None:
        isolated_factor = "N"
        plot_multiple_of = "m"
        

    for modelname in df["Model"].unique():
        base_color = model_colors.get(modelname, "blue")
        # substrings... TODO

        # Filter the data for the specific model, k, m, N
        filtered_data = df[
            df["Model"].str.contains(modelname) 
            & ((df["k"] == k) if k is not None and plot_multiple_of != "k" else True)
            & ((df["m"] == m) if m is not None and plot_multiple_of != "m" else True)
            & ((df["N"] == N) if N is not None and plot_multiple_of != "N" else True)
        ]


        # Ensure there is data to plot
        if filtered_data.empty:
            print(f"No examples found for: {(k, m, N, modelname)}.")
            continue

        plt.figure(figsize=(12, 6))
        lines_plotted = set()

        line_labe_values = sorted(filtered_data[plot_multiple_of].unique(), key=int)
        max_line_label_value = int(line_labe_values[-1])
        for line_label in line_labe_values:
            line_data = filtered_data[filtered_data[plot_multiple_of] == line_label]
            factor_values = sorted(line_data[isolated_factor].unique(), key=int)
            if len(factor_values) == 0: continue

            means = []
            lower_bounds = []
            upper_bounds = []

            for val in factor_values:
                sub_df = line_data[line_data[isolated_factor] == val]
                gen_toks = sub_df[gen_factor]
                mean = gen_toks.mean()
                std_err = stats.sem(gen_toks)
                conf_int = std_err * stats.t.ppf((1 + 0.95) / 2., len(gen_toks) - 1)

                means.append(mean)
                lower_bounds.append(mean - conf_int)
                upper_bounds.append(mean + conf_int)

            lines_plotted.add(line_label)

            color_intensity = int(line_label) / (max_line_label_value + 1)
            rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
            plt.plot(factor_values, means, label=f"{plot_multiple_of}={line_label}", color=rgba_color)
            plt.fill_between(factor_values, lower_bounds, upper_bounds, color=rgba_color, alpha=color_intensity)

        # Customize plot labels and legend
        plt.ylabel(f"Average {gen_factor}")
        plt.xlabel(isolated_factor)
        plt.title(
            f"Average {gen_factor} vs. {isolated_factor} ({k, m, N, modelname})"
        )
        plt.legend(loc="best", fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save and clear the figure
        filename = f"{gen_factor}_{isolated_factor}{''.join(sorted(list(lines_plotted)))}_"
        filename += f"k{k}_" if k is not None and plot_multiple_of != 'k' else ""
        filename += f"m{m}_" if m is not None and plot_multiple_of != 'm' else ""
        filename += f"N{N}_" if N is not None and plot_multiple_of != 'N' else ""
        filename += f"{modelname}.png"
        plt.savefig(
            os.path.join(foldername, filename)
        )
        plt.clf()

def plot_correctness_by_ttoks_isolate_factor(df, k, m, N, modelname, factor_vals):
    assert sum((factor is None for factor in (k, m, N, modelname))) == 1, f"{(k, m, N, modelname)} one must be None"
    # Filter the data for the specific model, k, m, N, modelname
    filtered_data = df[
        (df["Model"].str.contains(modelname) if modelname is not None else True)
        & ((df["k"] == k) if k is not None else True)
        & ((df["m"] == m) if m is not None else True)
        & ((df["N"] == N) if N is not None else True)
    ]

    if m is None:
        isolated_factor = "m"
    elif k is None:
        isolated_factor = "k"
    elif N is None:
        isolated_factor = "N"
    elif modelname is None: 
        isolated_factor = "Model"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: {(k, m, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))
    if isolated_factor == "Model":
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=str)
    else:
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=int)
        max_factor = int(factor_values[-1])

    used_factor_values = []
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


            # Overlay the regression line
            plt.plot(
                filtered_peak_range, predictions, linestyle="--", color=rgba_color
            )
        
        # Plot the average correctness for each model size and method
        plt.plot(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"],
            color=rgba_color,
            label=f"{isolated_factor}={factor_value}",
        )

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
        f"Average Correctness vs. No. of Generated Tokens ({k, m, N, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"{isolated_factor}{''.join(used_factor_values)}_"
    if k is None:
        filename += f"m{m}_N{N}_{modelname}.png"
    elif m is None:
        filename += f"k{k}_N{N}_{modelname}.png"
    elif N is None:
        filename += f"k{k}_m{m}_{modelname}.png"
    elif modelname is None:
        filename = f"byModel_k{k}_m{m}_N{N}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_wraps(df, m, modelname):
    # Filter the data for the specific model, m
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & (df["m"] == m)
    ]
    base_color = model_colors.get(modelname, "blue")

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: {(m, modelname)}.")
        return

    plt.figure(figsize=(12, 6))
    k_vals = sorted(df["k"].unique().astype(int))
    max_factor = k_vals[-1]
    used_vals = set()
    for k in k_vals:
        filtered_data_factor = filtered_data[filtered_data["k"] == k]
        bucket_avg, filtered_data_factor = calculate_buckets_samesize(filtered_data_factor)
        if filtered_data_factor is None: continue
        if len(bucket_avg) == 0: continue

        # Normalize the intensity of the color based on t
        color_intensity = int(k) / (max_factor + 1) 
        filtered_data_factor["No wraparound"] = filtered_data_factor["No wraparound"].astype(int)

        rgba_color = mcolors.to_rgba(base_color, alpha=0.8)
            
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("No wraparound")
            ["Correct?"].mean()
        )
        if len(performance.values) == 0: return
        performance = performance.sort_index()
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            marker="."
        )

        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for num_wraps in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["No wraparound"] == num_wraps]["Correct?"]
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
        used_vals.add(k)
    if len(used_vals) == 0: return
        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("No. wraparounds")
    plt.title(
        f"Correctness vs. num_wraps ({m,modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_wraps_m{m}_{modelname}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_N_isolate_factor(df, df_nocot, k, m, modelname):
    assert sum((factor is None for factor in (k, m))) == 1, f"{(k, m)} one must be None"
    # Filter the data for the specific model, k, m, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["k"] == k) if k is not None else True)
        & ((df["m"] == m) if m is not None else True)
    ]
    filtered_data_nocot = df_nocot[
        (df_nocot["Model"].str.contains(modelname) if len(df_nocot) > 0 else True)
        & ((df_nocot["k"] == k) if k is not None else True)
        & ((df_nocot["m"] == m) if m is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if m is None:
        isolated_factor = "m"
    elif k is None:
        isolated_factor = "k"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: {(k, m, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    def exponential_decay(x, a, b): 
       return a * np.exp(-b * x)

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]
        filtered_data_factor_nocot = filtered_data_nocot[filtered_data_nocot[isolated_factor] == str(factor_value)]

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("N")
            ["Correct?"].mean()
        )
        if len(performance.values) <= 1: continue
        used_vals.append(factor_value)
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            label=f"{isolated_factor}={factor_value}",
            marker="."
        )
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
        initial_a_guess = 1.0
        initial_b_guess = performance.values[0]
        popt, pcov = curve_fit(exponential_decay, performance.index.astype(int).tolist(), performance.values, p0=[initial_a_guess, initial_b_guess])

        # overlay fitted exponential decay on plot.
        fitted_values = exponential_decay(performance.index.astype(int), *popt)
        plt.plot(
            performance.index.astype(int),
            fitted_values,
            linestyle="--",
            color="black",
            label=f"Fitted Curve ({isolated_factor}={factor_value}): d={popt[1]:.2f}",
            alpha=color_intensity,
            marker="."
        )

        # plot the  noCOT
        performance_nocot = (
            filtered_data_factor_nocot.groupby("N")["Correct?"].mean()
        ).sort_index()

        # Convert index to integer and sort for plotting

        plt.plot(
            performance_nocot.index.astype(int),
            performance_nocot.values,
            color=colormap(cmap_idx),
            label=f"{isolated_factor}={factor_value} (no cot)",
            marker="."
        )

        # Initialize confidence interval lists
        ci_lower = []
        ci_upper = []

        # Generate confidence intervals
        for N in performance_nocot.index:
            sample = filtered_data_factor_nocot[filtered_data_factor_nocot["N"] == str(N)]["Correct?"]
            if sample.empty:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                ci = np.percentile(
                    np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), 
                    [2.5, 97.5]
                )
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

        # Plot confidence intervals
        plt.fill_between(
            performance_nocot.index.astype(int),
            ci_lower,
            ci_upper,
            color=colormap(cmap_idx),
            alpha=color_intensity,
        )



    if not len(used_vals): return
    # Add random guessing baseline (1/k) TODO
        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("N")
    plt.title(
        f"Correctness vs. N ({k, m, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_N_"
    if k is None:
        filename += f"m{m}_{modelname}.png"
    elif m is None:
        filename += f"k{k}_{modelname}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_k_isolate_factor(df, df_nocot, m, N, modelname):
    assert sum((factor is None for factor in (m, N))) == 1, f"{(m, N)} one must be None"
    # Filter the data for the specific model, k, m, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N"] == N) if N is not None else True)
        & ((df["m"] == m) if m is not None else True)
    ]
    filtered_data_nocot = df_nocot[
        (df_nocot["Model"].str.contains(modelname) if len(df_nocot) > 0 else True)
        & ((df_nocot["N"] == N) if N is not None else True)
        & ((df_nocot["m"] == m) if m is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if m is None:
        isolated_factor = "m"
    elif N is None:
        isolated_factor = "N"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: (m,N) {(m, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]
        filtered_data_factor["k"] = filtered_data_factor["k"].astype(int)
        filtered_data_factor_nocot = filtered_data_nocot[filtered_data_nocot[isolated_factor] == str(factor_value)]
        filtered_data_factor_nocot["k"] = filtered_data_factor_nocot["k"].astype(int)

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
            marker="."
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

        # plot the  noCOT
        performance_nocot = (
            filtered_data_factor_nocot.groupby("k")["Correct?"].mean()
        ).sort_index()

        # Convert index to integer and sort for plotting
        # sorted_index = performance_nocot.index.astype(int)
        # performance_nocot = performance_nocot.loc[sorted_index].sort_index()

        plt.plot(
            performance_nocot.index.astype(int),
            performance_nocot.values,
            color=colormap(cmap_idx),
            label=f"{isolated_factor}={factor_value} (no cot)",
            marker="."
        )

        # Initialize confidence interval lists
        ci_lower = []
        ci_upper = []

        # Generate confidence intervals
        for k in performance_nocot.index:
            sample = filtered_data_factor_nocot[filtered_data_factor_nocot["k"] == str(k)]["Correct?"]
            if sample.empty:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                ci = np.percentile(
                    np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), 
                    [2.5, 97.5]
                )
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

        # Plot confidence intervals
        plt.fill_between(
            performance_nocot.index.astype(int),
            ci_lower,
            ci_upper,
            color=colormap(cmap_idx),
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
        f"Correctness vs. k ({m, N, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_k_"
    if N is None:
        filename += f"m{m}_{modelname}.png"
    elif m is None:
        filename += f"N{N}_{modelname}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_m_isolate_factor(df, df_nocot, k, N, modelname):
    assert sum((factor is None for factor in (k, N))) == 1, f"{(k, N)} one must be None"
    # Filter the data for the specific model, k, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N"] == N) if N is not None else True)
        & ((df["k"] == k) if k is not None else True)
    ]
    filtered_data_nocot = df_nocot[
        (df_nocot["Model"].str.contains(modelname) if len(df_nocot) > 0 else True)
        & ((df_nocot["N"] == N) if N is not None else True)
        & ((df_nocot["k"] == k) if k is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if k is None:
        isolated_factor = "k"
    elif N is None:
        isolated_factor = "N"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: (k,N) {(k, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]
        filtered_data_factor["m"] = filtered_data_factor["m"].astype(int)
        filtered_data_factor_nocot = filtered_data_nocot[filtered_data_nocot[isolated_factor] == str(factor_value)]
        filtered_data_factor_nocot["m"] = filtered_data_factor_nocot["m"].astype(int)

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("m")
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
            marker="."
        )

        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for m in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["m"] == m]["Correct?"]
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

        # plot the  noCOT
        performance_nocot = (
            filtered_data_factor_nocot.groupby("m")
            ["Correct?"].mean()
        ).sort_index()

        # sorted_index = performance_nocot.index.astype(int)
        # performance_nocot = performance_nocot.loc[sorted_index].sort_index()

        plt.plot(
            performance_nocot.index.astype(int),
            performance_nocot.values,
            color=colormap(cmap_idx),
            label=f"{isolated_factor}={factor_value} (no cot)",
            marker="."
        )

        # Initialize confidence interval lists
        ci_lower = []
        ci_upper = []

        # Generate confidence intervals
        for m in performance_nocot.index:
            sample = filtered_data_factor_nocot[filtered_data_factor_nocot["m"] == str(m)]["Correct?"]
            if sample.empty:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                ci = np.percentile(
                    np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), 
                    [2.5, 97.5]
                )
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

        # Plot confidence intervals
        plt.fill_between(
            performance_nocot.index.astype(int),
            ci_lower,
            ci_upper,
            color=colormap(cmap_idx),
            alpha=color_intensity,
        )
    if not len(used_vals): return

    # Add random guessing baseline (1/k)
    if k:
        plt.axhline(
            y = compute_random(k),
            color='gray',
            linestyle='--',
            label=f'Random Guessing (1/k)',
        )

        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("m")
    plt.title(
        f"Correctness vs. m ({k, N, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_m_"
    if N is None:
        filename += f"k{k}"
    elif k is None:
        filename += f"N{N}"
    filename += f"_{modelname}.png"
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
    df, all_var_vals = load_data(args.output_folder, args.k_vals, args.m_vals, args.N_vals)
    df_nocot, all_var_vals_nocot = load_data(args.output_folder+"_nocot", args.k_vals, args.m_vals, args.N_vals)

    k_vals = args.k_vals if args.k_vals is not None else all_var_vals[0]
    m_vals = args.m_vals if args.m_vals is not None else all_var_vals[1]
    N_vals = args.N_vals if args.N_vals is not None else all_var_vals[2]
    for modelname in args.models:
        for m in m_vals:
            # plot_correctness_by_wraps(df, m, modelname)
            # for k in k_vals:
            #     if "N" in args.get_isolated:
            #         plot_correctness_by_ttoks_isolate_factor(df, k, m, None, modelname, N_vals)
                
            plot_correctness_by_N_isolate_factor(df, df_nocot, None, m, modelname)
            plot_correctness_by_k_isolate_factor(df, df_nocot,  m, None, modelname)
            
            for N in N_vals:
                if "k" in args.get_isolated:
                    plot_correctness_by_ttoks_isolate_factor(df, None, m, N, modelname, k_vals)
            
        for k in k_vals:
            plot_correctness_by_N_isolate_factor(df, df_nocot,  k, None, modelname)
            plot_correctness_by_m_isolate_factor(df, df_nocot,  k, None, modelname)

            for N in N_vals:
                if "m" in args.get_isolated:
                    plot_correctness_by_ttoks_isolate_factor(df, k, None, N, modelname, m_vals)

        for N in N_vals:
            plot_correctness_by_k_isolate_factor(df, df_nocot,  None, N, modelname)
            plot_correctness_by_m_isolate_factor(df, df_nocot,  None, N, modelname)

    for k in k_vals:
        plot_gen_by_factor(df, k, None, None, "No gen toks")
        plot_gen_by_factor(df, k, None, None, "Len gen no digits")

    for m in m_vals:
        plot_gen_by_factor(df, None, m, None, "No gen toks")
        plot_gen_by_factor(df, None, m, None, "Len gen no digits")

    for N in N_vals:
        plot_gen_by_factor(df, None, None, N, "No gen toks")
        plot_gen_by_factor(df, None, None, N, "Len gen no digits")
        for m in m_vals:
            if "Model" in args.get_isolated:
                for k in k_vals:
                    plot_correctness_by_ttoks_isolate_factor(df, k, m, N, None, args.models)
