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
from collections import defaultdict
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

import argparse

global temperature
global num_beams
global num_gens
global foldername
global n_buckets
global n_buckets_dfa_properties

def get_args():
    global temperature
    global num_beams
    global num_gens
    global foldername
    global n_buckets
    global n_buckets_dfa_properties
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_buckets", type=int, default=3)
    parser.add_argument("--n_buckets_dfa_properties", type=int, default=5)
    parser.add_argument("--num_gens", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--models", nargs='+', default=['3.1-8B'])
    parser.add_argument("--delete_old", action="store_true")
    args = parser.parse_args()
    foldername = os.path.join(f"{args.output_folder.rstrip('/')}_graphs_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}")
    temperature = args.temperature
    num_beams = args.num_beams
    num_gens = args.num_gens
    n_buckets = args.n_buckets
    n_buckets_dfa_properties = args.n_buckets_dfa_properties

    return args


model_colors = {
    "3.1-8B": "purple",
    "Qwen2.5-32B": "blue",
    "Qwen2.5-14B": "brown",
    "Qwen2.5-7B": "yellow",
    # "Mistral-7B": "red",
    "OLMo-2-1124-13B": "green",
    "OLMo-2-1124-7B": "black",
    "Ministral-8B": "orange",
    "gemma-2-9b": "pink",
}

colormap = get_cmap("tab10")  # Use a colormap with distinct colors

def map_to_closest(numbers, num_targets):
    # Reshape the data to fit the KMeans input
    numbers_array = np.array(numbers).reshape(-1, 1)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=min(len(numbers_array), num_targets), random_state=0)
    kmeans.fit(numbers_array)
    
    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_.flatten()
    targets = sorted(cluster_centers)
    mapped_numbers = []
    
    for num in numbers:
        # Calculate the absolute differences
        differences = [abs(num - target) for target in targets]
        # Find the target with the minimum difference
        closest_value = targets[differences.index(min(differences))]
        # Append the closest value
        mapped_numbers.append(closest_value)
    
    return mapped_numbers

def load_data(data_folder, modelnames):
    ks = []
    sparsities = []
    Ns = []
    models = []
    encoders = []
    tasks = []
    gen_toks = []
    correct = []

    # Load data from experiment files
    for task_folder in glob.glob(os.path.join(data_folder, "*")):
        taskname = os.path.basename(task_folder.rstrip("/"))
        for encoding_folder in glob.glob(os.path.join(task_folder, "*")):
            encoder = os.path.basename(encoding_folder.rstrip("/"))
            for dfa_folder in glob.glob(os.path.join(encoding_folder, "*")):
                parse_dfa = re.search(r'(\d+)_(\d+)', os.path.basename(dfa_folder.rstrip("/ ")))
                k = int(parse_dfa.group(1))
                s = int(parse_dfa.group(2))
                for experiment_file in glob.glob(os.path.join(dfa_folder, "*")):
                    if f"T{temperature}" not in experiment_file:
                        continue
                    if re.search(r'_B\d+_S\d+', experiment_file):
                        if f"_B{num_beams}_S{num_gens}.json" not in experiment_file: 
                            continue
                    elif temperature == 0.0: 
                        assert num_beams == 1 and num_gens == 1
                    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
                    if not any(model_str in modelname for model_str in modelnames):
                        continue
                    
                    results = json.load(open(experiment_file)).values()
                    results = [res for res in results if res["pred_answer"]]

                    ks.extend([k for res in results])
                    sparsities.extend([s for res in results])
                    # unmapped_sparsities = [int(100 * res['nedges'] / (res['nnodes'] * (res['nnodes'] - 1))) for res in results]
                    # sparsities.extend(map_to_closest(unmapped_sparsities, n_buckets_dfa_properties))
                    Ns.extend([res['lrun'] for res in results])
                    models.extend([modelname for _ in results])
                    tasks.extend([taskname for _ in results])
                    encoders.extend([encoder for _ in results])
                    gen_toks.extend([ex["generated_tokens"] for ex in results])
                    correct.extend([ex["correct"] for ex in results])

    data = {
        "k": ks,
        "s": sparsities,
        "N": Ns,
        "Model": models,
        "Encoder": encoders,
        "Task": tasks,
        "No gen toks": gen_toks,
        "Correct?": correct,
    }

    # Create a DataFrame for the new data
    df = pd.DataFrame(data)

    return df

def bucketize_values(sub_df, column):
    if len(sub_df) == 0: return None
    # Use qcut to create equal-sized buckets by count.
    sub_df[column] = sub_df[column].astype(int)
    if len(sub_df[column].unique()) < n_buckets:    
        sub_df.loc[:, f"{column} Bucket"] = pd.qcut(
            sub_df[column], q=len(sub_df[column].unique()), duplicates="drop"
        )
    else:
        sub_df.loc[:, f"{column} Bucket"] = pd.qcut(
            sub_df[column], q=n_buckets, duplicates="drop"
        )

    # Calculate the bucket center by averaging bin edges
    sub_df[f"{column} Bucket Center"] = sub_df[f"{column} Bucket"].apply(
        lambda x: int((x.left + x.right) / 2)
    ).astype(int)

    return sub_df

def average_correctness_by_buckets(sub_df, column, groupby_keys=["Model", "Task", "Encoder"]):
    sub_df = bucketize_values(sub_df, column)
    if sub_df is None: return None, None
    bucket_avg = (
        sub_df.groupby(groupby_keys + [f"{column} Bucket Center"], observed=True)["Correct?"].mean().reset_index()
    )

    bucket_avg["Correct?"] = bucket_avg["Correct?"].astype(float)
    sub_df = sub_df.merge(bucket_avg, on=groupby_keys, suffixes=('', '_mean'))

    return bucket_avg, sub_df

def plot_gen_by_factor(df, k, s, N, gen_factor="No gen toks"):
    assert sum((factor is None for factor in (k, s, N))) >= 1, f"{(k, s, N)} at least one must be None"

    if k is None:
        isolated_factor = "k"
        plot_multiple_of = "s"
    elif s is None:
        isolated_factor = "s"
        plot_multiple_of = "k"
    elif N is None:
        isolated_factor = "N Bucket Center"
        plot_multiple_of = "s"
        

    for modelname in df["Model"].unique():
        base_color = model_colors.get(modelname, "blue")
        # substrings... TODO

        # Filter the data for the specific model, k, s, N
        filtered_data = df[
            df["Model"].str.contains(modelname) 
            & ((df["k"] == k) if k is not None and plot_multiple_of != "k" else True)
            & ((df["s"] == s) if s is not None and plot_multiple_of != "s" else True)
            & ((df["N Bucket Center"] == N) if N is not None and plot_multiple_of != "N Bucket Center" else True)
        ]


        # Ensure there is data to plot
        if filtered_data.empty:
            print(f"No examples found for: {(k, s, N, modelname)}.")
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
            f"Average {gen_factor} vs. {isolated_factor[:1]} ({k, s, N, modelname})"
        )
        plt.legend(loc="best", fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save and clear the figure
        filename = f"{gen_factor}_{isolated_factor[:1]}{''.join(sorted(list(lines_plotted)))}_"
        filename += f"k{k}_" if k is not None and plot_multiple_of != 'k' else ""
        filename += f"s{s}_" if s is not None and plot_multiple_of != 'e' else ""
        filename += f"N{N}_" if N is not None and plot_multiple_of != 'N' else ""
        filename += f"{modelname}.png"
        plt.savefig(
            os.path.join(foldername, filename)
        )
        plt.clf()

def plot_correctness_by_ttoks(filtered_data, k, s, N, modelname, label, rgba_color, task_encoder_name, is_subplot=False):
    # plt.figure(figsize=(12, 6))
    bucket_avg, filtered_data = average_correctness_by_buckets(filtered_data, "No gen toks")
    if filtered_data is None: return False
    if len(bucket_avg) == 0: return False
    if len(bucket_avg) < 2: return
        
    # Find the index of the maximum value
    index_peak = np.argmax(bucket_avg["Correct?"])
    peak_ttoks = bucket_avg["No gen toks Bucket Center"][index_peak]
    best_performance = bucket_avg["Correct?"][index_peak]
    
    # Plot the average correctness for each model size and method
    plt.plot(
        bucket_avg["No gen toks Bucket Center"],
        bucket_avg["Correct?"],
        color=rgba_color,
        label=label,
    )

    sem_values = filtered_data.groupby("No gen toks Bucket Center", observed=True)["Correct?"].apply(lambda x: stats.sem(x) if len(x) > 1 else np.nan)
    # Calculate confidence intervals
    ci = sem_values * 1.96  # For 95% confidence
    ci = sem_values.reindex(bucket_avg["No gen toks Bucket Center"]).fillna(np.nan)
    if any(ci.isna()):
        return

    plt.fill_between(
        bucket_avg["No gen toks Bucket Center"],
        bucket_avg["Correct?"] - ci.values,
        bucket_avg["Correct?"] + ci.values,
        color=rgba_color,
    )

    # Place a dot at the maximum value
    plt.scatter(peak_ttoks, best_performance, color='red')
    
    if not is_subplot:
        # Customize plot labels and legend
        plt.xlim(xmin=0)
        plt.ylim(0, 1)
        plt.ylabel("Average Correctness")
        plt.xlabel("No. of Generated Tokens (Binned)")
        plt.title(
            f"Average Correctness vs. No. of Generated Tokens ({k, s, N, modelname})"
        )
        plt.legend(loc="best", fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save and clear the figure
        filename = f"k{k}_s{s}_N{N}_{modelname}.png"
        os.makedirs(os.path.join(foldername, task_encoder_name, "isolate_factor"), exist_ok=True)
        plt.savefig(
            os.path.join(foldername, "isolate_factor", filename)
        )
        plt.clf()
    return (peak_ttoks.item(), best_performance.item()) #, (best_performance - ci.values[index_peak]).item() > compute_random(k, s))

def plot_correctness_by_ttoks_isolate_factor(df, k, s, N, modelname, task_encoder_name):
    assert sum((factor is None for factor in (k, s, N, modelname))) == 1, f"{(k, s, N, modelname)} one must be None"
    # Filter the data for the specific model, k, s, N, modelname
    filtered_data = df[
        (df["Model"].str.contains(modelname) if modelname is not None else True)
        & ((df["k"] == k) if k is not None else True)
        & ((df["s"] == s) if s is not None else True)
        & ((df["N Bucket Center"] == N) if N is not None else True)
    ]

    if s is None:
        isolated_factor = "s"
    elif k is None:
        isolated_factor = "k"
    elif N is None:
        isolated_factor = "N Bucket Center"
    elif modelname is None: 
        isolated_factor = "Model"

    if filtered_data.empty:
        # print(f"No examples found for: {(k, s, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))
    if isolated_factor == "Model":
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=str)
    else:
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=int)
        base_color = model_colors.get(modelname, "blue")
        max_factor = int(factor_values[-1])

    factor_val_to_peak_ttoks = []
    used_vals = []
    # Iterate over unique t values
    for factor_value in factor_values:
        if isolated_factor == "Model":
            factor_filtered_data = filtered_data[isolated_factor].str.contains(factor_value)
            if factor_filtered_data.empty: continue
            base_color = model_colors.get(isolated_factor, "blue")
            rgba_color = mcolors.to_rgba(base_color, alpha=0.8)
            plot_results = plot_correctness_by_ttoks(factor_filtered_data, k, s, N, factor_value, rgba_color, task_encoder_name, is_subplot=True)
        else:
            factor_filtered_data = filtered_data[filtered_data[isolated_factor]==factor_value]
            if factor_filtered_data.empty: continue
            factor_value = int(factor_value)
            # Normalize the intensity of the color based on t
            color_intensity = factor_value / (max_factor + 1) 
            label = f"{isolated_factor[:1]}={factor_value}"
            rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
            if isolated_factor == "k":
                plot_results = plot_correctness_by_ttoks(factor_filtered_data, factor_value, s, N, modelname, label, rgba_color, task_encoder_name, is_subplot=True)
            elif isolated_factor == "s":
                plot_results = plot_correctness_by_ttoks(factor_filtered_data, k, factor_value, N, modelname, label, rgba_color, task_encoder_name, is_subplot=True)
            elif isolated_factor == "N Bucket Center":
                plot_results = plot_correctness_by_ttoks(factor_filtered_data, k, s, factor_value, modelname, label, rgba_color, task_encoder_name, is_subplot=True)
        if plot_results:
            used_vals.append(factor_value)
            (peak_ttoks, _) = plot_results
            # if task_doable:
            factor_val_to_peak_ttoks.append((factor_value, peak_ttoks))

    if len(factor_val_to_peak_ttoks) == 0: return
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Average Correctness")
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.title(
        f"Average Correctness vs. No. of Generated Tokens ({k, s, N, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"{isolated_factor[:1]}{''.join(str(uv) for uv in used_vals)}_"
    if k is None:
        filename += f"s{s}_N{N}_{modelname}.png"
    elif s is None:
        filename += f"k{k}_N{N}_{modelname}.png"
    elif N is None:
        filename += f"k{k}_s{s}_{modelname}.png"
    elif modelname is None:
        filename = f"byModel_k{k}_s{s}_N{N}.png"
    os.makedirs(os.path.join(foldername, task_encoder_name, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, task_encoder_name, "isolate_factor", filename)
    )
    plt.clf()

    return factor_val_to_peak_ttoks


def plot_correctness_by_N_isolate_factor(df, k, s, modelname, task_encoder_name):
    assert sum((factor is None for factor in (k, s))) == 1, f"{(k, s)} one must be None"
    # Filter the data for the specific model, k, s, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["k"] == k) if k is not None else True)
        & ((df["s"] == s) if s is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if s is None:
        isolated_factor = "s"
    elif k is None:
        isolated_factor = "k"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: {(k, s, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    def exponential_decay(x, a, b): 
       return a * np.exp(-b * x)

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("N Bucket Center")
            ["Correct?"].mean()
        )
        if len(performance.values) <= 1: continue
        used_vals.append(factor_value)
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            label=f"{isolated_factor[:1]}={factor_value}",
            marker="."
        )
        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for N in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["N Bucket Center"] == N]["Correct?"]
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
            label=f"Fitted Curve ({isolated_factor[:1]}={factor_value}): d={popt[1]:.2f}",
            alpha=color_intensity,
            marker="."
        )

    if not len(used_vals): return
    # Add random guessing baseline (1/k) TODO
        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("N Bucket Center")
    plt.title(
        f"Correctness vs. N ({k, s, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_N_"
    if k is None:
        filename += f"s{s}_{modelname}.png"
    elif s is None:
        filename += f"k{k}_{modelname}.png"
    os.makedirs(os.path.join(foldername, task_encoder_name, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, task_encoder_name, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_k_isolate_factor(df, s, N, modelname, task_encoder_name):
    assert sum((factor is None for factor in (s, N))) == 1, f"{(s, N)} one must be None"
    # Filter the data for the specific model, k, s, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N Bucket Center"] == N) if N is not None else True)
        & ((df["s"] == s) if s is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if s is None:
        isolated_factor = "s"
    elif N is None:
        isolated_factor = "N Bucket Center"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: (s,N) {(s, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]

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
            label=f"{isolated_factor[:1]}={factor_value}",
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


    if not len(used_vals): return

        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("k")
    plt.title(
        f"Correctness vs. k ({e, N, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_k_"
    if N is None:
        filename += f"s{s}_{modelname}.png"
    elif s is None:
        filename += f"N{N}_{modelname}.png"
    os.makedirs(os.path.join(foldername, task_encoder_name, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, task_encoder_name, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_s_isolate_factor(df, k, N, modelname, task_encoder_name):
    assert sum((factor is None for factor in (k, N))) == 1, f"{(k, N)} one must be None"
    # Filter the data for the specific model, k, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N Bucket Center"] == N) if N is not None else True)
        & ((df["k"] == k) if k is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if k is None:
        isolated_factor = "k"
    elif N is None:
        isolated_factor = "N Bucket Center"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: (k,N) {(k, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]
        filtered_data_factor["s"] = filtered_data_factor["s"].astype(int)

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("s")
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
            label=f"{isolated_factor[:1]}={factor_value}",
            marker="."
        )

        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for s in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["s"] == s]["Correct?"]
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


        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("s")
    plt.title(
        f"Correctness vs. s ({k, N, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_s_"
    if N is None:
        filename += f"k{k}"
    elif k is None:
        filename += f"N{N}"
    filename += f"_{modelname}.png"
    os.makedirs(os.path.join(foldername, task_encoder_name, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, task_encoder_name, "isolate_factor", filename)
    )
    plt.clf()

def plot_ptt_by_factor(factor_to_peak_ttoks, isolated_factor, task_encoder_name):
    all_factor_vals = []
    # all_normalized_avg_peak_tts = []
    all_peak_tts = []

    # Make one plot with and one without errbars
    errbars = []
    for modelname, factor_to_ptts in factor_to_peak_ttoks.items():
        base_color = model_colors.get(modelname, "blue")
        rgba_color = mcolors.to_rgba(base_color, alpha=0.9)

        fv_to_ptts_avged = {}
        for (_, isolated_factor_val_to_peak_tts) in factor_to_ptts.items():
            for (fv, ptt) in isolated_factor_val_to_peak_tts:
                if fv not in fv_to_ptts_avged:
                    fv_to_ptts_avged[fv] = []
                fv_to_ptts_avged[fv].append(ptt)
        if len(fv_to_ptts_avged) == 0:
            continue

        factor_vals = []
        avg_peak_tts = []
        ci_lower_bounds = []
        ci_upper_bounds = []

        # Calculate averages and confidence intervals
        for fv, ptts in fv_to_ptts_avged.items():
            avg_ptt = np.mean(ptts)
            factor_vals.append(fv)
            avg_peak_tts.append(avg_ptt)

            # Compute 95% confidence interval
            n = len(ptts)
            if n > 1:
                se = stats.sem(ptts)  # standard error
                h = se * stats.t.ppf((1 + 0.95) / 2., n-1)  # margin of error
                ci_lower_bounds.append(avg_ptt - h)
                ci_upper_bounds.append(avg_ptt + h)
            else:
                ci_lower_bounds.append(avg_ptt)
                ci_upper_bounds.append(avg_ptt)

        # Only plot if there are multiple for the model
        if len(avg_peak_tts) == 1: 
            continue

        # Normalize avg_peak_tts for the current model
        min_val = min(avg_peak_tts)
        max_val = max(avg_peak_tts)
        normalized_avg_peak_tts = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in avg_peak_tts]

        # Store the normalized values for plotting
        # normalized_values_per_model[modelname] = (factor_vals, normalized_avg_peak_tts, rgba_color)
        all_factor_vals.extend(factor_vals)
        all_peak_tts.extend(avg_peak_tts)
        # all_normalized_avg_peak_tts.extend(normalized_avg_peak_tts)

        plt.scatter(factor_vals, normalized_avg_peak_tts, 
                    marker='o', c=rgba_color, label=modelname)

        # Compute normalized error bars
        normalized_ci_lower = [(avg - min_val) / (max_val - min_val) if max_val != min_val else 0 for avg in ci_lower_bounds]
        normalized_ci_upper = [(avg - min_val) / (max_val - min_val) if max_val != min_val else 0 for avg in ci_upper_bounds]

        yerr_lower = np.maximum(0, np.subtract(normalized_avg_peak_tts, normalized_ci_lower))
        yerr_upper = np.maximum(0, np.subtract(normalized_ci_upper, normalized_avg_peak_tts))

        errbars.append(
            (list(factor_vals), list(normalized_avg_peak_tts), yerr_lower.copy(), yerr_upper.copy(), rgba_color, base_color, modelname)
        )
    if len(all_factor_vals) == 0: 
        return

    # See how well all collected points fit to the [0,1] line (it's normalized, remember)
    # slope, intercept, _, _, _ = stats.linregress(all_factor_vals, all_normalized_avg_peak_tts)
    slope = 1. / (max(all_factor_vals) - min(all_factor_vals))
    intercept = 1 - slope * max(all_factor_vals)

    # Generate x values for the target regression line
    x_vals = np.linspace(min(all_factor_vals), max(all_factor_vals), 100)
    y_vals = slope * x_vals + intercept

    # Calculate Mean Squared Error
    predicted_vals = slope * np.array(all_factor_vals) + intercept
    mse = np.mean((np.array(all_peak_tts) - predicted_vals) ** 2)

    # Plot the target regression line
    plt.plot(x_vals, y_vals, color='black', linestyle='--')

    # Annotate the MSE on the plot
    mse_annotation = f"MSE: {mse:.4f}"
    plt.text(0.05, 0.95, mse_annotation, transform=plt.gca().transAxes,
             fontsize=14, color='red', verticalalignment='top')

    # Finalize and save the plot
    plt.ylim(-1, 2)
    plt.xlabel(isolated_factor, fontsize=14)
    plt.ylabel("Normalized Tokens Where Returns Diminish (Peak Tokens)", fontsize=14)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    os.makedirs(os.path.join(foldername, task_encoder_name, "meta_plots"), exist_ok=True)
    plt.savefig(os.path.join(foldername, task_encoder_name, "meta_plots", f"diminish_{isolated_factor[:1]}.png"))

    for factor_vals, normalized_avg_peak_tts, yerr_lower, yerr_upper, rgba_color, base_color, modelname in errbars:
        # Plotting with confidence intervals (using normalized values)
        plt.errorbar(factor_vals, normalized_avg_peak_tts, 
                     yerr=[yerr_lower, yerr_upper],
                     fmt='o', color=rgba_color, ecolor=base_color, capsize=5)

    plt.savefig(os.path.join(foldername, task_encoder_name, "meta_plots", f"diminish_{isolated_factor[:1]}_errbars.png"))
    plt.clf()

if __name__ == "__main__":
    args = get_args()
    if args.delete_old and os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername, exist_ok=True)
    df = load_data(args.output_folder, args.models)
    df["s"] = df["s"].astype(int)
    df["k"] = df["k"].astype(int)
    # df_nocot, _ = load_data(args.output_folder+"_nocot", args.models)
    # df = pd.concat([df, df_nocot])

    tasks = df["Task"].unique()
    for task in tasks:
        for encoder in df[df["Task"] == task]["Encoder"].unique():

            N_to_peak_ttoks = {}
            k_to_peak_ttoks = {}
            s_to_peak_ttoks = {}
            for modelname in args.models:
                sub_df = df[
                    (df["Task"] == task) & (df["Encoder"] == encoder) & (df["Model"].str.contains(modelname))
                ]
                if len(sub_df) == 0: continue
                sub_df = bucketize_values(sub_df, "N")
                if len(sub_df) == 0: continue
                N_vals = sub_df["N Bucket Center"].astype(int).unique()
                s_vals = sub_df["s"].unique()
                k_vals = sub_df["k"].unique()

                N_to_peak_ttoks[modelname] = {}
                k_to_peak_ttoks[modelname] = {}
                s_to_peak_ttoks[modelname] = {}

                for s in s_vals:
                    for k in k_vals:
                        N_to_ptts = plot_correctness_by_ttoks_isolate_factor(sub_df, k, s, None, modelname, f"{task}_{encoder}")
                        if N_to_ptts:
                            N_to_peak_ttoks[modelname][(k, s, None)] = N_to_ptts
                    
                    plot_correctness_by_N_isolate_factor(sub_df, None, s, modelname, f"{task}_{encoder}")
                    plot_correctness_by_k_isolate_factor(sub_df, s, None, modelname, f"{task}_{encoder}")
                    
                    for N in N_vals:
                        k_to_ptts = plot_correctness_by_ttoks_isolate_factor(sub_df, None, s, N, modelname, f"{task}_{encoder}")
                        if k_to_ptts:
                            k_to_peak_ttoks[modelname][(None, s, N)] = k_to_ptts
                
                for k in k_vals:
                    plot_correctness_by_N_isolate_factor(sub_df, k, None, modelname, f"{task}_{encoder}")
                    plot_correctness_by_s_isolate_factor(sub_df, k, None, modelname, f"{task}_{encoder}")

                    for N in N_vals:
                        s_to_ptts = plot_correctness_by_ttoks_isolate_factor(sub_df, k, None, N, modelname, f"{task}_{encoder}")
                        if s_to_ptts:
                            s_to_peak_ttoks[modelname][(k, None, N)] = s_to_ptts

                for N in N_vals:
                    plot_correctness_by_k_isolate_factor(sub_df, None, N, modelname, f"{task}_{encoder}")
                    plot_correctness_by_s_isolate_factor(sub_df, None, N, modelname, f"{task}_{encoder}")

                if len(N_to_peak_ttoks[modelname]) == 0:
                    del N_to_peak_ttoks[modelname]
                if len(k_to_peak_ttoks[modelname]) == 0:
                    del k_to_peak_ttoks[modelname]
                if len(s_to_peak_ttoks[modelname]) == 0:
                    del s_to_peak_ttoks[modelname]

            plt.clf()
            plot_ptt_by_factor(N_to_peak_ttoks, "N Bucket Center", f"{task}_{encoder}")
            plot_ptt_by_factor(k_to_peak_ttoks, "k", f"{task}_{encoder}")
            plot_ptt_by_factor(s_to_peak_ttoks, "s", f"{task}_{encoder}")

            # for k in k_vals:
            #     plot_gen_by_factor(df, k, None, None, "No gen toks")

            # for s in s_vals:
            #     plot_gen_by_factor(df, None, s, None, "No gen toks")

            # for N in N_vals:
            #     plot_gen_by_factor(df, None, None, N, "No gen toks")
                # for s in s_vals:
                #     if "Model" in args.get_isolated:
                #         for k in k_vals:
                #             model_to_peak_ttoks = plot_correctness_by_ttoks_isolate_factor(df, k, s, N, None)
