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
    # "Mistral-7B": "red",
    "OLMo-2-1124-13B": "green",
    "OLMo-2-1124-7B": "black",
    "Ministral-8B": "orange",
    "gemma-2-9b": "pink",
}

colormap = get_cmap("tab10")  # Use a colormap with distinct colors

def load_data(data_folder, k_vals, m_vals, N_vals, modelnames, skip_nulls=True):
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
            if not any(model_str in modelname for model_str in modelnames):
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
    ).astype(float)

    bucket_avg["Correct?"] =  bucket_avg["Correct?"].astype(float)
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

def plot_correctness_by_ttoks(filtered_data, k, m, N, modelname, label, rgba_color, is_subplot=False):
    # plt.figure(figsize=(12, 6))

    bucket_avg, filtered_data = calculate_buckets_samesize(filtered_data)
    if filtered_data is None: return False
    if len(bucket_avg) == 0: return False
        
    # Find the index of the maximum value
    index_peak = np.argmax(bucket_avg["Correct?"])
    peak_ttoks = bucket_avg["Bucket Center"][index_peak]
    best_performance = bucket_avg["Correct?"][index_peak]
    
    # Plot the average correctness for each model size and method
    plt.plot(
        bucket_avg["Bucket Center"],
        bucket_avg["Correct?"],
        color=rgba_color,
        label=label,
    )

    sem_values = filtered_data.groupby("Bucket Center", observed=True)["Correct?"].apply(stats.sem)
    # Calculate confidence intervals
    ci = sem_values * 1.96  # For 95% confidence
    ci = sem_values.reindex(bucket_avg["Bucket Center"]).fillna(0)

    plt.fill_between(
        bucket_avg["Bucket Center"],
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
            f"Average Correctness vs. No. of Generated Tokens ({k, m, N, modelname})"
        )
        plt.legend(loc="best", fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save and clear the figure
        filename = f"k{k}_m{m}_N{N}_{modelname}.png"
        os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
        plt.savefig(
            os.path.join(foldername, "isolate_factor", filename)
        )
        plt.clf()
    return (peak_ttoks.item(), best_performance.item(), (best_performance - ci.values[index_peak]).item() > (1. / int(k)))

def plot_correctness_by_ttoks_isolate_factor(df, k, m, N, modelname):
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

    if filtered_data.empty:
        print(f"No examples found for: {(k, m, N, modelname)}.")
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
            plot_results = plot_correctness_by_ttoks(factor_filtered_data, k, m, N, factor_value, rgba_color, is_subplot=True)
        else:
            factor_filtered_data = filtered_data[filtered_data[isolated_factor]==factor_value]
            if factor_filtered_data.empty: continue
            factor_value = int(factor_value)
            # Normalize the intensity of the color based on t
            color_intensity = factor_value / (max_factor + 1) 
            label = f"{isolated_factor}={factor_value}"
            rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
            if isolated_factor == "k":
                plot_results = plot_correctness_by_ttoks(factor_filtered_data, factor_value, m, N, modelname, label, rgba_color, is_subplot=True)
            elif isolated_factor == "m":
                plot_results = plot_correctness_by_ttoks(factor_filtered_data, k, factor_value, N, modelname, label, rgba_color, is_subplot=True)
            elif isolated_factor == "N":
                plot_results = plot_correctness_by_ttoks(factor_filtered_data, k, m, factor_value, modelname, label, rgba_color, is_subplot=True)
        if plot_results:
            used_vals.append(factor_value)
            (peak_ttoks, _, task_doable) = plot_results
            if task_doable:
                factor_val_to_peak_ttoks.append((factor_value, peak_ttoks))
    #         else: breakpoint()
    #     else:
    #         breakpoint()
    
    # breakpoint()
    if len(factor_val_to_peak_ttoks) == 0: return
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
    filename = f"{isolated_factor}{''.join(str(uv) for uv in used_vals)}_"
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

    return factor_val_to_peak_ttoks

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

def plot_correctness_by_N_isolate_factor(df, k, m, modelname):
    assert sum((factor is None for factor in (k, m))) == 1, f"{(k, m)} one must be None"
    # Filter the data for the specific model, k, m, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["k"] == k) if k is not None else True)
        & ((df["m"] == m) if m is not None else True)
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

def plot_correctness_by_k_isolate_factor(df, m, N, modelname):
    assert sum((factor is None for factor in (m, N))) == 1, f"{(m, N)} one must be None"
    # Filter the data for the specific model, k, m, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N"] == N) if N is not None else True)
        & ((df["m"] == m) if m is not None else True)
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

def plot_correctness_by_m_isolate_factor(df, k, N, modelname):
    assert sum((factor is None for factor in (k, N))) == 1, f"{(k, N)} one must be None"
    # Filter the data for the specific model, k, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N"] == N) if N is not None else True)
        & ((df["k"] == k) if k is not None else True)
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

def plot_ptt_by_factor(factor_to_peak_ttoks, isolated_factor):
    isolated_factor_idx = 0 if isolated_factor == "k" else (1 if isolated_factor == "m" else 2)

    # normalized_values_per_model = {}
    
    # To store normalized values per model for later linregress
    all_factor_vals = []
    all_normalized_avg_peak_tts = []

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
        all_normalized_avg_peak_tts.extend(normalized_avg_peak_tts)

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
    if len(all_factor_vals) == 0: return

    # See how well all collected points fit to the [0,1] line (it's normalized, remember)
    # slope, intercept, _, _, _ = stats.linregress(all_factor_vals, all_normalized_avg_peak_tts)
    slope = 1. / (max(all_factor_vals) - min(all_factor_vals))
    intercept = 1 - slope * max(all_factor_vals)

    # Generate x values for the target regression line
    x_vals = np.linspace(min(all_factor_vals), max(all_factor_vals), 100)
    y_vals = slope * x_vals + intercept

    # Calculate Mean Squared Error
    predicted_vals = slope * np.array(all_factor_vals) + intercept
    mse = np.mean((np.array(all_normalized_avg_peak_tts) - predicted_vals) ** 2)

    # Plot the target regression line
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Linear Regression')

    # Annotate the MSE on the plot
    mse_annotation = f"MSE: {mse:.4f}"
    plt.text(0.05, 0.95, mse_annotation, transform=plt.gca().transAxes,
             fontsize=12, color='red', verticalalignment='top')

    # Finalize and save the plot
    plt.ylim(-1, 2)
    plt.xlabel(isolated_factor)
    plt.ylabel("Normalized Tokens Where Returns Diminish (Peak Tokens)")
    plt.title(f"Diminishing Returns vs. {isolated_factor}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    os.makedirs(os.path.join(foldername, "meta_plots"), exist_ok=True)
    plt.savefig(os.path.join(foldername, "meta_plots", f"diminish_{isolated_factor}.png"))

    for factor_vals, normalized_avg_peak_tts, yerr_lower, yerr_upper, rgba_color, base_color, modelname in errbars:
        # Plotting with confidence intervals (using normalized values)
        plt.errorbar(factor_vals, normalized_avg_peak_tts, 
                     yerr=[yerr_lower, yerr_upper],
                     fmt='o', color=rgba_color, ecolor=base_color, capsize=5)

    plt.savefig(os.path.join(foldername, "meta_plots", f"diminish_{isolated_factor}_errbars.png"))
    plt.clf()

if __name__ == "__main__":
    args = get_args()
    if args.delete_old and os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername, exist_ok=True)
    df, all_var_vals = load_data(args.output_folder, args.k_vals, args.m_vals, args.N_vals, args.models)
    df_nocot, _ = load_data(args.output_folder+"_nocot", args.k_vals, args.m_vals, args.N_vals, args.models)
    df = pd.concat([df, df_nocot])

    k_vals = args.k_vals if args.k_vals is not None else all_var_vals[0]
    m_vals = args.m_vals if args.m_vals is not None else all_var_vals[1]
    N_vals = args.N_vals if args.N_vals is not None else all_var_vals[2]

    N_to_peak_ttoks = {}
    k_to_peak_ttoks = {}
    m_to_peak_ttoks = {}

    for modelname in args.models:
        N_to_peak_ttoks[modelname] = {}
        k_to_peak_ttoks[modelname] = {}
        m_to_peak_ttoks[modelname] = {}
        for m in m_vals:
            # plot_correctness_by_wraps(df, m, modelname)
            for k in k_vals:
                if "N" in args.get_isolated:
                    N_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, k, m, None, modelname)
                    if N_to_ptts:
                        N_to_peak_ttoks[modelname][(k, m, None)] = N_to_ptts
                
            plot_correctness_by_N_isolate_factor(df, None, m, modelname)
            plot_correctness_by_k_isolate_factor(df, m, None, modelname)
            
            for N in N_vals:
                if "k" in args.get_isolated:
                    k_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, None, m, N, modelname)
                    if k_to_ptts:
                        k_to_peak_ttoks[modelname][(None, m, N)] = k_to_ptts
            
        for k in k_vals:
            plot_correctness_by_N_isolate_factor(df, k, None, modelname)
            plot_correctness_by_m_isolate_factor(df, k, None, modelname)

            for N in N_vals:
                if "m" in args.get_isolated:
                    m_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, k, None, N, modelname)
                    if m_to_ptts:
                        m_to_peak_ttoks[modelname][(k, None, N)] = m_to_ptts

        for N in N_vals:
            plot_correctness_by_k_isolate_factor(df, None, N, modelname)
            plot_correctness_by_m_isolate_factor(df, None, N, modelname)

        if len(N_to_peak_ttoks[modelname]) == 0:
            del N_to_peak_ttoks[modelname]
        if len(k_to_peak_ttoks[modelname]) == 0:
            del k_to_peak_ttoks[modelname]
        if len(m_to_peak_ttoks[modelname]) == 0:
            del m_to_peak_ttoks[modelname]

    plt.clf()
    plot_ptt_by_factor(N_to_peak_ttoks, "N")
    plot_ptt_by_factor(k_to_peak_ttoks, "k")
    plot_ptt_by_factor(m_to_peak_ttoks, "m")

    # for k in k_vals:
    #     plot_gen_by_factor(df, k, None, None, "No gen toks")
    #     plot_gen_by_factor(df, k, None, None, "Len gen no digits")

    # for m in m_vals:
    #     plot_gen_by_factor(df, None, m, None, "No gen toks")
    #     plot_gen_by_factor(df, None, m, None, "Len gen no digits")

    # for N in N_vals:
    #     plot_gen_by_factor(df, None, None, N, "No gen toks")
    #     plot_gen_by_factor(df, None, None, N, "Len gen no digits")
        # for m in m_vals:
        #     if "Model" in args.get_isolated:
        #         for k in k_vals:
        #             model_to_peak_ttoks = plot_correctness_by_ttoks_isolate_factor(df, k, m, N, None)
