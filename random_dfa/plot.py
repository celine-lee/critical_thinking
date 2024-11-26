import torch
import math
import json
import re
import sys
import ipdb
import traceback
import os
import ast
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import random
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from collections import defaultdict
from scipy.stats import sem

foldername = sys.argv[1]
temperature = sys.argv[2]

model_colors = {
    "3.1-8B": "purple",
}
method_markers = {
    "brief": "d",
    "none": "*",
    "detail": "^",
    "states_long": "o",
    "states_short": ".",
}
method_linestyle = {
    "brief": ":",
    "none": "--", 
    "detail": "-.",
    "states_long": "-",
    "states_short": "--",
}

ks = []
Ns = []
ts = []
models = []
methods = []
gen_toks = []
correct = []


# Load data from experiment files
for subfolder in glob.glob(os.path.join(foldername, f"k*")):
    parsed_experimentname = re.search(rf"k(\d+)_N(\d+)_t(\d+)", subfolder)
    if parsed_experimentname is None:
        continue
    k = parsed_experimentname.group(1)
    N = parsed_experimentname.group(2)
    t = parsed_experimentname.group(3)
    for experiment_file in glob.glob(os.path.join(subfolder, "*")):
        if f"T{temperature}" not in experiment_file:
            continue
        modelname = re.search(r"(Llama-3.+)_T", experiment_file).group(1)
        if not any(model_str in modelname for model_str in model_colors):
            continue
        
        results = json.load(open(experiment_file))
        for methodname in method_markers:
            if methodname in experiment_file:
                break
        if methodname not in experiment_file:
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
# Create a DataFrame for the new data
df = pd.DataFrame(data)
# Create a column for grouping
df["Group"] = list(zip(df["Method"], df["Model"], df["k"], df["N"], df["t"]))


# Find the minimum size across all groups
group_sizes = df["Group"].value_counts()
min_size = group_sizes.min()

# Subsample each group explicitly
balanced_data = []
for group, size in group_sizes.items():
    group_df = df[df["Group"] == group]
    balanced_data.append(group_df.sample(n=min_size, random_state=42))

# Combine all subsampled groups
balanced_df = pd.concat(balanced_data).reset_index(drop=True)

# Separate data by model size
model_data = {
    model_name: df[df["Model"].str.contains(model_name)].sort_values(by="No gen toks")
    for model_name in model_colors
}
balanced_model_data = {
    model_name: balanced_df[balanced_df["Model"].str.contains(model_name)].sort_values(
        by="No gen toks"
    )
    for model_name in model_colors
}


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

# to check that different prompting methods to produce differnt-lenght CoTs
def plot_requested_vs_generated():
    # Plot box plots for each model size and requested length category
    plt.figure(figsize=(12, 6))

    # Get sorted unique requested lengths for x-axis labels
    unique_lengths = sorted(df["Method"].unique())
    tick_positions = []

    plotted_something = False
    # Plot box plots for each requested length category for each model
    for i, (model, model_df) in enumerate(model_data.items(), start=1):
        grouped = model_df.groupby("Method")["No gen toks"].apply(list)
        if len(grouped) == 0:
            continue
        positions = [
            unique_lengths.index(req_len) + (i - 2) * 0.2 for req_len in grouped.index
        ]
        tick_positions = positions if i == 1 else tick_positions
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
    plt.legend(handles=legend_elements, loc="upper left", fancybox=True, shadow=True)

    plt.savefig(
        os.path.join(foldername, f"lengthvsrequested_boxplot_T{temperature}.png")
    )
    plt.clf()


def calculate_buckets_samerange(sub_df, num_buckets=5, bins=None, groupby_key="Model"):
    if len(sub_df) == 0:
        return None, None
    if bins is None:
        min_len = df["No gen toks"].min()
        max_len = df["No gen toks"].max()
        bins = np.linspace(min_len, max_len, num_buckets + 1)

    sub_df["Length Bucket"] = pd.cut(
        sub_df["No gen toks"], bins, include_lowest=True
    )

    bucket_avg = (
        sub_df.groupby([groupby_key, "Length Bucket"])["Correct?"].mean().reset_index()
    )
    sub_df = sub_df.merge(bucket_avg, on=groupby_key, suffixes=('', '_mean'))
    bucket_avg["Bucket Center"] = bucket_avg["Length Bucket"].apply(lambda x: x.mid)
    return bucket_avg, sub_df

def calculate_buckets_samesize(sub_df, num_buckets=5, groupby_key="Model"):
    if len(sub_df) == 0: return None, None
    # Use qcut to create equal-sized buckets by count
    sub_df["Length Bucket"] = pd.qcut(
        sub_df["No gen toks"], q=num_buckets, duplicates="drop"
    )

    bucket_avg = (
        sub_df.groupby([groupby_key, "Length Bucket"])["Correct?"].mean().reset_index()
    )

    # Calculate the bucket center by averaging bin edges
    bucket_avg["Bucket Center"] = bucket_avg["Length Bucket"].apply(
        lambda x: (x.left + x.right) / 2
    )
    sub_df = sub_df.merge(bucket_avg, on=groupby_key, suffixes=('', '_mean'))

    return bucket_avg, sub_df

def plot_correctness_by_ttoks_per_t_N_model(t, N, modelname, num_buckets=5):
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
    k_values = sorted(filtered_data['k'].unique(), key=int)
    max_k = int(k_values[-1])

    # Iterate over unique k values
    for k in k_values:
        filtered_data_k = filtered_data[filtered_data['k'] == k]
        bucket_avg, filtered_data_k = calculate_buckets_samesize(filtered_data_k, num_buckets=num_buckets)
        if bucket_avg is None:
            return
        # Normalize the intensity of the color based on k
        color_intensity = int(k) / max_k 
        base_color = model_colors.get(modelname, "blue")
        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Plot the average correctness for each model size and method
        plt.plot(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"],
            color=rgba_color,
            label=f"k={k}",
        )
        sem_values = filtered_data_k.groupby("Bucket Center")["Correct?"].apply(sem)
        # Calculate confidence intervals
        ci = sem_values * 1.96  # For 95% confidence
        plt.fill_between(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"] - ci.values,
            bucket_avg["Correct?"] + ci.values,
            alpha=color_intensity,
            color=rgba_color,
        )

        plt.axhline(
            y=1.0 / int(k), 
            color=rgba_color,
            linestyle=":",
        )

    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Average Correctness")
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.title(
        f"Average Correctness vs. No. of Generated Tokens (N={N}, t={t}, {modelname} Buckets={num_buckets})"
    )
    plt.legend(loc="upper left", fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(
        os.path.join(
            foldername,
            f"N{N}_t{t}_{modelname}_{num_buckets}buckets_T{temperature}.png",
        )
    )
    plt.clf()

def plot_correctness_by_ttoks_per_k_t_model(k, t, modelname, num_buckets=5):
    # Filter the data for the specific model, k, and t
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
    N_values = sorted(filtered_data['N'].unique(), key=int)
    max_N = int(N_values[-1])

    # Iterate over unique N values
    for N in N_values:
        filtered_data_N = filtered_data[filtered_data['N'] == N]
        bucket_avg, filtered_data_N = calculate_buckets_samesize(filtered_data_N, num_buckets=num_buckets)
        if filtered_data_N is None:
            return
        # Normalize the intensity of the color based on N
        color_intensity = int(N) / max_N 
        base_color = model_colors.get(modelname, "blue")
        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Plot the average correctness for each model size and method
        plt.plot(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"],
            color=rgba_color,
            label=f"N={N}",
        )
        sem_values = filtered_data_N.groupby("Bucket Center")["Correct?"].apply(sem)
        # Calculate confidence intervals
        ci = sem_values * 1.96  # For 95% confidence
        plt.fill_between(
            bucket_avg["Bucket Center"],
            bucket_avg["Correct?"] - ci.values,
            bucket_avg["Correct?"] + ci.values,
            alpha=color_intensity,
            color=rgba_color,
        )

    plt.axhline(y=1.0 / int(k), linestyle=":")
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Average Correctness")
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.title(
        f"Average Correctness vs. No. of Generated Tokens (k={k}, t={t}, {modelname} Buckets={num_buckets})"
    )
    plt.legend(loc="upper left", fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(
        os.path.join(
            foldername,
            f"k{k}_t{t}_{modelname}_{num_buckets}buckets_T{temperature}.png",
        )
    )
    plt.clf()

def plot_correctness_by_N_per_compute_budget(modelname, k, t, num_buckets=5):
    
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
        filtered_data, num_buckets=num_buckets, groupby_key="N"
    )
    
    # Ensure bucket calculation succeeded
    if filtered_data is None:
        print("Bucket calculation failed. Ensure there is sufficient data for bucketing.")
        return

    # Cast N to integers for proper sorting
    filtered_data["N"] = filtered_data["N"].astype(int)
    # Group by bucket center and N, and calculate average correctness
    grouped_data = (
        filtered_data.groupby(["Bucket Center", "N"])["Correct?"]
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
    plt.title(f"Average Correctness vs. N (Model: {modelname}, k={k} t={t}, Buckets: {num_buckets})")
    plt.legend(title="Compute Budget Buckets", loc="upper left", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save and clear the figure
    plt.savefig(
        os.path.join(foldername, f"{modelname}_k{k}_t{t}_correctness_by_N_buckets{num_buckets}.png")
    )
    plt.clf()

def scatter_ttoks_by_complexity(modelname, N, t=2, num_buckets=20):
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
    bucket_size = math.ceil(filtered_data["No gen toks"].max() / num_buckets)

    # Plot the average correctness for each model size and method
    data = np.zeros((num_buckets * bucket_size, filtered_data["k"].max()+1)) # 
    totals = np.zeros((num_buckets * bucket_size, filtered_data["k"].max()+1))
    for k_val, no_gen_val, correct in zip(filtered_data["k"], filtered_data["No gen toks"], filtered_data["Correct?"]):
        data[(no_gen_val // bucket_size) * bucket_size:((no_gen_val // bucket_size) + 1) * bucket_size, k_val] += int(correct)
        totals[(no_gen_val // bucket_size) * bucket_size:((no_gen_val // bucket_size) + 1) * bucket_size, k_val] += 1
    data = np.where(totals == 0, np.nan, data / totals)
    # for i in range(data.shape[1]):
    #     col = data[:, i]
    #     mask = np.isnan(col)
    #     col[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), col[~mask])

    plt.imshow(data, interpolation="nearest", aspect="auto", origin="lower", cmap="viridis")
    # plt.scatter(
    #     filtered_data["k"],
    #     filtered_data["No gen toks"],
    #     c=filtered_data["Correct?"],
    #     cmap="viridis"
    # )
    plt.colorbar()

    # Customize plot labels and legend
    plt.ylabel("No gen toks")
    plt.xlabel("k")
    plt.xlim(xmin=filtered_data["k"].min())

    plt.title(
        f"No. of Generated Tokens vs. DFA complexity (N={N}, t={t}, {modelname})"
    )
    plt.legend(loc="upper left", fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(
        os.path.join(
            foldername,
            f"N{N}_t{t}_{modelname}_T{temperature}_{num_buckets}buckets_ttok_by_k.png",
        )
    )
    plt.clf()

def scatter_ttoks_by_inst_complexity(modelname, k, t=2, num_buckets=20):
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
    bucket_size = math.ceil(filtered_data["No gen toks"].max() / num_buckets)

    # Plot the average correctness for each model size and method
    data = np.zeros((num_buckets * bucket_size, filtered_data["N"].max()+1)) # 
    totals = np.zeros((num_buckets * bucket_size, filtered_data["N"].max()+1))
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
    plt.legend(loc="upper left", fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(
        os.path.join(
            foldername,
            f"k{k}_t{t}_{modelname}_T{temperature}_{num_buckets}buckets_ttok_by_N.png",
        )
    )
    plt.clf()


if __name__ == "__main__":
    plot_requested_vs_generated()
    all_ks = set(ks)
    all_Ns = set(Ns)
    all_ts = set(ts)
    for modelname in model_colors:
        for t in all_ts:
            for k in all_ks:
                if int(t) > int(k):
                    continue
                scatter_ttoks_by_inst_complexity(modelname, k, t)
                plot_correctness_by_ttoks_per_k_t_model(k, t, modelname)
            for N in all_Ns:
                plot_correctness_by_ttoks_per_t_N_model(t, N, modelname)
                scatter_ttoks_by_complexity(modelname, N, t)