import json
import re
import sys
import ipdb
import traceback
import os
import shutil
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import random
import glob
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from scipy.stats import sem
from scipy.optimize import curve_fit

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

def get_args():
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
    parser.add_argument("--k_vals", nargs='+', default=None)
    parser.add_argument("--N_vals", nargs='+', default=None)
    parser.add_argument("--models", nargs='+', default=[])
    parser.add_argument("--delete_old", action="store_true")
    args = parser.parse_args()
    foldername = os.path.join(f"{args.output_folder.rstrip('/')}_graphs_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}")
    temperature = args.temperature
    num_beams = args.num_beams
    num_gens = args.num_gens
    n_buckets = args.n_buckets

    return args


model_colors = {
    "Llama-3.1-8B-Instruct": "purple",
    "Qwen2.5-32B-Instruct": "blue",
    "Qwen2.5-14B-Instruct": "brown",
    "Qwen2.5-7B-Instruct": "yellow",
    "OLMo-2-1124-7B-Instruct": "black",
    "Ministral-8B-Instruct-2410": "orange",
    "gemma-2-9b-it": "pink",
}

model_nicknames = {
    "Llama-3.1-8B-Instruct": "Ll3.1-8B",
    "Qwen2.5-32B-Instruct": "Qw2.5-32B",
    "Qwen2.5-14B-Instruct": "Qw2.5-14B",
    "Qwen2.5-7B-Instruct": "Qw2.5-7B",
    "OLMo-2-1124-7B-Instruct": "OLMO-7B",
    "Ministral-8B-Instruct-2410": "Ministral-8B",
    "gemma-2-9b-it": "Ge2-9B",
}
colormap = get_cmap("tab10")  # Use a colormap with distinct colors

def load_data(data_folder, models_to_plot):
    ks = []
    Ns = []
    models = []
    gen_toks = []
    correct = []

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(data_folder, f"k*")):
        parsed_experimentname = re.search(r"k(\d+)_N(\d+)", subfolder)
        if parsed_experimentname is None:
            continue
        k = parsed_experimentname.group(1)
        N = parsed_experimentname.group(2)

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
            if not any(model_str in modelname for model_str in models_to_plot):
                continue
            
            results = json.load(open(experiment_file))

            ks.extend([k for _ in results])
            Ns.extend([N for _ in results])
            models.extend([modelname for _ in results])
            gen_toks.extend([ex["generated_tokens"] for ex in results])
            correct.extend([ex["correct"] for ex in results])

    data = {
        "k": ks,
        "N": Ns,
        "Model": models,
        "No gen toks": gen_toks,
        "Correct?": correct,
    }

    # Create a DataFrame for the new data
    df = pd.DataFrame(data)

    return df

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
    bucket_avg["Correct?"] = bucket_avg["Correct?"].astype(float)
    sub_df = sub_df.merge(bucket_avg, on=groupby_key, suffixes=('', '_mean'))

    return bucket_avg, sub_df

sns.set("talk")

def plot_requested_vs_generated(df):
    # Separate data by model size
    model_data = {
        model_name: df[df["Model"].str.contains(model_name)].sort_values(by="No gen toks")
        for model_name in model_colors
    }

    # Plot box plots for each model size and requested length category
    plt.figure(figsize=(12, 6))

    # Get sorted unique requested lengths for x-axis labels
    tick_positions = []
    plotted_something = False
    labels = []
    for i, (model, model_df) in enumerate(model_data.items(), start=1):
        # Extract data for boxplot
        data = model_df["No gen toks"].tolist()
        if not data:
            continue

        # Define position for the boxplot
        position = [i]
        tick_positions.append(i)
        labels.append(model)

        # Plot the boxplot
        plt.boxplot(
            [data],
            positions=position,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=model_colors[model], color=model_colors[model]),
            medianprops=dict(color="black"),
            showfliers=False,
        )
        plotted_something = True

    if not plotted_something:
        return

    # Set x-axis labels
    plt.xticks(ticks=tick_positions, labels=labels, rotation=45)
    plt.xlabel("Method")
    plt.ylabel("No. Generate Tokens")

    # Legend for the models
    legend_elements = [
        Line2D([0], [0], color=model_colors[model], lw=2, label=model)
        for model in model_colors
    ]
    plt.legend(handles=legend_elements, loc="best", fancybox=True, shadow=True)

    plt.savefig(
        os.path.join(foldername, "genlength_boxplot.png")
    )
    plt.clf()

def plot_correctness_by_ttoks(filtered_data, k, N, modelname, label, rgba_color, is_subplot=False):
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
        plt.legend(loc="best", fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save and clear the figure
        filename = f"k{k}_N{N}_{modelname}.png"
        os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
        plt.savefig(
            os.path.join(foldername, "isolate_factor", filename)
        )
        plt.clf()
    return (peak_ttoks.item(), best_performance.item(), (best_performance - ci.values[index_peak]).item() > 0.5)

def plot_correctness_by_ttoks_isolate_factor(df, k, N, modelname, factor_vals):
    assert sum((factor is None for factor in (k, N, modelname))) == 1, f"{(k, N, modelname)} one must be None"
    # Filter the data for the specific model, k, N, modelname
    filtered_data = df[
        (df["Model"].str.contains(modelname) if modelname is not None else True)
        & ((df["k"] == k) if k is not None else True)
        & ((df["N"] == N) if N is not None else True)
    ]

    if k is None:
        isolated_factor = "k"
    elif N is None:
        isolated_factor = "N"
    elif modelname is None: 
        isolated_factor = "Model"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: {(k, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))
    if isolated_factor == "Model":
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=str)
    else:
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=int)
        max_factor = int(factor_values[-1])
        base_color = model_colors.get(modelname, "blue")

    factor_val_to_peak_ttoks = []
    used_factor_values = []
    # Iterate over unique t values
    for factor_value in factor_values:
        if modelname is None:
            factor_filtered_data = filtered_data[filtered_data[isolated_factor].str.contains(factor_value)]
            if factor_filtered_data.empty: continue
            base_color = model_colors.get(factor_value, "blue")
            rgba_color = mcolors.to_rgba(base_color, alpha=0.8)
            plot_results = plot_correctness_by_ttoks(factor_filtered_data, k, N, factor_value, factor_value, rgba_color, is_subplot=True)
        else:
            factor_filtered_data = filtered_data[filtered_data[isolated_factor] == factor_value]
            if factor_filtered_data.empty: continue
            factor_value = int(factor_value)
            # Normalize the intensity of the color based on t
            color_intensity = factor_value / (max_factor + 1) 
            label = f"{isolated_factor}={factor_value}"
            rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
            if isolated_factor == "k":
                plot_results = plot_correctness_by_ttoks(factor_filtered_data, factor_value, N, modelname, label, rgba_color, is_subplot=True)
            elif isolated_factor == "N":
                plot_results = plot_correctness_by_ttoks(factor_filtered_data, k, factor_value, modelname, label, rgba_color, is_subplot=True)
        if plot_results:
            used_factor_values.append(str(factor_value))
            (peak_ttoks, _, task_doable) = plot_results
            if task_doable:
                factor_val_to_peak_ttoks.append((factor_value, peak_ttoks))
        
    if len(used_factor_values) == 0: return
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Average Correctness")
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"{isolated_factor}{''.join(used_factor_values)}_"
    if k is None:
        filename += f"N{N}_{modelname}.png"
    elif N is None:
        filename += f"k{k}_{modelname}.png"
    elif modelname is None:
        filename = f"byModel_k{k}_N{N}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()
    return factor_val_to_peak_ttoks

def plot_ptt_by_factor(factor_to_peak_ttoks, isolated_factor, plot_individual_lines, metric="pearsonr"):
    plt.figure(figsize=(8, 5))
    # To store normalized values per model for later linregress
    # Find the factor vals that all models have at least some successes in
    min_max_factor_val = None
    max_min_factor_val = None
    for factor_to_ptts in factor_to_peak_ttoks.values():
        model_max_factor_val = None
        model_min_factor_val = None
        for isolated_factor_val_to_peak_tts in factor_to_ptts.values():
            fvs = [fv for (fv, _) in isolated_factor_val_to_peak_tts]
            if (model_min_factor_val is None) or min(fvs) < model_min_factor_val: 
                model_min_factor_val = min(fvs)
            if (model_max_factor_val is None) or max(fvs) > model_max_factor_val: 
                model_max_factor_val = max(fvs)
        if (min_max_factor_val is None) or model_max_factor_val < min_max_factor_val: 
            min_max_factor_val = model_max_factor_val
        if (max_min_factor_val is None) or model_min_factor_val > max_min_factor_val: 
            max_min_factor_val = model_min_factor_val
        
    all_factor_vals = []
    all_normalized_peak_tts =  []
    all_normalized_avg_peak_tts = []

    for modelname, factor_to_ptts in factor_to_peak_ttoks.items():
        base_color = model_colors.get(modelname, "blue")
        rgba_color = mcolors.to_rgba(base_color, alpha=0.9)

        fv_to_ptts_avged = {}
        for (_, isolated_factor_val_to_peak_tts) in factor_to_ptts.items():
            for (fv, ptt) in isolated_factor_val_to_peak_tts:
                if fv < max_min_factor_val: continue
                if fv > min_max_factor_val: continue
                if fv not in fv_to_ptts_avged:
                    fv_to_ptts_avged[fv] = []
                fv_to_ptts_avged[fv].append(ptt)
        if len(fv_to_ptts_avged) == 0:
            continue

        factor_vals = []
        avg_peak_tts = []
        all_peak_tts = []
        ci_lower_bounds = []
        ci_upper_bounds = []

        # Calculate averages and confidence intervals
        for fv, ptts in fv_to_ptts_avged.items():
            avg_ptt = np.mean(ptts)
            factor_vals.append(fv)
            all_factor_vals.extend(fv for _ in ptts)
            all_peak_tts.extend(ptts)
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
        if len(all_peak_tts) == 1: 
            continue

        # Normalize avg_peak_tts for the current model
        min_val = min(avg_peak_tts)
        max_val = max(avg_peak_tts)
        # Store the normalized values for MSE
        normalized_peak_tts = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in all_peak_tts]
        all_normalized_peak_tts.extend(normalized_peak_tts)
        legend_label = model_nicknames[modelname]

        # plot the normalized averageds
        normalized_avg_peak_tts = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in avg_peak_tts]
        all_normalized_avg_peak_tts.extend([(fv, napt) for fv, napt in zip(factor_vals, normalized_avg_peak_tts)])
        if plot_individual_lines:
            slope, intercept, _, _, _ = stats.linregress(factor_vals, normalized_avg_peak_tts)
            x_vals = np.linspace(min(factor_vals), max(factor_vals), 100)
            y_vals = slope * x_vals + intercept
            plt.plot(x_vals, y_vals, color=rgba_color, linestyle='--')
            
            if metric == 'mse':
                # Calculate Mean Squared Error
                predicted_vals = slope * np.array(all_factor_vals[-len(normalized_peak_tts):]) + intercept
                mse = np.mean((np.array(normalized_peak_tts) - predicted_vals) ** 2)
                legend_label = f"{model_nicknames[modelname]} (MSE: {mse_annotation:.2f})"
            elif metric == 'pearsonr':
                # Calculate pearson corr
                correlation, _ = stats.pearsonr(all_factor_vals[-len(normalized_peak_tts):], normalized_peak_tts)
                legend_label = f"{model_nicknames[modelname]} (Corr: {correlation:.2f})"

        sns.scatterplot(x=factor_vals, y=normalized_avg_peak_tts, 
                    marker='o', color=rgba_color, label=legend_label)

    if len(all_factor_vals) == 0: return

    if not plot_individual_lines:
        # See how well all collected points fit to the common linear regression line
        slope, intercept, _, _, _ = stats.linregress([fv for (fv, _) in all_normalized_avg_peak_tts], [napt for (_, napt) in all_normalized_avg_peak_tts])

        # Generate x values for the target regression line
        x_vals = np.linspace(min(all_factor_vals), max(all_factor_vals), 100)
        y_vals = slope * x_vals + intercept
        # Plot the target linear line
        plt.plot(x_vals, y_vals, color='black', linestyle='--')
        
        if metric == 'mse':
            # Calculate Mean Squared Error
            predicted_vals = slope * np.array(all_factor_vals) + intercept
            mse = np.mean((np.array(all_normalized_peak_tts) - predicted_vals) ** 2)
            # Annotate the MSE on the plot
            mse_annotation = f"MSE: {mse:.4f}"
            plt.text(0.05, 0.95, mse_annotation, transform=plt.gca().transAxes,
                    fontsize=42, color='red', verticalalignment='top')
        elif metric == 'pearsonr':
            # Calculate pearson corr
            correlation, _ = stats.pearsonr(all_factor_vals, all_normalized_peak_tts)
            corr_annotation = f"Correlation: {correlation:.4f}"
            plt.text(0.05, 0.95, corr_annotation, transform=plt.gca().transAxes,
                    fontsize=14, color='red', verticalalignment='top')

    # Finalize and save the plot
    plt.ylim(0, 1)
    plt.gca().set_aspect(max(all_factor_vals) - min(all_factor_vals))
    plt.xlabel(isolated_factor)
    plt.ylabel("Normalized Avg. Peak Tokens")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    os.makedirs(os.path.join(foldername, "meta_plots"), exist_ok=True)
    plt.savefig(os.path.join(foldername, "meta_plots", f"diminish_{isolated_factor}{'_ind' if plot_individual_lines else ''}_{metric}.png"))
    plt.clf()

def plot_correctness_by_isolate_factor(df, isolated_factor, plot_against_factor, modelname):
    # Filter the data for the specific model, isolated_factor, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
    ]
    base_color = model_colors.get(modelname, "blue")

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for {modelname}.")
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
            filtered_data_factor.groupby(plot_against_factor)
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
        for plot_against_factor_val in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor[plot_against_factor] == str(plot_against_factor_val)]["Correct?"]
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
        if plot_against_factor == "N" and len(performance) > 1:
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
        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel(plot_against_factor)
    plt.title(
        f"Correctness vs. {plot_against_factor} ({isolated_factor}={factor_value}, {modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_{plot_against_factor}_{modelname}.png"
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
    df = load_data(args.output_folder, args.models)
    df_nocot = load_data(args.output_folder+"_nocot", args.models)
    df = pd.concat([df, df_nocot])

    plot_requested_vs_generated(df)

    k_vals = df["k"].unique()
    N_vals = df["N"].unique()
    models = df["Model"].unique()
    all_var_vals = [k_vals, N_vals]

    N_to_peak_ttoks = {}
    k_to_peak_ttoks = {}

    for modelname in models:
        N_to_peak_ttoks[modelname] = {}
        k_to_peak_ttoks[modelname] = {}
        for k in k_vals:
            N_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, k, None, modelname, N_vals)
            if N_to_ptts:
                N_to_peak_ttoks[modelname][(k, None)] = N_to_ptts
        for N in N_vals:
            k_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, None, N, modelname, k_vals)
            if k_to_ptts:
                k_to_peak_ttoks[modelname][(None, N)] = k_to_ptts

        plot_correctness_by_isolate_factor(df, "k", "N", modelname)
        plot_correctness_by_isolate_factor(df, "N", "k", modelname)
            
        plt.clf()
        plot_ptt_by_factor(N_to_peak_ttoks, "N", False)
        plot_ptt_by_factor(k_to_peak_ttoks, "k", False)
        plot_ptt_by_factor(N_to_peak_ttoks, "N", True)
        plot_ptt_by_factor(k_to_peak_ttoks, "k", True)

    for k in k_vals:
        for N in N_vals:
            plot_correctness_by_ttoks_isolate_factor(df, k, N, None, models)
