
import os
import sys
import ipdb
import traceback
import glob
import re
import json
import argparse

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D    
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from scipy.optimize import curve_fit
from scipy.stats import sem
import seaborn as sns
sns.set("talk")

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger

sys.excepthook = debughook


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


def load_data(data_folder, varnames_and_wanted_vals, experiment_details_parser, kwargs):
    loaded_data = {
        "No gen toks": [],
        "Correct?": []
    }
    for varname in varnames_and_wanted_vals:
        loaded_data[varname] = []

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(data_folder, f"k*")):
        for experiment_file in glob.glob(os.path.join(subfolder, "*")):
            if f"T{kwargs['temperature']}" not in experiment_file:
                continue
            if re.search(r'_B\d+_S\d+', experiment_file):
                if f"_B{kwargs['num_beams']}_S{kwargs['num_gens']}.json" not in experiment_file: 
                    continue
            elif kwargs['temperature'] == 0.0: 
                assert kwargs['num_beams'] == 1 and kwargs['num_gens'] == 1
            
            experiment_details = experiment_details_parser(experiment_file)
            if experiment_details is None: continue
            for detail_name, detail_value in experiment_details.items():
                if (varnames_and_wanted_vals[detail_name] is not None) and detail_value not in varnames_and_wanted_vals[detail_name]:
                    continue
            
            results = json.load(open(experiment_file))
            results = [res for res in results if res["pred_answer"]]

            for varname in varnames_and_wanted_vals:
                loaded_data[varname].extend([experiment_details[varname] for _ in results])

            loaded_data["No gen toks"].extend([ex["generated_tokens"] for ex in results])
            loaded_data["Correct?"].extend([ex["correct"] for ex in results])

    # Create a DataFrame for the new data
    df = pd.DataFrame(loaded_data)

    return df

def calculate_buckets_samesize(sub_df, n_buckets, groupby_key="Model"):
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


def global_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_buckets", type=int, default=5)
    parser.add_argument("--num_gens", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--get_isolated", nargs='+')
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--delete_old", action="store_true")
    parser.add_argument("--only_meta", action="store_true")
    return parser

def plot_length_generated(df, kwargs, by_factor=None):
    # TODO by_factor
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
        os.path.join(kwargs['foldername'], "genlength_boxplot.png")
    )
    plt.clf()

def plot_correctness_by_ttoks(filtered_data, set_factor_values, label, rgba_color, is_subplot, kwargs):
    bucket_avg, filtered_data = calculate_buckets_samesize(filtered_data, kwargs['n_buckets'])
    if filtered_data is None: return False
    if len(bucket_avg) == 0: return False
        
    # Find the index of the maximum value
    index_peak = np.argmax(bucket_avg["Correct?"])
    peak_ttoks = bucket_avg["Bucket Center"][index_peak]
    best_performance = bucket_avg["Correct?"][index_peak]
    
    sem_values = filtered_data.groupby("Bucket Center", observed=True)["Correct?"].apply(stats.sem)
    # Calculate confidence intervals
    ci = sem_values * 1.96  # For 95% confidence
    ci = sem_values.reindex(bucket_avg["Bucket Center"]).fillna(0)

    if kwargs['only_meta']: 
        return (peak_ttoks.item(), best_performance.item(), (best_performance - ci.values[index_peak]).item() > kwargs['compute_random'](set_factor_values))
    # Plot the average correctness for each model size and method
    plt.plot(
        bucket_avg["Bucket Center"],
        bucket_avg["Correct?"],
        color=rgba_color,
        label=label,
    )

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
            f"Average Correctness vs. No. of Generated Tokens {set_factor_values}"
        )
        plt.legend(loc="best", fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save and clear the figure
        filename = "_".join(f"{factor_name}{factor_value}" for factor_name, factor_value in set_factor_values.items()) + ".png"
        os.makedirs(os.path.join(kwargs['foldername'], "isolate_factor"), exist_ok=True)
        plt.savefig(
            os.path.join(kwargs['foldername'], "isolate_factor", filename)
        )
        plt.clf()
    return (peak_ttoks.item(), best_performance.item(), (best_performance - ci.values[index_peak]).item() > kwargs['compute_random'](set_factor_values))

def plot_correctness_by_ttoks_isolate_factor(df, factor_set_values, isolated_factor, kwargs):
    # Filter the data for the specified factor_set_values
    bool_filter = True
    for factor_name, factor_val in factor_set_values.items():
        bool_filter = bool_filter & (df[factor_name] == factor_val)
    filtered_data = df[bool_filter]

    if filtered_data.empty:
        print(f"No examples found for: {factor_set_values}.")
        return

    if isolated_factor == "Model":
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=str)
    else:
        factor_values = sorted(filtered_data[isolated_factor].unique(), key=int)
        base_color = model_colors.get(factor_set_values["Model"], "blue")
        max_factor = int(factor_values[-1])

    factor_val_to_peak_ttoks = []
    used_vals = []
    plt.figure(figsize=(12, 6))
    # Iterate over unique t values
    for factor_value in factor_values:
        factor_filtered_data = filtered_data[filtered_data[isolated_factor]==factor_value]
        if factor_filtered_data.empty: continue
        
        if isolated_factor == "Model":
            base_color = model_colors.get(factor_value, "blue")
            rgba_color = mcolors.to_rgba(base_color, alpha=0.8)
            label = factor_value
        else:
            factor_value = int(factor_value)
            # Normalize the intensity of the color based on t
            color_intensity = factor_value / (max_factor + 1) 
            label = f"{isolated_factor}={factor_value}"
            rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)

        plot_results = plot_correctness_by_ttoks(factor_filtered_data, factor_set_values | {isolated_factor: factor_value}, label,rgba_color, True, kwargs)
        if plot_results:
            used_vals.append(factor_value)
            (peak_ttoks, _, task_doable) = plot_results
            if task_doable:
                factor_val_to_peak_ttoks.append((factor_value, peak_ttoks))
    if len(factor_val_to_peak_ttoks) == 0: return

    if kwargs['only_meta']: 
        return factor_val_to_peak_ttoks

    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Average Correctness")
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"{isolated_factor}{''.join(str(uv) for uv in used_vals)}_"
    filename += "_".join(f"{factor_name}{factor_value}" for factor_name, factor_value in factor_set_values.items()) + ".png"
    os.makedirs(os.path.join(kwargs['foldername'], "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(kwargs['foldername'], "isolate_factor", filename)
    )
    plt.clf()

    return factor_val_to_peak_ttoks

def plot_ptt_by_factor(factor_to_peak_ttoks, isolated_factor, plot_individual_lines, kwargs, metric="pearsonr"):
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
            # slope, intercept, _, _, _ = stats.linregress(factor_vals, normalized_avg_peak_tts)
            # x_vals = np.linspace(min(factor_vals), max(factor_vals), 100)
            # y_vals = slope * x_vals + intercept
            # plt.plot(x_vals, y_vals, color=rgba_color, linestyle='--')
            plt.plot(factor_vals, normalized_avg_peak_tts, color=rgba_color, linestyle="--")
            
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
                    color='red', verticalalignment='top')
        elif metric == 'pearsonr':
            # Calculate pearson corr
            correlation, _ = stats.pearsonr(all_factor_vals, all_normalized_peak_tts)
            corr_annotation = f"Correlation: {correlation:.4f}"
            plt.text(0.05, 0.95, corr_annotation, transform=plt.gca().transAxes,
                    color='red', verticalalignment='top')

    # Finalize and save the plot
    plt.ylim(-0.1, 1.1)
    plt.gca().set_aspect((max(all_factor_vals) - min(all_factor_vals)) / 1.2)
    plt.xlabel(isolated_factor)
    plt.ylabel("Normalized Avg. Peak Tokens")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    os.makedirs(os.path.join(kwargs['foldername'], "meta_plots"), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], "meta_plots", f"diminish_{isolated_factor}{'_ind' if plot_individual_lines else ''}_{metric}.png"))
    plt.clf()


def plot_correctness_by_isolate_factor(df, plot_against_factor, set_factors, kwargs):
    # Loop across modelnames... x axis is plot_against_factor. Each point is averaged across set_factors

    plt.figure(figsize=(12, 6))

    def exponential_decay(x, a, b): 
       return a * np.exp(-b * x)

    for modelname in df["Model"].unique():
        filtered_data = df[
            df["Model"].str.contains(modelname)
        ]
        base_color = model_colors.get(modelname, "blue")
        rgba_color = mcolors.to_rgba(base_color, alpha=0.8)

        # Ensure there is data to plot
        if filtered_data.empty:
            print(f"No examples found for {modelname}.")
            return

        # Group by both `plot_against_factor` and `set_factors` to calculate means
        grouped_data = filtered_data.groupby([plot_against_factor] + set_factors)["Correct?"].mean().reset_index()

        # Average across `set_factors`
        performance = grouped_data.groupby(plot_against_factor)["Correct?"].mean()

        # Ensure indices are integers for plotting
        performance.index = performance.index.astype(int)
        if len(performance.values) == 0: continue
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            label=modelname,
        )

        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for plot_against_factor_val in performance.index.astype(int):
            sample = filtered_data[filtered_data[plot_against_factor] == str(plot_against_factor_val)]["Correct?"].values
            if len(sample) < 2:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                res = stats.bootstrap((sample,), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
                # ci = np.percentile(np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), [2.5, 97.5])
                ci_lower.append(res.confidence_interval.low)
                ci_upper.append(res.confidence_interval.high)

        # Plot confidence intervals as a shaded region
        plt.fill_between(
            performance.index.astype(int),
            ci_lower,
            ci_upper,
            color=rgba_color,
            alpha=0.3,
        )

        # curve fit
        if plot_against_factor == "N" and len(performance) > 1:
            initial_a_guess = performance.max()
            initial_b_guess = (performance.values[0] - performance.values[1]) / (performance.index[1] - performance.index[0]) if performance.values[1] < performance.values[0] else 0.1

            popt, pcov = curve_fit(exponential_decay, performance.index.astype(int).tolist(), performance.values, p0=[initial_a_guess, initial_b_guess])

            # overlay fitted exponential decay on plot.
            fitted_values = exponential_decay(performance.index.astype(int), *popt)
            plt.plot(
                performance.index.astype(int),
                fitted_values,
                linestyle="--",
                color=rgba_color,
                label=f"d={popt[1]:.2f}",
                alpha=0.5
            )
            
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel(plot_against_factor)
    plt.title(
        f"Correctness vs. {plot_against_factor}"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_{plot_against_factor}.png"
    os.makedirs(os.path.join(kwargs['foldername'], "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(kwargs['foldername'], "isolate_factor", filename)
    )
    plt.clf()
