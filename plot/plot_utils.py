
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
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.cm import get_cmap
from scipy.optimize import curve_fit
from scipy.stats import sem, gaussian_kde
import seaborn as sns
sns.set("talk")

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger

sys.excepthook = debughook


model_colors = {
    # "Llama-3.1-8B-Instruct": "purple",
    "Qwen2.5-32B-Instruct": "blue",
    # "Qwen2.5-14B-Instruct": "brown",
    "Qwen2.5-7B-Instruct": "purple",
    # "OLMo-2-1124-7B-Instruct": "black",
    "Ministral-8B-Instruct-2410": "orange",
    "gemma-2-9b-it": "brown",
}

model_nicknames = {
    # "Llama-3.1-8B-Instruct": "Ll3.1-8B",
    "Qwen2.5-32B-Instruct": "Qw2.5-32B",
    # "Qwen2.5-14B-Instruct": "Qw2.5-14B",
    "Qwen2.5-7B-Instruct": "Qw2.5-7B",
    # "OLMo-2-1124-7B-Instruct": "OLMO-7B",
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
    # Separate data by model size
    model_data = {
        model_name: df[df["Model"].str.contains(model_name)].sort_values(by="No gen toks")
        for model_name in model_colors
    }

    # If by_factor is specified, ensure it exists in the DataFrame
    if by_factor and by_factor not in df.columns:
        raise ValueError(f"'{by_factor}' is not a column in the provided DataFrame.")

    # Prepare the figure
    plt.figure(figsize=(12, 6))

    tick_positions = []
    labels = []
    plotted_something = False

    # Iterate over models and optionally by_factor
    for i, (model, model_df) in enumerate(model_data.items(), start=1):
        if by_factor:
            # Group data by the factor
            grouped = model_df.groupby(by_factor)
            factor_values = sorted(grouped.groups.keys(), key=int)

            for j, factor_value in enumerate(factor_values):
                # Extract data for boxplot
                data = grouped.get_group(factor_value)["No gen toks"].tolist()
                if not data:
                    continue

                # Define position for the boxplot (spread models apart)
                position = [i + j * 0.2]
                tick_positions.append(i + j * 0.2)
                labels.append(f"{model_nicknames[model]} ({by_factor}={factor_value})")

                # Plot the boxplot
                plt.boxplot(
                    [data],
                    positions=position,
                    widths=0.15,
                    patch_artist=True,
                    boxprops=dict(facecolor=model_colors[model], color=model_colors[model]),
                    medianprops=dict(color="black"),
                    showfliers=False,
                )
                plotted_something = True

        else:
            # Extract data for the model
            data = model_df["No gen toks"].tolist()
            if not data:
                continue

            # Define position for the boxplot
            position = [i]
            tick_positions.append(i)
            labels.append(model_nicknames[model])

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
    plt.xticks(ticks=tick_positions, labels=labels, rotation=45, ha="right")
    plt.xlabel("Method" if not by_factor else f"Method (Grouped by {by_factor})")
    plt.ylabel("No. Generate Tokens")

    # Legend for the models
    # legend_elements = [
    #     Line2D([0], [0], color=model_colors[model], lw=2, label=model)
    #     for model in model_colors
    # ]
    # plt.legend(handles=legend_elements, loc="best", fancybox=True, shadow=True)

    # Save the plot
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.savefig(
        os.path.join(kwargs['foldername'], f"genlength_boxplot_by{by_factor}.png"),
        bbox_inches="tight"
    )
    plt.clf()

def plot_correctness_by_ttoks(filtered_data, set_factor_values, label, rgba_color, color_intensity, is_subplot, kwargs):
    bucket_avg, filtered_data = calculate_buckets_samesize(filtered_data, kwargs['n_buckets'])
    if filtered_data is None: return False
    if len(bucket_avg) == 0: return False
        
    # Find the index of the maximum value
    index_peak = np.argmax(bucket_avg["Correct?"])
    peak_ttoks = bucket_avg["Bucket Center"][index_peak]
    best_performance = bucket_avg["Correct?"][index_peak]
    # and maximum value for probability mass of incorrect sequences
    index_peak_incorrect = np.argmax(1 - bucket_avg["Correct?"]) #  bc of monte carlo we know probability mass is here, and we bucketed into even sizes...
    peak_ttoks_incorrect = bucket_avg["Bucket Center"][index_peak_incorrect]
    
    sem_values = filtered_data.groupby("Bucket Center", observed=True)["Correct?"].apply(stats.sem)
    # Calculate confidence intervals
    ci = sem_values * 1.96  # For 95% confidence
    ci = sem_values.reindex(bucket_avg["Bucket Center"]).fillna(0)

    if kwargs['only_meta']: 
        return (peak_ttoks.item(), peak_ttoks_incorrect.item(), best_performance.item(), (best_performance - ci.values[index_peak]).item() > kwargs['compute_random'](set_factor_values))
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

    if kwargs["plot_incorrect"]:
        plt.plot(
            bucket_avg["Bucket Center"],
            1 - bucket_avg["Correct?"], # is this right?
            color="red", alpha=color_intensity,
            label=label + " incorrect",
        )

        plt.fill_between(
            bucket_avg["Bucket Center"],
            1 - (bucket_avg["Correct?"] + ci.values),
            1 - (bucket_avg["Correct?"] - ci.values), # flip bounds 
            color="red", alpha=color_intensity,
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
            f"{set_factor_values}"
            # f"Average Correctness vs. No. of Generated Tokens {set_factor_values}"
        )
        plt.legend(loc="best", fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save and clear the figure
        filename = "_".join(f"{factor_name}{factor_value}" for factor_name, factor_value in set_factor_values.items()) + f"{'_wincorrect' if kwargs['plot_incorrect'] else ''}.png"
        os.makedirs(os.path.join(kwargs['foldername'], "isolate_factor"), exist_ok=True)
        plt.savefig(
            os.path.join(kwargs['foldername'], "isolate_factor", filename)
        )
        plt.clf()
    return (peak_ttoks.item(), peak_ttoks_incorrect.item(), best_performance.item(), (best_performance - ci.values[index_peak]).item() > kwargs['compute_random'](set_factor_values))

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

    factor_val_to_peak_ttoks, factor_val_to_peak_ttoks_incorrect, factor_val_to_peak_acc = [], [], []
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

        plot_results = plot_correctness_by_ttoks(factor_filtered_data, factor_set_values | {isolated_factor: factor_value}, label, rgba_color, color_intensity, True, kwargs)
        if plot_results:
            used_vals.append(factor_value)
            (peak_ttoks, peak_ttoks_incorrect, peak_acc, task_doable) = plot_results
            if task_doable:
                factor_val_to_peak_ttoks.append((factor_value, peak_ttoks))
            factor_val_to_peak_ttoks_incorrect.append((factor_value, (peak_ttoks, peak_ttoks_incorrect)))
            factor_val_to_peak_acc.append((factor_value, peak_acc))
    if len(factor_val_to_peak_ttoks) == 0: return factor_val_to_peak_ttoks, factor_val_to_peak_ttoks_incorrect, factor_val_to_peak_acc

    if kwargs['only_meta']: 
        return factor_val_to_peak_ttoks, factor_val_to_peak_ttoks_incorrect, factor_val_to_peak_acc

    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Average Correctness")
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"{isolated_factor}{''.join(str(uv) for uv in used_vals)}_"
    filename += "_".join(f"{factor_name}{factor_value}" for factor_name, factor_value in factor_set_values.items()) + f"{'_wincorrect' if kwargs['plot_incorrect'] else ''}.png"
    os.makedirs(os.path.join(kwargs['foldername'], "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(kwargs['foldername'], "isolate_factor", filename)
    )
    plt.clf()

    return factor_val_to_peak_ttoks, factor_val_to_peak_ttoks_incorrect, factor_val_to_peak_acc

def plot_ptt_by_factor(factor_to_peak_ttoks, isolated_factor, plot_individual_lines, kwargs, plot_error=False, metric="pearsonr"):
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
    all_factor_vals_normalized = []
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
        for fv, ptts in sorted(fv_to_ptts_avged.items(), key = lambda item: item[0]):
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
        min_factor_val = min(factor_vals)
        max_factor_val = max(factor_vals)
        normalized_factor_vals = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in all_factor_vals[-len(normalized_peak_tts):]]
        all_factor_vals_normalized.extend(normalized_factor_vals)
        legend_label = model_nicknames[modelname]

        # plot the normalized averageds
        normalized_avg_peak_tts = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in avg_peak_tts]
        all_normalized_avg_peak_tts.extend([(fv, napt) for fv, napt in zip(factor_vals, normalized_avg_peak_tts)])
        if plot_individual_lines:
            plt.plot(factor_vals, normalized_avg_peak_tts, color=rgba_color, linestyle="--")
            
            if plot_error:
                # Plot the confidence intervals as a shaded region
                plt.fill_between(
                    factor_vals,
                    [(ci - min_val) / (max_val - min_val) if max_val != min_val else 0 for ci in ci_lower_bounds],
                    [(ci - min_val) / (max_val - min_val) if max_val != min_val else 0 for ci in ci_upper_bounds],
                    color=rgba_color,
                    alpha=0.2
                )

            if metric == 'mse':
                # Calculate Mean Squared Error
                predicted_vals = slope * np.array(normalized_factor_vals[-len(normalized_peak_tts):]) + intercept
                mse = np.mean((np.array(normalized_peak_tts) - predicted_vals) ** 2)
                legend_label = f"{model_nicknames[modelname]} (MSE: {mse_annotation:.2f})"
            elif metric == 'pearsonr':
                # Calculate pearson corr
                correlation, _ = stats.pearsonr(normalized_factor_vals[-len(normalized_peak_tts):], normalized_peak_tts)
                legend_label = f"{model_nicknames[modelname]} (Corr: {correlation:.2f})"

        sns.scatterplot(x=factor_vals, y=normalized_avg_peak_tts, 
                    marker='o', color=rgba_color, label=legend_label)
        # if plot_error:

        #     # Calculate yerr in (2, n) format
        #     yerr = np.array([
        #         [avg_ptt - ci_lower for avg_ptt, ci_lower in zip(avg_peak_tts, ci_lower_bounds)],  # Lower error
        #         [ci_upper - avg_ptt for avg_ptt, ci_upper in zip(avg_peak_tts, ci_upper_bounds)]   # Upper error
        #     ])
        #     plt.errorbar(
        #         factor_vals,
        #         normalized_avg_peak_tts,
        #         yerr=yerr,
        #         fmt="o",
        #         color=rgba_color,
        #         alpha=0.7
        #     )

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
            predicted_vals = slope * np.array(all_factor_vals_normalized) + intercept
            mse = np.mean((np.array(all_normalized_peak_tts) - predicted_vals) ** 2)
            # Annotate the MSE on the plot
            mse_annotation = f"MSE: {mse:.4f}"
            plt.text(0.05, 0.95, mse_annotation, transform=plt.gca().transAxes,
                    color='red', verticalalignment='top')
        elif metric == 'pearsonr':
            # Calculate pearson corr
            correlation, _ = stats.pearsonr(all_factor_vals_normalized, all_normalized_peak_tts)
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
    plt.savefig(os.path.join(kwargs['foldername'], "meta_plots", f"diminish_{isolated_factor}{'_ind' if plot_individual_lines else ''}{'_err' if plot_error else ''}_{metric}.png"))
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

def plot_cdfs(df, desired_factors, kwargs):
    # Filter dataset for the given desired_factors
    bool_filter = True
    for factor_name, factor_val in desired_factors.items():
        bool_filter = bool_filter & (df[factor_name] == factor_val)
    filtered_df = df[bool_filter]

    # Separate correct and incorrect samples
    correct = filtered_df[filtered_df["Correct?"] == True]["No gen toks"].values
    incorrect = filtered_df[filtered_df["Correct?"] == False]["No gen toks"].values

    # Compute empirical CDFs
    correct_sorted = np.sort(correct)
    incorrect_sorted = np.sort(incorrect)

    correct_cdf = np.arange(1, len(correct_sorted) + 1) / len(correct_sorted)
    incorrect_cdf = np.arange(1, len(incorrect_sorted) + 1) / len(incorrect_sorted)

    # Plot CDFs
    plt.figure(figsize=(8, 6))
    plt.plot(correct_sorted, correct_cdf, label="Correct", linestyle="-", linewidth=2)
    plt.plot(incorrect_sorted, incorrect_cdf, label="Incorrect", linestyle="--", linewidth=2)

    # # Compute and plot probability of correctness vs. token length
    # all_tokens = np.concatenate([correct, incorrect])
    # correctness_labels = np.concatenate([np.ones_like(correct), np.zeros_like(incorrect)])

    # # Kernel Density Estimation (KDE) for smooth probability curve
    # kde_correct = gaussian_kde(correct, bw_method=0.2)
    # kde_all = gaussian_kde(all_tokens, bw_method=0.2)
    
    # x_vals = np.linspace(min(all_tokens), max(all_tokens), 300)
    # correctness_prob = kde_correct(x_vals) / (kde_all(x_vals) + 1e-9)  # Avoid division by zero

    # plt.plot(x_vals, correctness_prob, label="P(Correct | No gen toks)", color="black", linestyle="-.")

    # Labels and legend
    plt.xlabel("Number of Generated Tokens")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDFs of Token Length for Correct vs. Incorrect Answers ({desired_factors})")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.join(kwargs['foldername'], "prob_distrs"), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], "prob_distrs", ''.join(f"{k}{v}" for k, v in desired_factors.items()) + ".png"))
    plt.clf()

def plot_peak_accuracy_heatmap(experiment_to_peak_accuracy, kwargs):
    """
    Heatmap of peak accuracies across k and N.
    
    experiment_to_peak_accuracy: dict mapping modelname -> list of ((k, *, N) , accuracy at peak for that factor)
    """
    for modelname, results in experiment_to_peak_accuracy.items():
        # Define the custom colormap
        base_color = model_colors.get(modelname, "blue")
        cmap = LinearSegmentedColormap.from_list(
            "custom_colormap", 
            ["red", "white", base_color]
        )

        # Prepare data for the heatmap
        heatmap_data = []
        for (dfa_details, peak_accuracy) in results.items():
            k = dfa_details[0]
            N = dfa_details[-1]
            heatmap_data.append({"k": k, "N": N, "Peak Accuracy": peak_accuracy})

        # Convert to DataFrame
        df = pd.DataFrame(heatmap_data)
        df["k"] = df["k"].astype(int)
        df["N"] = df["N"].astype(int)


        # Pivot the DataFrame for the heatmap
        heatmap_df = df.pivot_table(index="N", columns="k", values="Peak Accuracy", aggfunc="mean")
        heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=cmap, cbar=True, center=0)
        plt.title(f"Peak Accuracy Heatmap ({modelname})")
        plt.xlabel("k")
        plt.ylabel("N")

        # Save the plot
        os.makedirs(os.path.join(kwargs['foldername'], "heatmaps"), exist_ok=True)
        plt.savefig(os.path.join(kwargs['foldername'], "heatmaps", f"{modelname}_peak_acc.png"))
        plt.clf()

def plot_peak_token_difference_heatmap(experiment_to_tok_differences, kwargs):
    """
    Heatmap of peak token differences between correct and incorrect across k and N.
    
    experiment_to_tok_differences: dict mapping modelname -> list of ((k, *, N) , (peak correct, peak incorrect))
    """
    grouped_data = []

    for modelname, results in experiment_to_tok_differences.items():
        # Define the custom colormap
        base_color = model_colors.get(modelname, "blue")
        cmap = LinearSegmentedColormap.from_list(
            "custom_colormap", 
            ["red", "white", base_color]
        )

        # Prepare data for the heatmap
        heatmap_data = []
        for (dfa_details, peak_ttoks) in results.items():
            k = dfa_details[0]
            N = dfa_details[-1]
            token_diff = peak_ttoks[0] - peak_ttoks[1]
            heatmap_data.append({"k": k, "N": N, "Token Diff": token_diff})
            # grouped_data.append({"Model": modelname, "k": k, "N": N, "Token Diff": token_diff})

        # Convert to DataFrame
        df = pd.DataFrame(heatmap_data)

        # Normalize Token Diff for the current model
        if not df["Token Diff"].empty:
            min_val = df["Token Diff"].min()
            max_val = df["Token Diff"].max()
            df["Normalized Token Diff"] = (
                (df["Token Diff"] - min_val) / (max_val - min_val)
                if max_val != min_val else 0
            )
            grouped_data.extend(df[["k", "N", "Normalized Token Diff"]].to_dict("records"))


        # Ensure k and N are integers for proper sorting
        df["k"] = df["k"].astype(int)
        df["N"] = df["N"].astype(int)

        # Pivot the DataFrame for the heatmap
        heatmap_df = df.pivot_table(index="N", columns="k", values="Token Diff", aggfunc="mean")
        # Sort the pivot_table by index (N) and columns (k)
        heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)


        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=cmap, cbar=True, center=0)
        plt.title(f"Token Difference ({modelname})")

        plt.xlabel("k")
        plt.ylabel("N")

        # Save the plot
        os.makedirs(os.path.join(kwargs['foldername'], "heatmaps"), exist_ok=True)
        plt.savefig(os.path.join(kwargs['foldername'], "heatmaps", f"{modelname}_peak_ttok_diffs.png"))
        plt.clf()

    # Convert to DataFrame
    df = pd.DataFrame(grouped_data)

    # Ensure k and N are integers for proper sorting
    df["k"] = df["k"].astype(int)
    df["N"] = df["N"].astype(int)

    # Compute averages across models
    avg_df = df.groupby(["N", "k"], as_index=False)["Normalized Token Diff"].mean()

    # Pivot the averaged data for heatmap
    heatmap_df = avg_df.pivot_table(index="N", columns="k", values="Normalized Token Diff", aggfunc="mean")

    # Sort index and columns
    heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)

    # Define the colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap", ["red", "white", "green"])

    # Plot the grouped heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=cmap, cbar=True, center=0)
    plt.title("Normalized Token Difference")
    plt.xlabel("k")
    plt.ylabel("N")

    # Save the plot
    os.makedirs(os.path.join(kwargs['foldername'], "heatmaps"), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], "heatmaps", "grouped_peak_ttok_diffs.png"))
    plt.clf()