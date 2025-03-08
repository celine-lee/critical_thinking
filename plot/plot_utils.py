
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
    "Llama-3.1-8B-Instruct": "tomato",
    "Llama-3.3-70B-Instruct-Turbo": "yellow",
    "Meta-Llama-3.1-405B-Instruct-Turbo": "bisque", 
    "Qwen2.5-7B-Instruct": "greenyellow",
    "Qwen2.5-32B-Instruct": "aquamarine",
    "Ministral-8B-Instruct-2410": "orange",
    "gemma-2-9b-it": "brown",
    "DeepSeek-R1-Distill-Llama-8B": "red",
    "DeepSeek-R1-Distill-Llama-70B": "gold",
    "DeepSeek-R1-Distill-Qwen-7B": "seagreen",
    "DeepSeek-R1-Distill-Qwen-32B": "lightseagreen",
    # "DeepSeek-R1-Distill-Qwen-14B": "green",
    # "DeepSeek-R1-Distill-Qwen-1.5B": "red",
    "gpt-4o-mini": "cornflowerblue",
    "gpt-4o": "blue",
    "o3-mini": "purple",
    "DeepSeek-R1": "gray",
}

model_nicknames = {
    "Llama-3.1-8B-Instruct": "Ll3.1-8B",
    "Llama-3.3-70B-Instruct-Turbo": "Ll3.3-70BT",
    "Meta-Llama-3.1-405B-Instruct-Turbo": "Ll3.1-405BT", 
    "Qwen2.5-7B-Instruct": "Qw2.5-7B",
    # "Qwen2.5-14B-Instruct": "Qw2.5-14B",
    "Qwen2.5-32B-Instruct": "Qw2.5-32B",
    # "OLMo-2-1124-7B-Instruct": "OLMO-7B",
    "Ministral-8B-Instruct-2410": "Ministral-8B",
    "gemma-2-9b-it": "Ge2-9B",
    # "DeepSeek-R1-Distill-Qwen-1.5B": "R1-Qw-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B": "R1-Qw-7B",
    # "DeepSeek-R1-Distill-Qwen-14B": "R1-Qw-14B",
    "DeepSeek-R1-Distill-Qwen-32B": "R1-Qw-32B",
    "DeepSeek-R1-Distill-Llama-8B": "R1-Ll-8B",
    "DeepSeek-R1-Distill-Llama-70B": "R1-Ll-70B",
    "gpt-4o-mini": "gpt4om",
    "gpt-4o": "gpt4o",
    "o3-mini": "o3-mini",
    "DeepSeek-R1": "DSR1",
}
colormap = get_cmap("tab10")  # Use a colormap with distinct colors

factor_to_description = {
    "k": "k (DFA size)",
    "N": "N (run length)",
    "m": "m (mult factor)",
    "l": "l (no. lines)",
    "d": "d (depth)",
}

def load_data(data_folder, varnames_and_wanted_vals, experiment_details_parser, kwargs):
    loaded_data = {
        "No gen toks": [],
        "Correct?": [],
        "Predicted": [],
        "True": [],
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
            skip_this = False
            for detail_name, detail_value in experiment_details.items():
                if (varnames_and_wanted_vals[detail_name] is not None) and detail_value not in varnames_and_wanted_vals[detail_name]:
                    skip_this = True
                    break
            if skip_this: continue
            results = json.load(open(experiment_file))
            results = [res for res in results if res["pred_answer"]]

            for varname in varnames_and_wanted_vals:
                loaded_data[varname].extend([experiment_details[varname] for _ in results])

            loaded_data["No gen toks"].extend([ex["generated_tokens"] for ex in results])
            loaded_data["Correct?"].extend([ex["correct"] for ex in results])
            loaded_data["Predicted"].extend([ex["pred_answer"] for ex in results])
            loaded_data["True"].extend([ex["true_answer"] for ex in results])

    # Create a DataFrame for the new data
    df = pd.DataFrame(loaded_data)

    return df

def calculate_precision_recall(sub_df, bucket_name):
    if len(sub_df) == 0:
        return None

    # Group by bucket and compute TP, FP, FN
    grouped = sub_df.groupby(bucket_name, observed=True)

    precision_recall = grouped.apply(lambda x: pd.Series({
        "TP": ((x["Correct?"] == True) & ((x["True"] == True) | (x["True"] == "True"))).sum(),
        "FP": ((x["Correct?"] == False) & ((x["Predicted"] == True) | (x["Predicted"] == "True"))).sum(),
        "FN": ((x["Correct?"] == False) & ((x["True"] == True) | (x["True"] == "True"))).sum(),
    }))
    if 'TP' not in precision_recall: return None
    # Compute precision and recall with safe division
    precision_recall["Precision"] = precision_recall["TP"] / (precision_recall["TP"] + precision_recall["FP"])
    precision_recall["Recall"] = precision_recall["TP"] / (precision_recall["TP"] + precision_recall["FN"])

    # Handle cases where all predictions were negative
    precision_recall.loc[(precision_recall["TP"] == 0) & (precision_recall["FP"] == 0), "Precision"] = 1.0
    precision_recall.loc[(precision_recall["TP"] == 0) & (precision_recall["FN"] == 0), "Recall"] = 0.0

    # Fill any remaining NaN values with 0 (if there was no positive prediction at all)
    precision_recall.fillna(0, inplace=True)

    # Reset index for merging
    precision_recall = precision_recall.reset_index()

    return precision_recall

def calculate_buckets(sub_df, n_buckets, bucket_by="No gen toks", bucket_name="Length Bucket", y_axis="Correct?", groupby_key="Model", get_precision_metrics=True):
    if len(sub_df) == 0:
        return None, None

    unique_lengths = sub_df[bucket_by].unique()

    if len(unique_lengths) == 1:
        # Assign everything to a single bucket
        sub_df[bucket_name] = f"({unique_lengths[0]}, {unique_lengths[0]})"
        bucket_avg = (
            sub_df.groupby([groupby_key, bucket_name], observed=True)[y_axis]
            .mean()
            .reset_index()
        )
        bucket_avg[bucket_name + " Center"] = unique_lengths[0]  # Single center
        bucket_avg[y_axis] = bucket_avg[y_axis].astype(float)
    else:
        # Normal binning process
        if len(unique_lengths) < n_buckets:
            sub_df.loc[:, bucket_name] = pd.qcut(
                sub_df[bucket_by], q=len(unique_lengths) + 1, duplicates="drop"
            )
        else:
            unique_vals, counts = np.unique(sub_df[bucket_by], return_counts=True)
            total_count = len(sub_df)
            cumulative_counts = np.cumsum(counts)

            boundaries = [unique_vals[0]]
            target_size = total_count / n_buckets

            for b in range(1, n_buckets):
                cutoff = b * target_size
                idx = np.searchsorted(cumulative_counts, cutoff, side="left")

                if idx >= len(unique_vals):
                    idx = len(unique_vals) - 1

                while idx < len(unique_vals) and unique_vals[idx] <= boundaries[-1]:
                    idx += 1

                if idx >= len(unique_vals):
                    break

                boundaries.append(unique_vals[idx])

            if boundaries[-1] < unique_vals[-1]:
                boundaries.append(unique_vals[-1])

            boundaries = np.unique(boundaries)

            sub_df.loc[:, bucket_name] = pd.cut(
                sub_df[bucket_by], bins=boundaries, include_lowest=True, duplicates="drop"
            )

        bucket_avg = (
            sub_df.groupby([groupby_key, bucket_name], observed=True)[y_axis]
            .mean()
            .reset_index()
        )

        bucket_avg[bucket_name + " Center"] = bucket_avg[bucket_name].apply(
            lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan
        ).astype(float)

        bucket_avg[y_axis] = bucket_avg[y_axis].astype(float)

    # Group the original sub_df by the same keys and compute std and count.
    grouped = sub_df.groupby([groupby_key, bucket_name], observed=True)[y_axis]
    bucket_sem = grouped.std() / np.sqrt(grouped.count())
    bucket_sem = bucket_sem.reset_index().rename(columns={y_axis: "sem"})
    bucket_avg = bucket_avg.merge(bucket_sem, on=[groupby_key, bucket_name], how="left")
    bucket_avg["ci95"] = bucket_avg["sem"] * 1.96

    if get_precision_metrics:
        # Compute precision and recall
        precision_recall = calculate_precision_recall(sub_df, bucket_name)

        # Merge precision-recall data
        bucket_avg = bucket_avg.merge(precision_recall, on=bucket_name, how="left")

    sub_df = sub_df.merge(bucket_avg, on=[groupby_key, bucket_name], suffixes=('', '_mean'))

    return bucket_avg, sub_df

def global_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_buckets", type=int, default=5)
    parser.add_argument("--num_gens", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    # parser.add_argument("--get_isolated", nargs='+')
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--delete_old", action="store_true")
    parser.add_argument("--only_meta", action="store_true")
    return parser

def plot_length_generated(df, kwargs, by_factor=None):
    # Separate data by model size
    model_data = {
        model_name: df[df["Model"] == model_name].sort_values(by="No gen toks")
        # model_name: df[df["Model"].str.contains(model_name)].sort_values(by="No gen toks")
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
    bucket_avg, filtered_data = calculate_buckets(filtered_data, kwargs['n_buckets'])
    if filtered_data is None: return None
    if len(bucket_avg) == 0: return None
        
    # Find the index of the maximum value
    index_peak = np.argmax(bucket_avg["Correct?"])
    peak_ttoks = bucket_avg["Length Bucket Center"][index_peak]
    best_performance = bucket_avg["Correct?"][index_peak]

    # Find the index of the best precision and recall
    index_best_precision = np.argmax(bucket_avg["Precision"])
    index_best_recall = np.argmax(bucket_avg["Recall"])

    best_precision = bucket_avg["Precision"][index_best_precision]
    best_recall = bucket_avg["Recall"][index_best_recall]

    # and maximum value for probability mass of incorrect sequences
    index_peak_incorrect = np.argmax(1 - bucket_avg["Correct?"]) #  bc of monte carlo we know probability mass is here, and we bucketed into even sizes...
    peak_ttoks_incorrect = bucket_avg["Length Bucket Center"][index_peak_incorrect]
    
    sem_values = filtered_data.groupby("Length Bucket Center", observed=True)["Correct?"].apply(stats.sem)
    # Calculate confidence intervals
    ci = sem_values * 1.96  # For 95% confidence
    ci = sem_values.reindex(bucket_avg["Length Bucket Center"]).fillna(0)

    if kwargs['only_meta']: 
        return (peak_ttoks.item(), peak_ttoks_incorrect.item(), best_performance.item(), best_precision, best_recall, (best_performance - ci.values[index_peak]).item() > kwargs['compute_random'](set_factor_values))
    # Plot the average correctness for each model size and method
    plt.plot(
        bucket_avg["Length Bucket Center"],
        bucket_avg["Correct?"],
        color=rgba_color,
        label=label + f"({len(filtered_data)})",
    )

    plt.fill_between(
        bucket_avg["Length Bucket Center"],
        bucket_avg["Correct?"] - ci.values,
        bucket_avg["Correct?"] + ci.values,
        color=rgba_color,
    )

    if kwargs["plot_incorrect"]:
        plt.plot(
            bucket_avg["Length Bucket Center"],
            1 - bucket_avg["Correct?"], # is this right?
            color="red", alpha=color_intensity,
            label=label + " incorrect" + f"({len(filtered_data)})",
        )

        plt.fill_between(
            bucket_avg["Length Bucket Center"],
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
    return (peak_ttoks.item(), peak_ttoks_incorrect.item(), best_performance.item(), best_precision, best_recall, (best_performance - ci.values[index_peak]).item() > kwargs['compute_random'](set_factor_values))

def plot_correctness_by_ttoks_isolate_factor(df, factor_set_values, isolated_factor, kwargs):
    # Filter the data for the specified factor_set_values
    bool_filter = True
    for factor_name, factor_val in factor_set_values.items():
        bool_filter = bool_filter & (df[factor_name] == factor_val)
    filtered_data = df[bool_filter]

    if filtered_data.empty:
        print(f"No examples found for: {factor_set_values}.")
        return None

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
            (peak_ttoks, peak_ttoks_incorrect, peak_acc, peak_precision, peak_recall, task_doable) = plot_results
            if task_doable:
                factor_val_to_peak_ttoks.append((factor_value, peak_ttoks))
            factor_val_to_peak_ttoks_incorrect.append((factor_value, (peak_ttoks, peak_ttoks_incorrect)))
            factor_val_to_peak_acc.append((factor_value, (peak_acc, peak_precision, peak_recall)))
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

def plot_normalized_correctness_by_ttoks(df, kwargs, include_legend=False):
    """
    1. For each (Model, k, N):
       - Bucket the data by 'No gen toks'.
       - Find the bucket with peak correctness (peak bucket center).
       - Shift & scale x -> domain of size 1, with peak at 0.
       - Normalize y -> [0,1] by min & max correctness in that subset.
       - Store the resulting curve (x, y) in a list.

    2. For each Model:
       - Combine all (k, N) curves by interpolation onto a common x-grid.
       - Average them pointwise, compute SEM -> 95% CI.
       - Plot one aggregated curve with confidence band.
    """

    # A place to store all normalized curves for each model.
    # curves_by_model[model_name] = list of (x_array, y_array) for each (k,N).
    curves_by_model = {}

    # --- STEP 1: Build normalized curves per (Model, k, N) ---
    for model_name in df["Model"].unique():
        model_subdf = df[df["Model"] == model_name]
        curves_list = []

        # Group by (k, N)
        for (k_val, n_val), group_kN in model_subdf.groupby(["k", "N"]):
            if group_kN.empty:
                continue

            # 1a) Bucket in original domain to find peak bucket
            bucket_avg, _ = calculate_buckets(
                group_kN, 
                n_buckets=kwargs["n_buckets"], 
                bucket_by="No gen toks", 
                bucket_name="Toks Bucket",
                y_axis="Correct?", 
                groupby_key="Model", 
                get_precision_metrics=False
            )
            if bucket_avg is None or len(bucket_avg) == 0:
                continue

            # 1b) Identify the bucket with the highest correctness
            idx_peak = bucket_avg["Correct?"].idxmax()
            peak_x = bucket_avg.loc[idx_peak, "Toks Bucket Center"]

            # We'll also get the min & max from the bucketed correctness
            # to define a [0,1] range in y for this subset
            min_y = bucket_avg["Correct?"].min()
            max_y = bucket_avg["Correct?"].max()
            if max_y == min_y:
                # Degenerate case: all correctness the same
                max_y = min_y + 1e-9

            # For the x-scaling to size 2, we need the overall min_x and max_x in this subset
            # (Alternatively, you could use the min and max from the bucketed data.)
            raw_min_x = group_kN["No gen toks"].min()
            raw_max_x = group_kN["No gen toks"].max()
            if raw_max_x == raw_min_x:
                raw_max_x = raw_min_x + 1e-9

            # 1c) Build a "curve" from the *bucketed* data or from the raw points?
            #     Usually for interpolation, we want an (x, y) series. We'll do it from the bucketed data.
            #     That way we have ~n_buckets points. Then we do a second interpolation step across (k,N).
            #     Alternatively, you could re-bucket raw points for finer resolution.
            
            curve_x = []
            curve_y = []

            # Sort the bucket_avg by the bucket center
            bucket_sorted = bucket_avg.sort_values("Toks Bucket Center")
            for i, row in bucket_sorted.iterrows():
                original_center = row["Toks Bucket Center"]
                # Shift & scale x so peak -> 0, domain -> size 1
                norm_x = 1. * (original_center - peak_x) / (raw_max_x - raw_min_x)

                original_correct = row["Correct?"]
                # Normalize correctness to [0,1]
                norm_y = (original_correct - min_y) / (max_y - min_y)

                curve_x.append(norm_x)
                curve_y.append(norm_y)

            # We store the resulting curve
            curves_list.append((np.array(curve_x), np.array(curve_y)))

        # Store all (k, N) curves for this model
        curves_by_model[model_name] = curves_list

    # --- STEP 2: Interpolate & Aggregate per Model ---
    plt.figure(figsize=(10, 6))
    all_x_for_plot = []

    for model_name, curve_list in curves_by_model.items():
        if not curve_list:
            continue

        # 2a) Gather all x-values from all (k,N) curves to define a global min/max
        all_x = np.concatenate([c[0] for c in curve_list])
        global_min_x = all_x.min()
        global_max_x = all_x.max()
        if global_min_x == global_max_x:
            global_max_x = global_min_x + 1e-9

        # 2b) Create a common grid
        num_points = 100
        common_grid = np.linspace(global_min_x, global_max_x, num_points)

        # 2c) Interpolate each (k,N) curve onto the common grid
        interpolated_curves = []
        for (x_vals, y_vals) in curve_list:
            # Make sure x_vals is sorted
            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            y_sorted = y_vals[sort_idx]
            interp_y = np.interp(common_grid, x_sorted, y_sorted)
            interpolated_curves.append(interp_y)

        interpolated_curves = np.array(interpolated_curves)  # shape: (num_curves, num_points)

        # 2d) Average & SEM
        mean_curve = interpolated_curves.mean(axis=0)
        sem_curve = stats.sem(interpolated_curves, axis=0)
        ci95 = 1.96 * sem_curve

        # 2e) Plot
        base_color = model_colors.get(model_name, "blue")
        rgba_color = mcolors.to_rgba(base_color, alpha=0.8)
        label = model_nicknames.get(model_name, model_name)

        plt.plot(common_grid, mean_curve, color=rgba_color, marker=",", label=label)
        plt.fill_between(
            common_grid,
            mean_curve - ci95,
            mean_curve + ci95,
            color=rgba_color,
            alpha=0.3
        )

        all_x_for_plot.extend(common_grid)

    # --- STEP 3: Final Plot Settings ---
    # plt.xlabel("Normalized Generation Length")
    # plt.ylabel("Normalized Correctness")
    plt.ylim(0, 1)
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the left edge are off
        labelleft=False) # labels along the bottom edge are off

    if len(all_x_for_plot) > 0:
        min_x, max_x = min(all_x_for_plot), max(all_x_for_plot)
        margin = 0.1 * (max_x - min_x)
        plt.xlim(min_x - margin, max_x + margin)
    else:
        plt.xlim(-1.5, 1.5)

    if include_legend:
        plt.legend(loc="lower right", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    norm_filename = "normalized_corr_vs_len.png"
    os.makedirs(os.path.join(kwargs['foldername'], "meta_plots"), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], "meta_plots", norm_filename), bbox_inches="tight")
    plt.clf()


def make_just_title_and_legend(
    title,
    models,
    filename, 
    kwargs,
    title_fontsize=40,
    legend_fontsize=16
    ):
    """
    Creates a figure with only a title (top-left or center) and a legend on the right.
    'models' should be a list of label strings.
    """
    # Create a figure; adjust figsize as needed
    fig = plt.figure(figsize=(20, 2))
    ax = fig.add_subplot(111)

    # Hide the axes themselves (so you only see title & legend)
    ax.set_axis_off()

    # Create some dummy lines for the legend
    for model in models:
        # We don't actually plot any data; we just need a "handle" for the legend
        ax.plot([], [], label=model_nicknames[model])

    # Place the legend on the right side
    # bbox_to_anchor=(1, 0.5) means anchor the legend at x=1 (right edge), y=0.5 (center)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.), fontsize=legend_fontsize)

    # Center the title at the top; adjust x, y, and fontsize
    fig.suptitle(
        title,
        x=0.5,       # horizontal center
        y=0.95,      # near top
        ha='center', # horizontally centered
        fontsize=title_fontsize
    )

    # Option B: Title at the top-center
    # fig.suptitle(title)

    # Tighten layout so there's minimal extra space
    plt.tight_layout()

    os.makedirs(os.path.join(kwargs['foldername'], "meta_plots"), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], "meta_plots", filename), bbox_inches="tight")
    plt.clf()

def plot_ptt_by_factor(factor_to_peak_ttoks, isolated_factor, kwargs):
    plt.figure(figsize=(12, 6))
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

    corrs = {}

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
            # fv: value of factor , e.g. k=2
            # ptts: list of L*s for all configurations where factor=factor value, e.g. for [(k=2,N=5), (k=2,N=9), (k=2,N=12)]
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

        # plot the normalized averages
        normalized_avg_peak_tts = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in avg_peak_tts]
        all_normalized_avg_peak_tts.extend([(fv, napt) for fv, napt in zip(factor_vals, normalized_avg_peak_tts)])
        plt.plot(factor_vals, normalized_avg_peak_tts, color=rgba_color, linestyle="--")
        # Plot the confidence intervals as a shaded region
        plt.fill_between(
            factor_vals,
            [(ci - min_val) / (max_val - min_val) if max_val != min_val else 0 for ci in ci_lower_bounds],
            [(ci - min_val) / (max_val - min_val) if max_val != min_val else 0 for ci in ci_upper_bounds],
            color=rgba_color,
            alpha=0.2
            )

        # Calculate pearson corr
        correlation, _ = stats.pearsonr(normalized_factor_vals[-len(normalized_peak_tts):], normalized_peak_tts)
        legend_label = f"{model_nicknames[modelname]} (Corr: {correlation:.2f})"
        corrs[model_nicknames[modelname]] = correlation

        sns.scatterplot(x=factor_vals, y=normalized_avg_peak_tts, 
                    marker='o', color=rgba_color, label=legend_label)

    if len(all_factor_vals) == 0 or max(all_factor_vals) == min(all_factor_vals): return {f"Corr(ptts, {isolated_factor})": corrs}

    # Finalize and save the plot
    plt.ylim(-10, 11)
    plt.xlim(min(all_factor_vals)-1, max(all_factor_vals)+1)
    plt.gca().set_aspect((max(all_factor_vals) - min(all_factor_vals)) / 1.2)
    plt.xlabel(factor_to_description[isolated_factor])
    plt.ylabel("Normalized Peak Tokens")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    os.makedirs(os.path.join(kwargs['foldername'], "meta_plots"), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], "meta_plots", f"diminish_{isolated_factor}_ind_err.png"), bbox_inches='tight')
    plt.clf()

    return {f"Corr(ptts, {isolated_factor})": corrs}


from tabulate import tabulate
def ptt_table(factor_model_corrs, output_folder):
    # factors_to_peak_ttoks is a dict of isolated_factor string to its values and peak ttoks. 
    # | model | corr(normalized ptts, k) | corr(normalized ptts, N) | ** any other isolated_factors included  

    # merge them first
    combined_corrs = {}
    for model_corrs in factor_model_corrs:
        for corr_title, correlation_info_dict in model_corrs.items():
            if corr_title not in combined_corrs: combined_corrs[corr_title] = {}
            combined_corrs[corr_title] |= correlation_info_dict

    df = pd.DataFrame(combined_corrs)

    with open(os.path.join(output_folder, "ptt_table.txt"), 'w') as wf:
        wf.write(tabulate(df, headers='keys', tablefmt='psql'))


def acc_table(df, output_folder):
    # | model | accuracy | precision | recall | random |
    # Compute accuracy, precision, and recall per task
    TP = lambda x: (((x["Predicted"] == True) | (x["Predicted"] == "True")) & ((x["True"] == True) | (x["True"] == "True"))).sum() 
    FP = lambda x: (((x["Predicted"] == True) | (x["Predicted"] == "True")) & ((x["True"] == False) | (x["True"] == "False"))).sum() 
    FN = lambda x: (((x["Predicted"] == False) | (x["Predicted"] == "False")) & ((x["True"] == True) | (x["True"] == "True"))).sum() 
    model_metrics = df.groupby("Model").apply(lambda x: pd.Series({
        "Accuracy": x["Correct?"].mean(),
        "Precision": TP(x) / (TP(x) + FP(x)),
        "Recall": TP(x) / (TP(x) + FN(x)),
        "Random": ((x["True"] == True) | (x["True"] == "True")).sum() / len(x)
    })).reset_index()
    
    with open(os.path.join(output_folder, "accuracy_table.txt"), 'w') as wf:
        wf.write(tabulate(model_metrics, headers='keys', tablefmt='psql'))
    


def plot_correctness_by_isolate_factor(df, plot_against_factor, set_factors, kwargs):
    # Loop across modelnames... x axis is plot_against_factor. Each point is averaged across set_factors

    plt.figure(figsize=(12, 6))

    def exponential_decay(x, a, b): 
       return a * np.exp(-b * x)

    for modelname in df["Model"].unique():
        filtered_data = df[
            df["Model"] == modelname
            # df["Model"].str.contains(modelname)
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
            label=model_nicknames[modelname],
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
    plt.xlabel(factor_to_description[plot_against_factor])
    plt.title(
        f"Correctness vs. {factor_to_description[plot_against_factor]}"
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

def plot_peak_accuracy_heatmap(experiment_to_peak_accuracy, title, kwargs):
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
            heatmap_data.append({"k": k, "N": N, title: peak_accuracy})
        if len(heatmap_data) == 0: continue
        # Convert to DataFrame
        df = pd.DataFrame(heatmap_data)
        df["k"] = df["k"].astype(int)
        df["N"] = df["N"].astype(int)


        # Pivot the DataFrame for the heatmap
        heatmap_df = df.pivot_table(index="N", columns="k", values=title, aggfunc="mean")
        heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=cmap, cbar=True, center=0)
        plt.title(f"{title} Heatmap ({modelname})")
        plt.xlabel(factor_to_description["k"])
        plt.ylabel(factor_to_description["N"])

        # Save the plot
        os.makedirs(os.path.join(kwargs['foldername'], "heatmaps"), exist_ok=True)
        plt.savefig(os.path.join(kwargs['foldername'], "heatmaps", f"{modelname}_{title}.png"))
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
        if len(heatmap_data) == 0: continue
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

        plt.xlabel(factor_to_description["k"])
        plt.ylabel(factor_to_description["N"])

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
    plt.xlabel(factor_to_description["k"])
    plt.ylabel(factor_to_description["N"])

    # Save the plot
    os.makedirs(os.path.join(kwargs['foldername'], "heatmaps"), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], "heatmaps", "grouped_peak_ttok_diffs.png"))
    plt.clf()
