import json
import re
import os
import shutil
import pandas as pd
import glob
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy import stats
from matplotlib import colors as mcolors

from plot_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_buckets", type=int, default=4)
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--delete_old", action="store_true")
    parser.add_argument("--tasks", nargs='+', default=['dyck', 'array_idx', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies', 'cruxeval', 'logical_deduction'])
    parser.add_argument("--select_tasks", nargs='+', default=['array_idx', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies'])

    args = parser.parse_args()
    args.foldername = os.path.join(
        f"all_tasks_graphs_{args.n_buckets}buckets"
    )

    return args

def load_task_data(taskname, compute_random, foldername_parser, dfa_factors, output_folder, filter_stddev_count=1):
    loaded_data = {
        "No gen toks": [],
        "Correct?": [],
        "Predicted": [],
        "True": [],
        "prompt": [],
    }
    for varname in dfa_factors:
        loaded_data[varname] = []

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(output_folder, f"k*")):
        for experiment_file in glob.glob(os.path.join(subfolder, "*")):
            if f"T0.0" not in experiment_file:
                continue
            
            experiment_details = foldername_parser(experiment_file)
            if experiment_details is None: continue
            if experiment_details["Model"] not in model_colors: continue
            # print(experiment_file)
            results = json.load(open(experiment_file))
            results = [res for res in results if res["pred_answer"]]

            for varname in dfa_factors:
                loaded_data[varname].extend([experiment_details[varname] for _ in results])

            loaded_data["No gen toks"].extend([ex["generated_tokens"] for ex in results])
            loaded_data["Correct?"].extend([ex["correct"] for ex in results])
            loaded_data["Predicted"].extend([ex["pred_answer"] for ex in results])
            loaded_data["True"].extend([ex["true_answer"] for ex in results])
            loaded_data["prompt"].extend([ex["query"] for ex in results])
    
    loaded_data["task"] = [taskname for _ in loaded_data["prompt"]]
    # Create a DataFrame for the new data
    df = pd.DataFrame(loaded_data)

    if filter_stddev_count is not None:
        # Remove models that fail in any configuration
        grouped_df = df.groupby("Model")
        models_to_remove = set()

        for modelname, group in grouped_df:
            accuracy = group["Correct?"].mean()
            avg_param_combo = {"k": group["k"].mean(), "N": group["N"].mean()}
            random_baseline = compute_random(avg_param_combo)
            stddev = group["Correct?"].std() if len(group) > 1 else 0
            
            if accuracy < random_baseline + filter_stddev_count * stddev or group["No gen toks"].nunique() < 5:
                models_to_remove.add(modelname) 
        df = df[~df["Model"].isin(models_to_remove)].reset_index(drop=True)
        
    return df

def load_data(args, kwargs, filter_stddev_count=1):
    all_df = None
    for task in args.tasks:
        compute_random, foldername_parser, dfa_factors_order, output_folder = get_task_info(task)
        task_df = load_task_data(task, compute_random, foldername_parser, list(dfa_factors_order.keys()) + ["Model"], output_folder)
        if all_df is not None:
            all_df = pd.concat([all_df, task_df])
        else: all_df = task_df
    return all_df

def plot_correctness_by_ttoks(df, kwargs):
    """
    Per (task, model): 
           * Bucket then normalize y (accuracy).
           * Identify the bucket with the peak correctness (record its center and value).
           * Plot the resulting line along with a scatter point at the peak.
    
    Parameters:
      df: DataFrame containing the data.
      kwargs: Dictionary with additional arguments (expects at least "n_buckets" and "foldername" and "to_highlight").
    """
    grouped = df.groupby(["task", "Model"])
    
    plt.figure(figsize=(10, 6))
    for (task, modelname), group in grouped:
        if (task, modelname) in kwargs["to_highlight"]: continue 
            
        if group.empty:
            continue
        
        task_model_df = {
            "No gen toks": [],
            "Correct?": []
        }
        
        # Bucket the data and normalize correctness values
        bucket_avg, _ = calculate_buckets(
            group,
            n_buckets=kwargs["n_buckets"],
            bucket_by="No gen toks",
            bucket_name="Toks Bucket",
            y_axis="Correct?",
            groupby_key="Model",
        )
        
        if bucket_avg is None or len(bucket_avg) == 0:
            continue
        
        max_y = bucket_avg["Correct?"].max()
        min_y = bucket_avg["Correct?"].min()
        
        for _, row in bucket_avg.iterrows():
            x_val = row["Toks Bucket Center"]
            y_val = row["Correct?"]
            y_val = (y_val - min_y) / (max_y - min_y) if max_y != min_y else y_val
            
            task_model_df["No gen toks"].append(x_val)
            task_model_df["Correct?"].append(y_val)
    
        task_model_df = pd.DataFrame(task_model_df)
        
        peak_row = bucket_avg.loc[bucket_avg["Correct?"].idxmax()]
        peak_x, peak_y_normalized = peak_row["Toks Bucket Center"], (peak_row["Correct?"] - min_y) / (max_y - min_y) if max_y != min_y else peak_row["Correct?"]
    
        label = None
        line_color, dot_color = "gray", "gray"
        dotsize = 3

        plt.plot(task_model_df["No gen toks"], task_model_df["Correct?"], color=line_color, marker=",", alpha=0.3)
        plt.scatter(peak_x, peak_y_normalized, color=dot_color, zorder=dotsize, label=label,  alpha=0.3)

    max_x = 0
    for (task, modelname) in kwargs["to_highlight"]:
        group = df[(df["task"] == task) * (df["Model"] == modelname)]
        if group.empty:
            continue
        
        task_model_df = {
            "No gen toks": [],
            "Correct?": []
        }
        
        # Bucket the data and normalize correctness values
        bucket_avg, _ = calculate_buckets(
            group,
            n_buckets=kwargs["n_buckets"],
            bucket_by="No gen toks",
            bucket_name="Toks Bucket",
            y_axis="Correct?",
            groupby_key="Model",
        )
        
        if bucket_avg is None or len(bucket_avg) == 0:
            continue
        
        max_y = bucket_avg["Correct?"].max()
        min_y = bucket_avg["Correct?"].min()
        
        for _, row in bucket_avg.iterrows():
            x_val = row["Toks Bucket Center"]
            y_val = row["Correct?"]
            y_val = (y_val - min_y) / (max_y - min_y) if max_y != min_y else y_val
            
            task_model_df["No gen toks"].append(x_val)
            task_model_df["Correct?"].append(y_val)
    
        max_x = max(max_x, max(task_model_df["No gen toks"]))
        task_model_df = pd.DataFrame(task_model_df)
        
        peak_row = bucket_avg.loc[bucket_avg["Correct?"].idxmax()]
        peak_x, peak_y_normalized = peak_row["Toks Bucket Center"], (peak_row["Correct?"] - min_y) / (max_y - min_y) if max_y != min_y else peak_row["Correct?"]

        label = model_nicknames.get(modelname, modelname) + f" {task}"
        line_color = model_colors.get(modelname, "blue")
        dot_color = model_colors.get(modelname, "blue")
        dotsize = 5
            
        plt.plot(task_model_df["No gen toks"], task_model_df["Correct?"], color=line_color, marker=",")
        plt.scatter(peak_x, peak_y_normalized, color=dot_color, zorder=dotsize, label=label)
    
    plt.xlabel("Generated Tokens")
    plt.ylabel("Accuracy (normalized)")
    # plt.ylim(0, 1)
    plt.xlim(0, min(5000, max_x))
    plt.legend(loc="lower right", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plot_filename = "corr_vs_len.png"
    os.makedirs(os.path.join(kwargs['foldername']), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], plot_filename), bbox_inches="tight")
    plt.clf()

def plot_correctness_by_ttoks_model_pairs(df, models_and_tasks, kwargs, normalize):
    grouped = df.groupby(["task", "Model"])
    
    models = [models_and_tasks[0], models_and_tasks[1]]
    tasks = models_and_tasks[-1]
    plt.figure(figsize=(10, 6))
    for (task, modelname), group in grouped:
        if modelname not in models: continue
            
        if group.empty:
            continue
        
        task_model_df = {
            "No gen toks": [],
            "Correct?": []
        }
        
        # Bucket the data and normalize correctness values
        bucket_avg, _ = calculate_buckets(
            group,
            n_buckets=kwargs["n_buckets"],
            bucket_by="No gen toks",
            bucket_name="Toks Bucket",
            y_axis="Correct?",
            groupby_key="Model",
        )
        
        if bucket_avg is None or len(bucket_avg) == 0:
            continue
        
        max_y = bucket_avg["Correct?"].max()
        min_y = bucket_avg["Correct?"].min()
        
        for _, row in bucket_avg.iterrows():
            x_val = row["Toks Bucket Center"]
            y_val = row["Correct?"]
            if normalize:
                y_val = (y_val - min_y) / (max_y - min_y) if max_y != min_y else y_val
            
            task_model_df["No gen toks"].append(x_val)
            task_model_df["Correct?"].append(y_val)
    
        task_model_df = pd.DataFrame(task_model_df)
        
        peak_row = bucket_avg.loc[bucket_avg["Correct?"].idxmax()]
        peak_x, peak_y_normalized = peak_row["Toks Bucket Center"], ( - min_y) / (max_y - min_y) if max_y != min_y else peak_row["Correct?"]

        if task in tasks:
            label = model_nicknames.get(modelname, modelname) + f" {task}"
            line_color = model_colors.get(modelname, "blue")
            dot_color = model_colors.get(modelname, "blue")
            dotsize = 5
            opacity=1.0
        else: 
            label = None
            line_color, dot_color = "gray", "gray"
            dotsize = 2
            opacity=0.3
            
        plt.plot(task_model_df["No gen toks"], task_model_df["Correct?"], color=line_color, marker=",", alpha=opacity)
        plt.scatter(peak_x, peak_row["Correct?"], color=dot_color, zorder=dotsize, label=label, alpha=opacity)
    
    plt.xlabel("Generated Tokens")
    plt.ylabel(f"Accuracy{' (normalized)' if normalize else ''}")
    plt.ylim(0, 1)
    plt.xlim(0, 5000)
    plt.legend(loc="lower right", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plot_filename = f"{model_nicknames[models[0]]}_{model_nicknames[models[1]]}{'_norm' if normalize else ''}.png"
    os.makedirs(os.path.join(kwargs['foldername']), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], plot_filename), bbox_inches="tight")
    plt.clf()

def plot_fig1_on_ax(ax, task, df, kwargs, include_raw, clamp_upper):
    """
    Plot the fig1 curve for a single task onto a given axis (ax),
    with customized x-axis ticks: the tick at 0 (the normalized peak) is labeled 'L*',
    and one additional tick, at the average normalized value corresponding to raw 0 across all curves, is labeled '0'.
    """
    # Build normalized curves per (Model, k, N)
    curves_by_model = {}
    # normalized_zero_values = []  # to store each curve's normalized value for raw x = 0
    df_task = df[df["task"] == task]
    for model_name in df_task["Model"].unique():
        model_subdf = df_task[df_task["Model"] == model_name]
        curves_list = []
        for (k_val, n_val), group_kN in model_subdf.groupby(["k", "N"]):
            if group_kN.empty:
                continue

            bucket_avg, _ = calculate_buckets(
                group_kN, 
                n_buckets=kwargs["n_buckets"], 
                bucket_by="No gen toks", 
                bucket_name="Toks Bucket",
                y_axis="Correct?", 
                groupby_key="Model", 
            )
            if bucket_avg is None or len(bucket_avg) == 0:
                continue

            # Find the peak and leftmost bucket centers
            idx_peak = bucket_avg["Correct?"].idxmax()
            peak_x = bucket_avg.loc[idx_peak, "Toks Bucket Center"]
            bucket_min_x = bucket_avg["Toks Bucket Center"].min()
            if peak_x == bucket_min_x:
                # Avoid zero division by nudging bucket_min_x
                bucket_min_x -= 1

            # For this group, compute the normalized value for raw x = 0.
            # norm_zero = (0 - peak_x) / (peak_x - bucket_min_x)
            # normalized_zero_values.append(norm_zero)

            # Scaling: leftmost bucket center -> -1, peak -> 0.
            # Then clamp any values above clamp_upper.
            def scale_point(point):
                scaled_val = (point - peak_x) / (peak_x - bucket_min_x)
                return min(scaled_val, clamp_upper)

            bucket_sorted = bucket_avg.sort_values("Toks Bucket Center")
            curve_x = [scale_point(row["Toks Bucket Center"]) for _, row in bucket_sorted.iterrows()]
            curve_y = [row["Correct?"] for _, row in bucket_sorted.iterrows()]
            curves_list.append((np.array(curve_x), np.array(curve_y)))
        curves_by_model[model_name] = curves_list

    # Set up color maps. (Make sure models_in_order and rl_models_in_order
    # are defined in your environment or passed via kwargs.)
    cmap_blues = matplotlib.colormaps.get_cmap("Blues")
    model_colors = {model: cmap_blues(i / (len(models_in_order) - 1)) 
                    for i, model in enumerate(models_in_order)}
    cmap_oranges = matplotlib.colormaps.get_cmap("Oranges")
    rl_model_colors = {model: cmap_oranges(i / (len(rl_models_in_order) - 1)) 
                       for i, model in enumerate(rl_models_in_order)}

    # Optionally plot raw curves (for debugging)
    for model_name, curve_list in curves_by_model.items():
        if include_raw:
            for x_vals, y_vals in curve_list:
                ax.plot(x_vals, y_vals, '-o', alpha=0.15)

    # Interpolate & aggregate curves for each model
    for model_name, curve_list in curves_by_model.items():
        if not curve_list:
            continue

        # Get the domain for interpolation from all curves in this model
        try:
            all_x = np.concatenate([c[0] for c in curve_list])
        except Exception:
            continue
        global_min_x = all_x.min()
        global_max_x = all_x.max()
        if global_min_x == global_max_x:
            global_max_x = global_min_x + 1

        num_points = 100
        common_grid = np.linspace(global_min_x, global_max_x, num_points)

        interpolated_curves = []
        for (x_vals, y_vals) in curve_list:
            if len(x_vals) < 2:
                continue
            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            y_sorted = y_vals[sort_idx]
            interp_y = np.interp(common_grid, x_sorted, y_sorted)
            interpolated_curves.append(interp_y)
        if not interpolated_curves:
            continue

        interpolated_curves = np.array(interpolated_curves)
        mean_curve = interpolated_curves.mean(axis=0)

        if model_name in model_colors:
            ax.plot(common_grid, mean_curve, color=model_colors[model_name], marker=",")
        elif model_name in rl_model_colors:
            ax.plot(common_grid, mean_curve, color=rl_model_colors[model_name], marker=",")

    # Set axis limits and grid
    ax.set_xlim(left=-1.2, right=1.2)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Customize x-axis ticks:
    # Replace the tick at 0 (the normalized peak) with label 'L*'
    # and add one other tick at the average normalized value for raw x=0 labeled '0'
    # if normalized_zero_values:
    #     avg_norm_zero = np.mean(normalized_zero_values)
    #     # Order the ticks from left to right.
    #     ticks = [avg_norm_zero, 0]
    #     labels = ["0", "L*"]
    #     ax.set_xticks(ticks)
    #     ax.set_xticklabels(labels)
    ticks = [-1.2, 0]
    labels = ["0", "L*"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    
    ax.set_title(task)

def fig1_all_tasks(tasks, df, kwargs, include_raw, clamp_upper, n_cols):
    """
    Create one figure with a grid of subplots, one for each task in tasks.
    """
    n_tasks = len(tasks)
    n_rows = math.ceil(n_tasks / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    
    # Flatten axes (if only one row/column, axes may not be an array)
    if n_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, task in enumerate(tasks):
        ax = axes[i]
        plot_fig1_on_ax(ax, task, df, kwargs, include_raw=include_raw, clamp_upper=clamp_upper)
    
    # Remove any extra subplots if the grid has more slots than tasks.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    out_filename = os.path.join(kwargs["foldername"], "fig1_all_tasks.png")
    plt.savefig(out_filename, bbox_inches="tight")
    plt.clf()

def fig2(select_tasks, df, kwargs):
    # Create a single figure with 2 subplots, side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 6))
    
    # We store correlations in a dict so we can print them after plotting
    all_correlations = {}
    
    # Iterate over factors and corresponding subplot indices
    for i, factor in enumerate(["k", "N"]):
        ax = axes[i]  # the subplot for this factor
        #------------------------------------------------
        # Step 1) Collect raw data: model -> { task -> [(fv, L*), ...] }
        #------------------------------------------------
        raw_data = {}
        # Group by (Model, task, factor_value)
        for (model, task, fv), group in df.groupby(["Model", "task", factor]):
            if task not in select_tasks:
                continue

            bucket_avg, _ = calculate_buckets(
                group,
                n_buckets=kwargs["n_buckets"],
                bucket_by="No gen toks",
                bucket_name="Toks Bucket",
                y_axis="Correct?",
                groupby_key="Model"
            )
            if bucket_avg is None or bucket_avg.empty:
                continue
            # Take the Toks Bucket Center at the peak "Correct?" as L*
            idx_peak = bucket_avg["Correct?"].idxmax()
            Lstar = bucket_avg.loc[idx_peak, "Toks Bucket Center"]
            raw_data.setdefault(model, {}).setdefault(task, []).append((fv, Lstar))

        #------------------------------------------------
        # Step 2) Normalize within each (model, task) & combine across tasks
        #------------------------------------------------
        normalized_data = {}
        for model, task_to_fv_Lstar_list in raw_data.items():
            all_points = []
            for task, fv_Lstar_list in task_to_fv_Lstar_list.items():
                if not fv_Lstar_list:
                    continue
                fvs = [fv for (fv, _) in fv_Lstar_list]
                Lstars = [L for (_, L) in fv_Lstar_list]

                # Normalize factor values to [0,1]
                task_min_fv = min(fvs)
                task_max_fv = max(fvs)
                task_range_fv = max(task_max_fv - task_min_fv, 1e-12)

                # Normalize L* to [0,1]
                Lstar_min = min(Lstars)
                Lstar_max = max(Lstars)
                Lstar_range = max(Lstar_max - Lstar_min, 1e-12)

                for (fv, L) in fv_Lstar_list:
                    norm_fv = (fv - task_min_fv) / task_range_fv
                    norm_L = (L - Lstar_min) / Lstar_range
                    all_points.append((norm_fv, norm_L))

            normalized_data[model] = all_points

        #------------------------------------------------
        # Step 3) For each model, bin (smooth) + compute CIs + plot
        #------------------------------------------------
        corrs = {}
        for model, norm_points in normalized_data.items():
            if len(norm_points) < 2:
                continue

            # Separate into x, y
            norm_points = sorted(norm_points, key=lambda p: p[0])
            norm_x = [p[0] for p in norm_points]
            norm_y = [p[1] for p in norm_points]

            # Compute Pearson correlation on unbinned data
            if len(norm_x) > 1:
                corr, _ = stats.pearsonr(norm_x, norm_y)
            else:
                corr = float("nan")
            corrs[model] = corr
            label_text = f"{model_nicknames[model]} (r={corr:.2f})"

            # Binning / smoothing
            n_bins = 8  # tweak as desired
            min_x_val = norm_x[0]
            max_x_val = norm_x[-1]
            if max_x_val == min_x_val:
                # All x are the same, skip
                continue
            bin_edges = np.linspace(min_x_val, max_x_val, n_bins + 1)

            binned_x = []
            binned_y = []
            ci_lower = []
            ci_upper = []

            for j in range(n_bins):
                left = bin_edges[j]
                right = bin_edges[j+1]
                # Collect all y-values whose x is in [left, right)
                bin_vals = [norm_y[k] for k in range(len(norm_x))
                            if norm_x[k] >= left and (norm_x[k] < right or (j == n_bins - 1 and norm_x[k] <= right))]
                bin_xs = [norm_x[k] for k in range(len(norm_x))
                          if norm_x[k] >= left and (norm_x[k] < right or (j == n_bins - 1 and norm_x[k] <= right))]
                if not bin_vals:
                    continue

                mean_x = np.mean(bin_xs)
                mean_y = np.mean(bin_vals)
                n = len(bin_vals)
                if n > 1:
                    se = stats.sem(bin_vals)
                    margin = se * stats.t.ppf(0.975, n - 1)
                else:
                    margin = 0.0

                lower = mean_y - margin
                upper = mean_y + margin

                binned_x.append(mean_x)
                binned_y.append(mean_y)
                ci_lower.append(lower)
                ci_upper.append(upper)

            # Plot the smoothed data + confidence intervals
            if len(binned_x) == 0:
                continue

            base_color = model_colors.get(model, "blue")
            rgba_color = mcolors.to_rgba(base_color, alpha=0.9)

            # Scatter the binned points
            # ax.scatter(binned_x, binned_y, color=rgba_color, label=label_text)
            # Connect them with a line (optional)
            ax.plot(binned_x, binned_y, color=rgba_color, alpha=0.7)
            # Confidence intervals with fill_between
            # if len(binned_x) > 1:
            #     ax.fill_between(binned_x, ci_lower, ci_upper, color=rgba_color, alpha=0.1)

        # Store the correlations so we can print them after
        all_correlations[factor] = corrs

        # Final formatting for this subplot
        ax.set_aspect(1)
        ax.set_ylim(-0.25, 1.25)
        # ax.set_ylim(-1.6, 2.6)
        ax.set_xlim(-0.15, 1.15)
        ax.set_xlabel(f"Normalized {factor}")
        if factor == "k": ax.set_ylabel("Normalized L*")
        # ax.set_title(f"Factor = {factor}")
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Tight layout for the entire figure
    fig.tight_layout()

    # Save once, containing both subplots
    out_filename = os.path.join(kwargs["foldername"], "fig2_by_model.png")
    fig.savefig(out_filename, bbox_inches="tight")
    plt.close(fig)
    # Print correlations for each factor
    for model in df["Model"].unique():
        corrs = all_correlations.get(factor, {})
        if model in all_correlations['k'] and model in all_correlations['N']:
            print(f"{model} k:{all_correlations['k'][model]:.3f} N:{all_correlations['N'][model]:.3f}")
        elif model in all_correlations['k']:
            print(f"{model} k:{all_correlations['k'][model]:.3f} N:None")
        elif model in all_correlations['N']:
            print(f"{model} k:None N:{all_correlations['N'][model]:.3f}")
        else: 
            print(f"No data for {model}")

# python src/plot_all_tasks.py --models ${ALL_MODELS_PLOTTING}
if __name__ == "__main__":
    args = get_args()

    if args.delete_old and os.path.exists(args.foldername):
        shutil.rmtree(args.foldername)
    os.makedirs(args.foldername, exist_ok=True)

    to_highlight = [
        ("even_odd", "DeepSeek-R1-Distill-Qwen-32B"), 
        ("dyck", "o3-mini"), 
        ("bool", "gpt-4o"),
        ("arith", "DeepSeek-R1-Distill-Llama-70B"),
        ("shuffled_objects", "DeepSeek-R1"),
    ]

    plot_kwargs = {
        "n_buckets": args.n_buckets,
        "foldername": args.foldername,
        "to_highlight": to_highlight
    }

    df = load_data(
        args,
        plot_kwargs,
    )

    fig1_all_tasks(args.tasks, df, plot_kwargs, include_raw=False, clamp_upper=2, n_cols=3)
    fig2(args.tasks, df, plot_kwargs)

    # plot_correctness_by_ttoks(df, plot_kwargs)
    # model_pairs = [
    #     ("Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Llama-8B", ["even_odd"]),
    #     ("Llama-3.3-70B-Instruct-Turbo", "DeepSeek-R1-Distill-Llama-70B", ["arith"]),
    #     ("Qwen2.5-7B-Instruct", "DeepSeek-R1-Distill-Qwen-7B", ["array_idx"]),
    #     ("Qwen2.5-32B-Instruct", "DeepSeek-R1-Distill-Qwen-32B", ["even_odd"]),
    #     ("gpt-4o", "o3-mini", ["bool"]),
    #     ("DeepSeek-R1", "DeepSeek-V3", ["shuffled_objects"])
    # ]
    # for model_pair in model_pairs:
    #     plot_correctness_by_ttoks_model_pairs(df, model_pair, plot_kwargs, normalize=False)
    
    # for task in args.tasks:
    #     df_task = df[df["task"] == task]
    #     fig1_per_task(task, df_task, plot_kwargs)
