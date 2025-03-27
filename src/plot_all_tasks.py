import json
import re
import os
import shutil
import random
import pandas as pd
import glob
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import colors as mcolors
import matplotlib.lines as mlines

from plot_utils import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_buckets", type=int, default=4)
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--f2_models", nargs='+')
    parser.add_argument("--delete_old", action="store_true")
    parser.add_argument("--all_tasks", nargs='+', default=['array_idx', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies', 'dyck', 'cruxeval', 'logical_deduction'])
    parser.add_argument("--f1_tasks", nargs='+', default=['dyck', 'array_idx', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'cruxeval', 'logical_deduction'])
    parser.add_argument("--f2_tasks", nargs='+', default=['dyck', 'array_idx', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies', 'logical_deduction'])

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

def load_data(args, kwargs, filter_stddev_count=1, include_all=False):
    all_df = None
    for task in args.all_tasks:
        compute_random, foldername_parser, dfa_factors_order, output_folder = get_task_info(task)
        if include_all: compute_random = lambda x: -100.
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

def plot_fig1_on_ax(ax, task, df, kwargs, include_raw, clamp_upper, y_normalization):
    """
    Plot the fig1 curve for a single task onto a given axis (ax),
    with customized x-axis ticks: the tick at 0 (the normalized peak) is labeled 'L*',
    and one additional tick, at the average normalized value corresponding to raw 0 across all curves, is labeled '0'.
    """
    model_to_task_to_Lstar = {}
    # Build normalized curves per (Model, k, N)
    curves_by_model = {}
    # normalized_zero_values = []  # to store each curve's normalized value for raw x = 0
    df_task = df[df["task"] == task]
    for model_name in df_task["Model"].unique():
        model_to_task_to_Lstar[model_name] = {}
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
            model_to_task_to_Lstar[model_name][(k_val, n_val)] = peak_x
            bucket_min_x = bucket_avg["Toks Bucket Center"].min()
            if peak_x == bucket_min_x:
                # Avoid zero division by nudging bucket_min_x
                bucket_min_x -= 1

            # Scaling: leftmost bucket center -> -1, peak -> 0.
            # Then clamp any values above clamp_upper.
            def scale_point(point):
                scaled_val = (point - peak_x) / (peak_x - bucket_min_x)
                if scaled_val <= clamp_upper: return scaled_val
                return None

            bucket_sorted = bucket_avg.sort_values("Toks Bucket Center")
            curve_x = [scale_point(row["Toks Bucket Center"]) for _, row in bucket_sorted.iterrows()]
            curve_x = [point for point in curve_x if point is not None]
            curve_y = [row["Correct?"] for _, row in bucket_sorted.iterrows()][:len(curve_x)]
            curves_list.append((np.array(curve_x), np.array(curve_y)))
        curves_by_model[model_name] = curves_list


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
        mean_curve = y_normalization(mean_curve)

        if model_name in nonrl_model_colors:
            ax.plot(common_grid, mean_curve, color=nonrl_model_colors[model_name], marker=",")
        elif model_name in rl_model_colors:
            ax.plot(common_grid, mean_curve, color=rl_model_colors[model_name], marker=",")

    # Set axis limits and grid
    ax.set_xlim(left=-1.2, right=1.2)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Customize x-axis ticks:
    ticks = [-1.2, 0]
    labels = ["0", "L*"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    
    ax.set_title(task_full_names[task])

    return model_to_task_to_Lstar

def fig1_all_tasks(tasks, df, kwargs, include_raw, clamp_upper, n_cols, y_normalization=lambda x: x, suffix=''):
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
    
    all_model_to_task_to_Lstar = {}
    for i, task in enumerate(tasks):
        ax = axes[i]
        model_to_task_to_Lstar = plot_fig1_on_ax(ax, task, df, kwargs, include_raw=include_raw, clamp_upper=clamp_upper, y_normalization=y_normalization)
        for model, task_to_Lstar in model_to_task_to_Lstar.items():
            if model not in all_model_to_task_to_Lstar: all_model_to_task_to_Lstar[model] = {}
            all_model_to_task_to_Lstar[model][task] = task_to_Lstar
    # Remove any extra subplots if the grid has more slots than tasks.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    out_filename = os.path.join(kwargs["foldername"], f"fig1{suffix}.png")
    plt.savefig(out_filename, bbox_inches="tight")
    plt.clf()

    # make latex table from task_to_model_to_taskwise_Lstars: rows are models, columns are tasks.
    # cells are min and max L*s for the diff k, N configs.. displayed as [low, ... high]
    # Now, produce the LaTeX code for the final summary table:
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \begin{adjustbox}{max width=\textwidth}")
    latex_lines.append(r"    \begin{tabular}{l " + ''.join([r">{\centering\arraybackslash}p{1.7cm}" for _ in range(5)]) + "}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    \textbf{Model} & " + " & ".join([f"\\textbf{{{task_full_names[name]}}}" for name in tasks[:5]]) + r"\\")
    latex_lines[-1].replace("Nested Boolean Expression", "Bool Expr")
    latex_lines.append(r"    \midrule")
    
    for model, task_Lstars in all_model_to_task_to_Lstar.items():
        latex_line = f"    {model_nicknames[model]} "
        for task in tasks[:5]:
            if task not in task_Lstars: 
                latex_line += " &  -- "
                continue
            low = min(task_Lstars[task].values())
            high = max(task_Lstars[task].values())
            latex_line += f" & $[{int(low)},..{int(high)}]$"
        latex_line += r"   \\"
        latex_lines.append(latex_line)
    latex_lines.append(r"    \midrule")
    latex_lines.append(r"    \textbf{Model} & " + " & ".join([f"\\textbf{{{task_full_names[name]}}}" for name in tasks[5:]]) + r"\\")
    latex_lines.append(r"    \midrule")
    
    for model, task_Lstars in all_model_to_task_to_Lstar.items():
        latex_line = f"    {model_nicknames[model]} "
        for task in tasks[5:]:
            if task not in task_Lstars: 
                latex_line += " &  -- "
                continue
            low = min(task_Lstars[task].values())
            high = max(task_Lstars[task].values())
            latex_line += f" & $[{int(low)},..{int(high)}]$"
        latex_line += r"   \\"
        latex_lines.append(latex_line)

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    \end{adjustbox}")
    latex_lines.append(r"    \caption{$L^*$ varies by task and model.}")
    latex_lines.append(r"    \label{tab:f1}")
    latex_lines.append(r"\end{table}")

    final_latex = "\n".join(latex_lines)
    print("L* table (LaTeX):")
    print(final_latex)

def fig2(select_tasks, select_models, df, kwargs, fig_suffix, plot_confidence=False):

    # Create a single figure with 2 subplots, side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 12))
    
    # We store correlations in a dict so we can print them after plotting
    all_correlations = {}
    
    # Iterate over factors and corresponding subplot indices
    for i, factor in enumerate(["N", "k"]):
        ax = axes[i]  # the subplot for this factor
        #------------------------------------------------
        # Step 1) Collect raw data: model -> { task -> [(fv, L*), ...] }
        #------------------------------------------------
        raw_data = {}
        # Group by (Model, task, factor_value)
        for (model, task, fv), group in df.groupby(["Model", "task", factor]):
            if task not in select_tasks:
                continue
            if model not in select_models: continue

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
                continue
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

            if model in nonrl_model_colors:
                base_color = nonrl_model_colors.get(model, "blue")
            if model in rl_model_colors:
                base_color = rl_model_colors.get(model, "blue")
            rgba_color = mcolors.to_rgba(base_color, alpha=0.9)

            ax.plot(binned_x, binned_y, color=rgba_color, alpha=0.7)
            # Confidence intervals with fill_between. 
            if plot_confidence and len(binned_x) > 1:
                ax.fill_between(binned_x, ci_lower, ci_upper, color=rgba_color, alpha=0.1)

        # Store the correlations so we can print them after
        all_correlations[factor] = corrs

        # Final formatting for this subplot
        ax.set_aspect(1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 1.15)
        if factor == "k":
            ax.set_xlabel("Number of States")
            ax.set_ylabel("")
            ax.set_yticklabels([])
            # ax.set_xticklabels([])
        elif factor == "N":
            ax.set_xlabel("Run length")
            ax.set_ylabel("L* (critical length)")
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])

    # Tight layout for the entire figure
    fig.tight_layout()

    # Save once, containing both subplots
    out_filename = os.path.join(kwargs["foldername"], f"fig2{fig_suffix}.png")
    fig.savefig(out_filename, bbox_inches="tight")
    plt.close(fig)

    def get_order(m):
        # Return the index if present in ordered_model_list, else a large number
        return all_models_size_ordered.index(m) if m in all_models_size_ordered else 999999

    # Generate LaTeX table string
    latex_lines = [
        r"\begin{tabular}{lcc}",
        r"  \toprule",
        r"  Model & Corr$(L^*,N)$ & Corr$(L^*,k)$ \\",
        r"  \midrule"
    ]

    # Add rows
    models_sorted = sorted(set(all_correlations['k'].keys()) | set(all_correlations['N'].keys()), key=get_order)
    avg_corr_N = []
    avg_corr_k = []
    for model in models_sorted:
        corr_N = all_correlations['N'].get(model, None)
        corr_k = all_correlations['k'].get(model, None)
        if corr_N is not None: avg_corr_N.append(corr_N)
        if corr_k is not None: avg_corr_k.append(corr_k)
        corr_N_str = f"${corr_N:.2f}$" if corr_N is not None else "N/A"
        corr_k_str = f"${corr_k:.2f}$" if corr_k is not None else "N/A"
        latex_lines.append(f"  {model_nicknames[model]} & {corr_N_str} & {corr_k_str} \\\\")

    avg_corr_k = sum(avg_corr_k) / len(avg_corr_k)
    avg_corr_N = sum(avg_corr_N) / len(avg_corr_N)
    latex_lines.append(r"  \midrule")
    latex_lines.append(f"  Average & ${avg_corr_N:.2f}$ & ${avg_corr_k:.2f}$ \\\\")
    # Finish table
    latex_lines.append(r"  \bottomrule")
    latex_lines.append(r"\end{tabular}")

    # Print the LaTeX table
    latex_table = "\n".join(latex_lines)
    print(fig_suffix)
    print(latex_table)

def plot_fig2_on_ax(axes, task, df, kwargs, less_info=False):
    """
    For a given task, plot two smoothed curves (with error shading)
    for the correlations of L* with factor "k" and factor "N"
    on the two axes provided in 'axes' (axes[0] for "k", axes[1] for "N").
    """
    corrs = {}
    for ax_idx, factor in enumerate(["N", "k"]):
        corrs[factor] = {}
        # --- Step 1: Collect raw (fv, L*) data for this task and factor ---
        raw_data = {}
        for (model, t, fv), group in df.groupby(["Model", "task", factor]):
            if t != task:
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
            idx_peak = bucket_avg["Correct?"].idxmax()
            Lstar = bucket_avg.loc[idx_peak, "Toks Bucket Center"]
            raw_data.setdefault(model, []).append((fv, Lstar))
        
        # --- Step 2: Normalize data per model ---
        normalized_data = {}
        for model, fv_Lstar_list in raw_data.items():
            if not fv_Lstar_list:
                continue
            fvs = [fv for (fv, _) in fv_Lstar_list]
            Lstars = [L for (_, L) in fv_Lstar_list]
            task_min_fv = min(fvs)
            task_max_fv = max(fvs)
            task_range_fv = max(task_max_fv - task_min_fv, 1e-12)
            Lstar_min = min(Lstars)
            Lstar_max = max(Lstars)
            Lstar_range = max(Lstar_max - Lstar_min, 1e-12)
            norm_points = []
            for (fv, L) in fv_Lstar_list:
                norm_fv = (fv - task_min_fv) / task_range_fv
                norm_L = (L - Lstar_min) / Lstar_range
                norm_points.append((norm_fv, norm_L))
            normalized_data[model] = norm_points

        # --- Step 3: For each model, bin (smooth) + compute CI and correlation ---
        for model, norm_points in normalized_data.items():
            if len(norm_points) < 2:
                continue
            norm_points = sorted(norm_points, key=lambda p: p[0])
            norm_x = [p[0] for p in norm_points]
            norm_y = [p[1] for p in norm_points]
            if len(norm_x) > 1:
                corr, _ = stats.pearsonr(norm_x, norm_y)
                corrs[factor][model] = corr
            else:
                corr = float("nan")
            n_bins = 8
            min_x_val = norm_x[0]
            max_x_val = norm_x[-1]
            if max_x_val == min_x_val:
                continue
            bin_edges = np.linspace(min_x_val, max_x_val, n_bins + 1)
            binned_x, binned_y = [], []
            ci_lower, ci_upper = [], []
            for j in range(n_bins):
                left = bin_edges[j]
                right = bin_edges[j+1]
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
                binned_x.append(mean_x)
                binned_y.append(mean_y)
                ci_lower.append(mean_y - margin)
                ci_upper.append(mean_y + margin)
            if len(binned_x) == 0:
                continue

            # --- Step 4: Plot the smoothed curve and CI shading ---
            if model in nonrl_model_colors:
                base_color = nonrl_model_colors.get(model, "blue")
            elif model in rl_model_colors:
                base_color = rl_model_colors.get(model, "blue")
            else:
                base_color = "blue"
            rgba_color = mcolors.to_rgba(base_color, alpha=0.9)
            # Use a solid line for "k" and dashed for "N"
            axes[ax_idx].plot(binned_x, binned_y, color=rgba_color,
                    label=f"{model_nicknames[model]} (r={corr:.2f})")
            axes[ax_idx].fill_between(binned_x, ci_lower, ci_upper, color=rgba_color, alpha=0.2)
        
        # --- Step 5: Format the current axis ---
        axes[ax_idx].set_aspect(1)
        axes[ax_idx].set_xlim(-0.05, 1.05)
        axes[ax_idx].set_ylim(-0.15, 1.15)
        if factor == "k":
            axes[ax_idx].set_xlabel("Number of States")
            axes[ax_idx].set_ylabel("")
            axes[ax_idx].set_yticklabels([])
            # axes[ax_idx].set_xticklabels([])
        elif factor == "N":
            axes[ax_idx].set_xlabel("Run length")
            if not less_info:
                axes[ax_idx].set_ylabel("L* (critical length)")
            # axes[ax_idx].set_yticklabels([])
            # axes[ax_idx].set_xticklabels([])
        axes[ax_idx].grid(True, linestyle="--", alpha=0.6)

    return corrs

def fig2_per_task(tasks, df, kwargs, n_cols=2):
    """
    Create one figure with a grid of subplots—each task occupies two adjacent axes 
    (one for k and one for N). A common title is placed above the pair.
    """

    n_tasks = len(tasks)
    n_rows = math.ceil(n_tasks / n_cols)
    
    # Increase hspace to give more vertical room between rows
    fig, axes = plt.subplots(
        n_rows, n_cols * 2,
        figsize=(n_cols * 6, n_rows * 4),
        gridspec_kw={'wspace': 0.5, 'hspace': 0.6}
    )

    if n_rows == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()

    task_corrs = {}
    # Plot each task on two adjacent axes
    for i, task in enumerate(tasks):
        task_axes = axes[i*2 : i*2+2]
        task_corrs[task] = plot_fig2_on_ax(task_axes, task, df, kwargs, less_info=i % 2 == 1)

    # Remove extra subplots if grid > tasks
    total_axes = len(axes)
    used_axes = n_tasks * 2
    for j in range(used_axes, total_axes):
        fig.delaxes(axes[j])

    # Step 1: do a preliminary tight_layout to set positions
    plt.tight_layout(pad=2)
    # Step 2: force a draw so positions are finalized
    fig.canvas.draw()

    # Now place the task titles above each pair
    for i, task in enumerate(tasks):
        task_axes = axes[i*2 : i*2+2]

        pos0 = task_axes[0].get_position()
        pos1 = task_axes[1].get_position()
        # Horizontal center = midpoint of left axis's left edge & right axis's right edge
        x_center = (pos0.x0 + pos1.x1) / 2
        # Vertical top = whichever top is higher
        y_top = max(pos0.y1, pos1.y1)
        # Place text above that
        fig.text(
            x_center, 
            y_top + 0.01,  # increase offset if needed
            task_full_names.get(task, task),
            ha='center', va='bottom', fontsize=20
        )

    out_filename = os.path.join(kwargs["foldername"], "fig2_per_task.png")
    plt.savefig(out_filename, bbox_inches="tight")
    plt.clf()


    
    def get_order(m):
        # Return the index if present in ordered_model_list, else a large number
        return all_models_size_ordered.index(m) if m in all_models_size_ordered else 999999

    # Generate LaTeX table string
    latex_lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{l >{\centering\arraybackslash}p{1cm}>{\centering\arraybackslash}p{1cm}>{\centering\arraybackslash}p{1cm}>{\centering\arraybackslash}p{1cm}>{\centering\arraybackslash}p{1cm}>{\centering\arraybackslash}p{1cm}}",
        r"  \toprule",
        r"  Model & $\rho(L^*,N)$ & $\rho(L^*,k)$ & $\rho(L^*,N)$ & $\rho(L^*,k)$ & $\rho(L^*,N)$ & $\rho(L^*,k)$ \\",
    ]

    # Add rows
    models_sorted = sorted(df["Model"].unique(), key=get_order)
    task_rows = [tasks[i*3:i*3+3] for i in range(1 + len(tasks)//3)]
    for task_row in task_rows:
        latex_lines.append(r"    \midrule")
        latex_lines.append(r"   & \multicolumn{2}{c}{\textbf{" + r"}} & \multicolumn{2}{c}{\textbf{".join(task_row) + r"}} \\")
        for modelname in models_sorted:
            model_tasks_line = [model_nicknames[modelname]]
            for taskname in task_row:
                task_info = task_corrs[taskname]
                corr_N = task_info['N'].get(modelname, None)
                corr_k = task_info['k'].get(modelname, None)
                corr_N_str = f"${corr_N:.2f}$" if corr_N is not None else "--"
                corr_k_str = f"${corr_k:.2f}$" if corr_k is not None else "--"
                model_tasks_line.extend([corr_N_str, corr_k_str])
            model_tasks_line = " & ".join(model_tasks_line) + r"\\"
            latex_lines.append(model_tasks_line)

    # Finish table
    latex_lines.append(r"  \bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"    \end{adjustbox}")
    latex_lines.append(r"    \caption{Per-task improvement by constraining to $L^*$.}")
    latex_lines.append(r"    \label{tab:tasks_performance}")
    latex_lines.append(r"\end{table}")
    # Print the LaTeX table
    latex_table = "\n".join(latex_lines)
    print("fig2 per task")
    print(latex_table)

def fig3(tasks, df, kwargs, fig_suffix):
    if not os.path.exists("extrapolated.json"):
        return

    # Load your aggregated results: 
    # { "ModelName": { "taskA": ([old_accs], [new_accs], [deltas]), ... } }
    modelname_to_task_to_row = json.load(open("extrapolated.json"))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Create a dictionary for quick index lookup
    model_to_jitter_index = {m: i for i, m in enumerate(all_models_size_ordered)}

    # 2) Helper to compute a stable log‐scale jitter
    def jitter_log_x(size, model, base_offset_multiplier=0.02):
        """
        Shifts 'size' in log10 space based on:
         - The model's index in the ordered_model_list
        """
        # If model not in the dictionary, place it at the end
        idx = model_to_jitter_index.get(model, len(all_models_size_ordered))
        total_count = len(all_models_size_ordered)

        # Base offset in log space: spread models from negative to positive
        # around 0. Larger magnitude for first/last in the list.
        base_offset = base_offset_multiplier * (idx - total_count/2)

        # Convert to log space
        log_size = np.log10(size)
        jittered_log_size = log_size + base_offset
        return 10 ** jittered_log_size

    # 3) Helper to draw a vertical "error region" around the new accuracy
    def fill_vertical_error(ax, x_center, y_center, y_err, color, alpha=0.3, fraction=0.02):
        """
        Draws a vertical rectangle centered at (x_center, y_center) 
        spanning y_center ± y_err, with a small horizontal width in log space.
        
        fraction controls the half‐width in log space (e.g., 0.02 => ±0.01 in log10).
        """
        # Half‐width in log space
        log_x = np.log10(x_center)
        half_log_width = fraction / 2.0

        x_left = 10 ** (log_x - half_log_width)
        x_right = 10 ** (log_x + half_log_width)

        y_low = y_center - y_err
        y_high = y_center + y_err

        ax.fill_between(
            [x_left, x_right],
            y_low,   # lower y-bound
            y_high,  # upper y-bound
            color=color,
            alpha=alpha,
            linewidth=0
        )

    # We’ll keep track of which models we actually plot, to build a custom legend.
    plotted_models = set()

    # 4) Aggregate across tasks for each model
    for model, task_metrics in modelname_to_task_to_row.items():
        # Combine old_accs, new_accs, and deltas from all tasks
        all_old_accs = []
        all_new_accs = []
        all_deltas = []
        for (old_list, new_list, delta_list, _, _) in task_metrics.values():
            all_old_accs.extend(old_list)
            all_new_accs.extend(new_list)
            all_deltas.extend(delta_list)

        if not all_old_accs or not all_new_accs:
            continue

        # Means (×100 => percentages) 
        old_acc = np.mean(all_old_accs) * 100
        new_acc = np.mean(all_new_accs) * 100

        # Delta's SE (has already been *100)
        delta_se = np.std(all_deltas, ddof=1) / np.sqrt(len(all_deltas)) if len(all_deltas) > 1 else 0.0

        # Determine color
        if model in nonrl_model_colors:
            color = nonrl_model_colors[model]
        elif model in rl_model_colors:
            color = rl_model_colors[model]
        else:
            color = "black"

        # Extract approximate size
        size_match = re.search(r'(\d+)[bB]', model)
        if size_match is None:
            # fallback
            size = 685 if "DeepSeek" in model else 200
        else:
            size = int(size_match.group(1))

        # Apply log-scale jitter
        x_val = jitter_log_x(size, model)

        # Plot old (circle) and new (square)
        ax.scatter(x_val, old_acc, color=color, marker="o", s=20)
        ax.scatter(x_val, new_acc, color=color, marker="o", s=20)

        # Arrow from old to new
        ax.annotate(
            "",
            xy=(x_val, new_acc),
            xytext=(x_val, old_acc),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
        )

        # Shaded region for new_acc ± delta_se
        # fill_vertical_error(ax, x_val, new_acc, delta_se, color=color, alpha=0.3, fraction=0.05)

        plotted_models.add(model)

    # --- 5) Format axes ---
    ax.set_xscale("log")
    ax.set_xlabel("Model size")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.6)

    # If you want to remove the x-axis tick labels but keep the ticks:
    #   ax.set_xlabel("")
    #   ax.set_xticklabels([])

    # --- 6) Build a 2-column legend in the same order as your color dicts ---
    nonrl_first_order = [
        "Ministral-8B-Instruct-2410",
        "Qwen2.5-7B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Llama-3.3-70B-Instruct-Turbo",
        "gpt-4o",
        "DeepSeek-V3",
        "Meta-Llama-3.1-405B-Instruct-Turbo",
        "gemma-2-9b-it",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-32B",
        "DeepSeek-R1-Distill-Llama-70B",
        "o3-mini",
        "DeepSeek-R1",
    ]
    def get_legend_order(m):
        # Return the index if present in ordered_model_list, else a large number
        return nonrl_first_order.index(m) if m in nonrl_first_order else 999999

    legend_handles = []
    for m in sorted(plotted_models, key=get_legend_order):
        if m in nonrl_model_colors:
            c = nonrl_model_colors[m]
        elif m in rl_model_colors:
            c = rl_model_colors[m]
        else:
            c = "black"
        lbl = model_nicknames.get(m, m)
        h = mlines.Line2D([], [], color=c, marker='s', linestyle='None', label=lbl)
        legend_handles.append(h)

    ax.legend(handles=legend_handles, loc="best", fontsize="small", frameon=True, ncol=2)

    fig.tight_layout()
    out_filename = os.path.join(kwargs["foldername"], f"fig3{fig_suffix}.png")
    fig.savefig(out_filename, bbox_inches="tight")
    plt.close(fig)

def generation_lengths(df, kwargs):
    # Separate data by model size
    model_data = {
        model_name: df[df["Model"] == model_name].sort_values(by="No gen toks")
        for model_name in all_models_size_ordered
    }

    # Prepare the figure
    plt.figure(figsize=(12, 6))

    tick_positions = []
    labels = []
    plotted_something = False

    # Iterate over models and optionally by_factor
    for i, (model, model_df) in enumerate(model_data.items(), start=1):

        # get average L^* across tasks and task configurations.
        Lstars = []
        for task in model_df["task"].unique():
            df_model_task = model_df[model_df["task"] == task]
            for (k_val, n_val), group_kN in df_model_task.groupby(["k", "N"]):
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
                Lstars.append(peak_x)
        avg_Lstar = sum(Lstars) / len(Lstars)

        # Extract data for the model
        data = model_df["No gen toks"].tolist()
        if not data:
            continue

        # Define position for the boxplot
        position = [i]
        tick_positions.append(i)
        labels.append(model_nicknames[model])
        if model in nonrl_model_colors:
            plot_color = nonrl_model_colors[model]
        elif model in rl_model_colors:
            plot_color = rl_model_colors[model]

        # Plot the boxplot
        plt.boxplot(
            [data],
            positions=position,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=plot_color, color=plot_color),
            medianprops=dict(color="black"),
            showfliers=False,
        )

        # Place average Lstar
        plt.scatter(
            i, 
            avg_Lstar, 
            color='white', 
            edgecolors="black", 
            marker="D", 
            s=100, 
            zorder=3
        )

    # Set x-axis labels
    plt.xticks(ticks=tick_positions, labels=labels, rotation=45, ha="right")
    plt.ylabel("Generation Length")

    # Save the plot
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.savefig(
        os.path.join(kwargs['foldername'], "genlength_boxplot.png"),
        bbox_inches="tight"
    )
    plt.clf()

def scatter_len_to_acc(df, kwargs):
    # Prepare the figure
    plt.figure(figsize=(10, 6))

    # Loop over models in the prescribed order
    plotted_models = set()
    for model in all_models_size_ordered:
        model_df = df[df["Model"] == model]
        if model_df.empty:
            continue
        
        # Compute average generation length and average accuracy
        avg_gen_length = model_df["No gen toks"].mean()
        avg_acc = model_df["Correct?"].mean() * 100  # Convert fraction to percentage
        
        # Select color based on model type
        if model in nonrl_model_colors:
            color = nonrl_model_colors[model]
        elif model in rl_model_colors:
            color = rl_model_colors[model]
        else:
            color = "black"
        plotted_models.add(model)
        # Plot the scatter point
        plt.scatter(avg_gen_length, avg_acc, color=color, s=80, label=model_nicknames[model])
    # Set axis labels and title
    plt.xlabel("Average Generation Length")
    plt.ylabel("Average Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Create a legend with unique entries
    nonrl_first_order = [
        "Ministral-8B-Instruct-2410",
        "Qwen2.5-7B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Llama-3.3-70B-Instruct-Turbo",
        "gpt-4o",
        "DeepSeek-V3",
        "Meta-Llama-3.1-405B-Instruct-Turbo",
        "gemma-2-9b-it",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-32B",
        "DeepSeek-R1-Distill-Llama-70B",
        "o3-mini",
        "DeepSeek-R1",
    ]
    def get_legend_order(m):
        # Return the index if present in ordered_model_list, else a large number
        return nonrl_first_order.index(m) if m in nonrl_first_order else 999999

    legend_handles = []
    for m in sorted(plotted_models, key=get_legend_order):
        if m in nonrl_model_colors:
            c = nonrl_model_colors[m]
        elif m in rl_model_colors:
            c = rl_model_colors[m]
        else:
            c = "black"
        lbl = model_nicknames.get(m, m)
        h = mlines.Line2D([], [], color=c, marker='s', linestyle='None', label=lbl)
        legend_handles.append(h)

    plt.legend(handles=legend_handles, loc="best", fontsize="small", frameon=True, ncol=2)

    # Save the plot
    plt.tight_layout()  # Adjust layout to avoid overlap
    out_filename = os.path.join(kwargs['foldername'], "genlength_accuracy.png")
    plt.savefig(out_filename, bbox_inches="tight")
    plt.clf()

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

    # def relative_to_start(y_curve):
    #     original_val = y_curve[0]
    #     norm_y = [(y_val-original_val)/original_val for y_val in y_curve]
    #     return norm_y
    # def relative_to_max(y_curve):
    #     max_val = y_curve.max()
    #     norm_y = [1. - (max_val - y_val)/max_val for y_val in y_curve]
    #     return norm_y
    # def normalize(y_curve):
    #     max_val = y_curve.max()
    #     min_val = y_curve.min()
    #     norm_y = [(y_val - min_val)/(max_val - min_val) for y_val in y_curve]
    #     return norm_y
        
    # fig1_all_tasks(args.f1_tasks, df, plot_kwargs, include_raw=False, clamp_upper=2, n_cols=3, suffix='_9')
    # fig1_all_tasks(args.all_tasks, df, plot_kwargs, include_raw=False, clamp_upper=2, n_cols=3, suffix="_all")
    
    fig2(args.all_tasks, args.models, df, plot_kwargs, '_all')
    fig2_per_task(args.all_tasks, df, plot_kwargs)
    
    # fig3(args.all_tasks, df, plot_kwargs, "")

    # nonfiltered_df = load_data(
    #     args,
    #     plot_kwargs,
    #     filter_stddev_count=0,
    #     include_all=True,
    # )
    # generation_lengths(nonfiltered_df, plot_kwargs)
    # scatter_len_to_acc(nonfiltered_df, plot_kwargs)

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
