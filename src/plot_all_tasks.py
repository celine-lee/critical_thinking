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

from plot_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_buckets", type=int, default=4)
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--delete_old", action="store_true")
    parser.add_argument("--tasks", nargs='+', default=['dyck', 'array_idx', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies', 'cruxeval', 'logical_deduction'])

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
        if task == "cruxeval": 
            task_df = load_cruxeval_data(output_folder)
        else:
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

        plt.plot(task_model_df["No gen toks"], task_model_df["Correct?"], color=line_color, marker=",")
        plt.scatter(peak_x, peak_y_normalized, color=dot_color, zorder=dotsize, label=label)
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
        peak_x, peak_y_normalized = peak_row["Toks Bucket Center"], (peak_row["Correct?"] - min_y) / (max_y - min_y) if max_y != min_y else peak_row["Correct?"]

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
        plt.scatter(peak_x, peak_y_normalized, color=dot_color, zorder=dotsize, label=label, alpha=opacity)
    
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

def fig1_per_task(taskname, df, kwargs, include_raw=False, clamp_upper=3):
    """
    1. For each (Model, k, N):
       - Bucket the data by 'No gen toks'.
       - Find the bucket with peak correctness (peak bucket center).
       - Shift & scale x -> peak at 0, scaled so leftmost at -1, right side wherever accordingly
       - Store the resulting curve (x, y) in a list.

    2. For each Model:
       - Combine all (k, N) curves by interpolation onto a common x-grid.
       - Plot one aggregated curve.
       - Color in a gradient -- large models get darker colors.
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
            )
            if bucket_avg is None or len(bucket_avg) == 0:
                continue

            # 1b) Identify the bucket with the highest correctness and define the scale factor
            idx_peak = bucket_avg["Correct?"].idxmax()
            peak_x = bucket_avg.loc[idx_peak, "Toks Bucket Center"]
            bucket_min_x = bucket_avg["Toks Bucket Center"].min()
            # Shift & scale x so peak -> 0, domain -> leftmost at -1
            # Scale function to normalize x-values
            if peak_x == bucket_min_x:
                # Avoid zero division
                bucket_min_x -= 1
            # print(f"peak: {peak_x} while bucket min: {bucket_min_x}")
            def scale_point(point):
                scaled_val = (point - peak_x) / (peak_x - bucket_min_x)
                return min(scaled_val, clamp_upper)

            bucket_sorted = bucket_avg.sort_values("Toks Bucket Center")
            curve_x = [scale_point(row["Toks Bucket Center"]) for _, row in bucket_sorted.iterrows()]
            curve_y = [row["Correct?"] for _, row in bucket_sorted.iterrows()]
            curves_list.append((np.array(curve_x), np.array(curve_y)))

        # Store all (k, N) curves for this model
        curves_by_model[model_name] = curves_list

    # --- STEP 2: Interpolate & Aggregate per Model ---
    plt.figure(figsize=(10, 6))

    cmap = matplotlib.colormaps.get_cmap("Blues")  # Use a gradient colormap
    model_colors = {model: cmap(i / (len(models_in_order) - 1)) for i, model in enumerate(models_in_order)}
    cmap = matplotlib.colormaps.get_cmap("Oranges")  # Use a gradient colormap
    rl_model_colors = {model: cmap(i / (len(rl_models_in_order) - 1)) for i, model in enumerate(rl_models_in_order)}

    for model_name, curve_list in curves_by_model.items():
        if not curve_list:
            continue
        if include_raw:
            # debug -- after scaling but before interpolation
            for x_vals, y_vals in curve_list:
                plt.plot(x_vals, y_vals, '-o', alpha=0.15)  # see the raw shapes

        # 2a) Gather all x-values from all (k,N) curves to define a global min/max
        all_x = np.concatenate([c[0] for c in curve_list])
        global_min_x = all_x.min()
        global_max_x = all_x.max()
        if global_min_x == global_max_x:
            global_max_x = global_min_x + 1

        # 2b) Create a common grid
        num_points = 100
        common_grid = np.linspace(global_min_x, global_max_x, num_points)

        # 2c) Interpolate each (k,N) curve onto the common grid
        interpolated_curves = []
        for (x_vals, y_vals) in curve_list:
            if len(x_vals) < 2:
                continue  # Avoid issues with interpolation
            # Make sure x_vals is sorted
            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            y_sorted = y_vals[sort_idx]
            interp_y = np.interp(common_grid, x_sorted, y_sorted)
            interpolated_curves.append(interp_y)

        if not interpolated_curves:
            continue  # Skip models with no valid curves

        interpolated_curves = np.array(interpolated_curves)  # shape: (num_curves, num_points)
        mean_curve = interpolated_curves.mean(axis=0)

        if model_name in model_colors:
            plt.plot(common_grid, mean_curve, color=model_colors[model_name], marker=",")
        elif model_name in rl_model_colors:
            plt.plot(common_grid, mean_curve, color=rl_model_colors[model_name], marker=",")

    plt.xlabel("Normalized Generation Length")
    plt.ylabel("Accuracy")
    plt.xlim(-1.2, clamp_upper-0.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    norm_filename = f"fig1_{taskname}.png"
    os.makedirs(os.path.join(kwargs['foldername']), exist_ok=True)
    plt.savefig(os.path.join(kwargs['foldername'], norm_filename), bbox_inches="tight")
    plt.clf()
    
def plot_fig1_on_ax(ax, task, df, kwargs, include_raw=False, clamp_upper=3):
    """
    Plot the fig1 curve for a single task onto a given axis (ax).
    This code is adapted from your fig1_per_task function.
    """
    # Build normalized curves per (Model, k, N)
    curves_by_model = {}
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

    # Set axis labels, limits, and grid
    ax.set_xlim(-1.2, clamp_upper - 0.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title(task)


def fig1_all_tasks(tasks, df, kwargs, include_raw=False, clamp_upper=3, n_cols=2):
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

    fig1_all_tasks(args.tasks, df, plot_kwargs, include_raw=False, clamp_upper=3, n_cols=2)


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
    #     plot_correctness_by_ttoks_model_pairs(df, model_pair, plot_kwargs, True)
    #     plot_correctness_by_ttoks_model_pairs(df, model_pair, plot_kwargs, False)
    
    for task in args.tasks:
        df_task = df[df["task"] == task]
        fig1_per_task(task, df_task, plot_kwargs)
