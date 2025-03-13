import json
import re
import os
import shutil
import random
import itertools
import pandas as pd
import glob
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from scipy.optimize import curve_fit

import seaborn as sns
sns.set("talk")

import sys
import ipdb
import traceback

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
    "gpt-4o-mini": "cornflowerblue",
    "gpt-4o": "blue",
    "o3-mini": "purple",
    "DeepSeek-R1": "black",
    "DeepSeek-V3": "pink"
}

model_nicknames = {
    "Llama-3.1-8B-Instruct": "Ll3.1-8B",
    "Llama-3.3-70B-Instruct-Turbo": "Ll3.3-70BT",
    "Meta-Llama-3.1-405B-Instruct-Turbo": "Ll3.1-405BT", 
    "Qwen2.5-7B-Instruct": "Qw2.5-7B",
    "Qwen2.5-32B-Instruct": "Qw2.5-32B",
    "Ministral-8B-Instruct-2410": "Ministral-8B",
    "gemma-2-9b-it": "Ge2-9B",
    "DeepSeek-R1-Distill-Qwen-7B": "R1-Qw-7B",
    "DeepSeek-R1-Distill-Qwen-32B": "R1-Qw-32B",
    "DeepSeek-R1-Distill-Llama-8B": "R1-Ll-8B",
    "DeepSeek-R1-Distill-Llama-70B": "R1-Ll-70B",
    "gpt-4o-mini": "gpt4om",
    "gpt-4o": "gpt4o",
    "o3-mini": "o3-mini",
    "DeepSeek-R1": "DSR1",
    "DeepSeek-V3": "DSV3"
}

factor_to_description = {
    "k": "k (DFA size)",
    "N": "N (run length)",
    "m": "m (mult factor)",
    "l": "l (no. lines)",
    "d": "d (depth)",
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_buckets", type=int, default=4)
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--delete_old", action="store_true")
    parser.add_argument("--tasks", nargs='+', default=['dyck', 'array_idx', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies'])

    args = parser.parse_args()
    args.foldername = os.path.join(
        f"all_tasks_graphs_{args.n_buckets}buckets"
    )

    return args

def parse_kdN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_d(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = int(parsed_experimentname.group(1))
    d = int(parsed_experimentname.group(2))
    N = int(parsed_experimentname.group(3))
    return {"k": k, "d": d, "N": N, "Model": modelname}

def parse_kmN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_m(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = int(parsed_experimentname.group(1))
    m = int(parsed_experimentname.group(2))
    N = int(parsed_experimentname.group(3))
    return {"k": k, "m": m, "N": N, "Model": modelname}

def parse_kN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = int(parsed_experimentname.group(1))
    N = int(parsed_experimentname.group(2))
    return {"k": k, "N": N, "Model": modelname}

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
        match task:
            case 'dyck':
                compute_random = lambda factor_vals: 0.5
                foldername_parser = parse_kdN
                dfa_factors_order = {"k": 0, "d": 1, "N": 2}
                output_folder = "dyck/outputs"
            case 'arith':
                compute_random = lambda factor_vals: 1. / (factor_vals["k"])
                foldername_parser = parse_kmN
                dfa_factors_order = {"k": 0, "m": 1, "N": 2}
                output_folder = "arithmetic/outputs"
            case 'array_idx':
                compute_random = lambda factor_vals: 1. / (factor_vals["k"])
                foldername_parser = parse_kmN
                dfa_factors_order = {"k": 0, "m": 1, "N": 2}
                output_folder = "array_idx_mult/outputs"
            case 'even_odd':
                compute_random = lambda factor_vals: 0.5
                foldername_parser = parse_kmN
                dfa_factors_order = {"k": 0, "m": 1, "N": 2}
                output_folder = "even_odd_mult/outputs"
            case 'bool':
                compute_random = lambda factor_vals: 0.5
                foldername_parser = parse_kN
                dfa_factors_order = {"k": 0, "N": 1}
                output_folder = "nested_boolean_expression/outputs"
            case 'navigate':
                compute_random = lambda factor_vals: 0.5
                foldername_parser = parse_kdN
                dfa_factors_order = {"k": 0, "d": 1, "N": 2}
                output_folder = "navigate/outputs"
            case 'shuffled_objects':
                compute_random = lambda factor_vals: 1. / factor_vals["k"]
                foldername_parser = parse_kN
                dfa_factors_order = {"k": 0, "N": 1}
                output_folder = "shuffled_objects/outputs"
            case 'web_of_lies':
                compute_random = lambda factor_vals: 1/factor_vals["N"]
                foldername_parser = parse_kN
                dfa_factors_order = {"k": 0, "N": 1}
                output_folder = "web_of_lies/outputs"
        task_df = load_task_data(task, compute_random, foldername_parser, list(dfa_factors_order.keys()) + ["Model"], output_folder)
        if all_df is not None:
            all_df = pd.concat([all_df, task_df])
        else: all_df = task_df
    return all_df

def calculate_buckets(sub_df, n_buckets, bucket_by="No gen toks", bucket_name="Length Bucket", y_axis="Correct?", groupby_key="Model"):
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

    sub_df = sub_df.merge(bucket_avg, on=[groupby_key, bucket_name], suffixes=('', '_mean'))

    return bucket_avg, sub_df

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
    plt.ylim(0, 1)
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
        else: 
            label = None
            line_color, dot_color = "gray", "gray"
            dotsize = 2
            
        plt.plot(task_model_df["No gen toks"], task_model_df["Correct?"], color=line_color, marker=",")
        plt.scatter(peak_x, peak_y_normalized, color=dot_color, zorder=dotsize, label=label)
    
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

    plot_correctness_by_ttoks(df, plot_kwargs)

    model_pairs = [
        ("Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Llama-8B", ["even_odd"]),
        ("Llama-3.3-70B-Instruct-Turbo", "DeepSeek-R1-Distill-Llama-70B", ["arith"]),
        ("Qwen2.5-7B-Instruct", "DeepSeek-R1-Distill-Qwen-7B", ["array_idx"]),
        ("Qwen2.5-32B-Instruct", "DeepSeek-R1-Distill-Qwen-32B", ["even_odd"]),
        ("gpt-4o", "o3-mini", ["bool"]),
        ("DeepSeek-R1", "DeepSeek-V3", ["shuffled_objects"])
    ]
    for model_pair in model_pairs:
        plot_correctness_by_ttoks_model_pairs(df, model_pair, plot_kwargs, True)
        plot_correctness_by_ttoks_model_pairs(df, model_pair, plot_kwargs, False)

