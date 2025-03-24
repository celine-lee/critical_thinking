import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
sns.set("talk")

task_full_names = {
    'arith': "Multi-Step Arithmetic",
    'array_idx': "Index Tracking",
    'dyck': "Dyck-$D$",
    'navigate': "Navigate",
    'even_odd': "Even/Odd Tracking",
    'cruxeval': "CRUXEval",
    'bool': "Nested Boolean Expression",
    'web_of_lies': "Web of Lies",
    'shuffled_objects': "Shuffled Objects",
    'logical_deduction': "Logical Deduction",
}

all_models_size_ordered = [
    "Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Ministral-8B-Instruct-2410",
    "Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B",
    "gemma-2-9b-it",
    "Qwen2.5-32B-Instruct",
    "DeepSeek-R1-Distill-Qwen-32B",
    "Llama-3.3-70B-Instruct-Turbo",
    "DeepSeek-R1-Distill-Llama-70B",
    "Meta-Llama-3.1-405B-Instruct-Turbo",
    # "gpt-4o-mini",
    "o3-mini",
    "gpt-4o",
    "DeepSeek-V3",
    "DeepSeek-R1",
]

models_in_order = [
    "Qwen2.5-7B-Instruct",
    "Ministral-8B-Instruct-2410",
    "Llama-3.1-8B-Instruct",
    "gemma-2-9b-it",
    "Qwen2.5-32B-Instruct",
    "Llama-3.3-70B-Instruct-Turbo",
    "Meta-Llama-3.1-405B-Instruct-Turbo",
    "gpt-4o-mini",
    "DeepSeek-V3",
    "gpt-4o",
]

rl_models_in_order = [
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Llama-70B",
    "DeepSeek-R1",
    "o3-mini",
]

# Set up color maps. (Make sure models_in_order and rl_models_in_order
# are defined in your environment or passed via kwargs.)
cmap_blues = matplotlib.colormaps.get_cmap("Blues")
cmap_oranges = matplotlib.colormaps.get_cmap("Oranges")
nonrl_model_colors = {}
n_nonrl = len(models_in_order)
for i, model in enumerate(models_in_order):
    # fraction in [0.2 to 1.0]
    fraction = 0.4 + 0.8 * (i / (n_nonrl - 1))
    nonrl_model_colors[model] = cmap_blues(fraction)

rl_model_colors = {}
n_rl = len(rl_models_in_order)
for i, model in enumerate(rl_models_in_order):
    # fraction in [0.2:
    fraction = 0.4 + 0.8 * (i / (n_rl - 1))
    rl_model_colors[model] = cmap_oranges(fraction)

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

def get_task_info(task):
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
        case 'logical_deduction':
            compute_random = lambda factor_vals: 1/factor_vals["k"]
            foldername_parser = parse_kN
            dfa_factors_order = {"k": 0, "N": 1}
            output_folder = "logical_deduction/outputs"
        case 'cruxeval':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kN
            dfa_factors_order = {"k": 0, "N": 1}
            output_folder = "cruxeval/outputs_straightlined"
    return compute_random, foldername_parser, dfa_factors_order, output_folder

def calculate_buckets(sub_df, n_buckets, bucket_by="No gen toks", bucket_name="Length Bucket", y_axis="Correct?", groupby_key="Model", get_precision_metrics=False):
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