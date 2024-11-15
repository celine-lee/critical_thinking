import torch

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
from matplotlib.lines import Line2D
from collections import defaultdict

foldername = sys.argv[1]
temperature = sys.argv[2]

model_colors = {
    "3.1-8B": "purple",
    "3.2-3B": "blue",
    "3.2-1B": "red",
}
method_markers = {
    # "brief": "d",
    "none": "*",
    # "detail": "^",
    "states_long": "o",
    "states_short": ".",
}
method_linestyle = {
    # "brief": ":",
    "none": "--", # usually unused
    # "detail": "-.",
    "states_long": "-",
    "states_short": "--"
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
    parsed_experimentname = re.search(rf'k(\d+)_N(\d+)_t(\d+)', subfolder)
    if parsed_experimentname is None: continue
    k = parsed_experimentname.group(1)
    N = parsed_experimentname.group(2)
    t = parsed_experimentname.group(3)
    for experiment_file in glob.glob(os.path.join(subfolder, "*")):
        if f"T{temperature}" not in experiment_file: continue
        modelname = re.search(r'(Llama-3.+)_T', experiment_file).group(1)
        results = json.load(open(experiment_file))
        for methodname in method_markers: 
            if methodname in experiment_file: break
        if methodname not in experiment_file: continue

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
    "Correct?": correct
}
# Create a DataFrame for the new data
df = pd.DataFrame(data)
# Create a column for grouping
df['Group'] = list(zip(df['Method'], df['Model'], df['k'], df['N'], df['t']))

# Find the minimum size across all groups
group_sizes = df['Group'].value_counts()
min_size = group_sizes.min()

# Subsample each group explicitly
balanced_data = []
for group, size in group_sizes.items():
    group_df = df[df['Group'] == group]
    balanced_data.append(group_df.sample(n=min_size, random_state=42))

# Combine all subsampled groups
balanced_df = pd.concat(balanced_data).reset_index(drop=True)

# Separate data by model size
model_data = {model_name: df[df['Model'].str.contains(model_name)].sort_values(by="No gen toks") for model_name in model_colors}
balanced_model_data = {model_name: balanced_df[balanced_df['Model'].str.contains(model_name)].sort_values(by="No gen toks") for model_name in model_colors}

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
    unique_lengths = sorted(df['Method'].unique())
    tick_positions = []

    plotted_something = False
    # Plot box plots for each requested length category for each model
    for i, (model, model_df) in enumerate(model_data.items(), start=1):
        grouped = model_df.groupby('Method')['No gen toks'].apply(list)
        if len(grouped) == 0: continue
        positions = [unique_lengths.index(req_len) + (i - 2) * 0.2 for req_len in grouped.index]
        # tick_positions = positions if i == 1 else tick_positions
        plt.boxplot(
            grouped, positions=positions, widths=0.15, patch_artist=True,
            boxprops=dict(facecolor=model_colors[model], color=model_colors[model]),
            medianprops=dict(color='black'), showfliers=False
        )
        plotted_something = True
    if not plotted_something: return

    # Set x-axis labels
    plt.xticks(ticks=positions, labels=unique_lengths, rotation=45)
    plt.xlabel('Method')
    plt.ylabel("Actual length (no. tokens)")
    plt.title(f'Box Plot of Generated Tokens by Method and Model')
    
    # Legend for the models
    legend_elements = [
        Line2D([0], [0], color=model_colors[model], lw=2, label=model) for model in model_colors
    ]
    plt.legend(handles=legend_elements, loc='upper left', fancybox=True, shadow=True)
    
    plt.savefig(os.path.join(foldername, f"lengthvsrequested_boxplot_T{temperature}.png"))
    plt.clf()


# to check that the test toks necessary for correct inference increases with N / (k&t)
def plot_N_vs_tts(modelname, k, t):
    # Filter the data for the specific model, k, and t
    filtered_data = df[
        (df['Model'].str.contains(modelname)) & 
        (df['k'] == k) & 
        (df['t'] == t) & 
        (df['Correct?'] == True)
    ]
    
    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No correct examples found for Model: {modelname}, k={k}, t={t}.")
        return
    
    # Create the scatterplot
    plt.figure(figsize=(12, 6))
    plt.scatter(filtered_data['No gen toks'], filtered_data['N'], alpha=0.7, color='blue')
    
    # Customize the plot
    plt.xlabel('No. of Generated Tokens')
    plt.ylabel('N')
    plt.title(f'Scatter Plot for Model: {modelname}, k={k}, t={t} (Correct Examples)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(os.path.join(foldername, f"N_vs_tts_{modelname}_k{k}_t{t}_T{temperature}.png"))
    plt.clf()


def plot(k, N, t, num_buckets):
    # Calculate average correctness in specified number of buckets
    plt.figure(figsize=(10, 6))

    # Function to create buckets based on number of buckets
    min_len, max_len = balanced_df['No gen toks'].min(), balanced_df['No gen toks'].max()
    bins = np.linspace(min_len, max_len, num_buckets + 1)

    def calculate_buckets_samerange(sub_df):
        if len(sub_df) == 0: return None
        
        sub_df['Length Bucket'] = pd.cut(sub_df['No gen toks'], bins, include_lowest=True)
        
        bucket_avg = df.groupby(['Model', 'Length Bucket'])['Correct?'].mean().reset_index()
        bucket_avg['Bucket Center'] = bucket_avg['Length Bucket'].apply(lambda x: (x.left + x.right) / 2)
        return bucket_avg

    def calculate_buckets_samesize(sub_df):
        # Use qcut to create equal-sized buckets by count
        sub_df['Length Bucket'] = pd.qcut(sub_df['No gen toks'], q=num_buckets, duplicates='drop')
        bucket_avg = sub_df.groupby(['Model', 'Length Bucket'])['Correct?'].mean().reset_index()
        
        # Calculate the bucket center by averaging bin edges
        bucket_avg['Bucket Center'] = bucket_avg['Length Bucket'].apply(lambda x: (x.left + x.right) / 2)
        
        return bucket_avg

    plotted_something = False
    # Get bucketed averages for each model
    model_buckets = {}
    for model_substring in model_colors:
        model_df = balanced_model_data[model_substring]
        model_df = model_df[model_df['k'].str.contains(f"{k}") & model_df['N'].str.contains(f"{N}") & model_df['t'].str.contains(f"{t}") ]
        model_buckets[model_substring] = calculate_buckets_samesize(model_df)
        if model_buckets[model_substring] is None: continue
        # Plot the average correctness for each model size and method
        plt.plot(model_buckets[model_substring]['Bucket Center'], model_buckets[model_substring]['Correct?'], color=model_colors[model_substring], label=model_substring)
        plotted_something = True

    if not plotted_something: return
    plt.axhline(y=1./int(k), linestyle=':')
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel('Average Correctness')
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.title(f'Average Correctness vs. No. of Generated Tokens (k={k}, N={N}, t={t}, Buckets={num_buckets})')
    plt.legend(loc='upper left', fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(os.path.join(foldername, f"k{k}_N{N}_t{t}_avg_correctness_{num_buckets}buckets_T{temperature}.png"))
    plt.clf()

def plot_by_k(k, t):

    Ns_to_data = {}
    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(foldername, f"k{k}_N*_t{t}")):
        N = re.search(r'_N(\d+)', subfolder).group(1)
        if N not in Ns_to_data: Ns_to_data[N] = {}
        for experiment_file in glob.glob(os.path.join(subfolder, "*.json")):
            if f"T{temperature}" not in experiment_file: continue
            modelname = re.search(r'(Llama-3.+)_T', experiment_file).group(1)
            for methodname in method_markers: 
                if methodname in experiment_file: break
            if methodname not in experiment_file: continue
            if (modelname, methodname) not in Ns_to_data[N]: Ns_to_data[N][(modelname, methodname)] = []
            results = json.load(open(experiment_file))

            Ns_to_data[N][(modelname, methodname)].extend([ex["correct"] for ex in results])

    models = []
    methods = []
    Ns = []
    accuracy = []
    confidence_margins = []

    for N, modeldata in Ns_to_data.items():
        if len(modeldata) == 0: continue
        min_examples = min(len(data) for data in modeldata.values())
        for (modelname, methodname), correctness in modeldata.items():
            sampled_correctness = random.sample(correctness, min_examples)
            models.append(modelname)
            methods.append(methodname)
            Ns.append(int(N))
            correct_vals = sampled_correctness
            acc = sum(correct_vals) / len(correct_vals)
            accuracy.append(acc)
            
            # Compute 95% confidence margin for accuracy
            ci = stats.sem(correct_vals) * stats.t.ppf((1 + 0.95) / 2., len(correct_vals) - 1)
            confidence_margins.append(ci)

    # Create a DataFrame for the new data
    data = {
        "Model": models,
        "Method": methods,
        "N": Ns,
        "Accuracy": accuracy,
        "Confidence Margin": confidence_margins,
    }
    if len(models) == 0: return
    df = pd.DataFrame(data)

    # Calculate average correctness in specified number of buckets
    plt.figure(figsize=(10, 6))

    to_plot_data = {}
    for model_substring in model_colors:
        if model_substring not in to_plot_data: to_plot_data[model_substring] = {}
        for method in method_linestyle:
            to_plot_data[model_substring][method] = df[df['Model'].str.contains(model_substring) & df['Method'].str.contains(method)].sort_values(by='N')
            # Plot the average correctness for each model size and method

    for method in method_linestyle:
        for model_substring in model_colors:
            model_data = to_plot_data[model_substring][method]
            N_values = model_data['N']
            if len(N_values) == 0: continue
            accuracies = model_data['Accuracy']
            ci_margins = model_data['Confidence Margin']
            plt.plot(
                N_values, accuracies,
                color=model_colors[model_substring],
                marker=method_markers[method],
                linestyle=method_linestyle[method],
                label=model_substring
            )

            # Add confidence interval shading
            plt.fill_between(
                N_values,
                accuracies - ci_margins,
                accuracies + ci_margins,
                color=model_colors[model_substring],
                alpha=0.2
            )

        plt.axhline(y=1./int(k), linestyle=':')

        # Customize plot labels and legend
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.xlabel("N")
        plt.title(f'Average Correctness vs. N (k={k} t={t}) {method}')
        plt.legend(loc='upper left', fancybox=True, shadow=True)

        # Save and clear the figure
        plt.savefig(os.path.join(foldername, f"k{k}_t{t}_{method}_T{temperature}.png"))
        plt.clf()

    for model_substring in model_colors:
        plotted_something = False
        for method in method_linestyle:

            model_data = to_plot_data[model_substring][method]
            N_values = model_data['N']
            if len(N_values) == 0: continue
            accuracies = model_data['Accuracy']
            ci_margins = model_data['Confidence Margin']
            plt.plot(
                N_values, accuracies,
                color=model_colors[model_substring],
                marker=method_markers[method],
                linestyle=method_linestyle[method],
                label=method
            )
            plotted_something = True
            # Add confidence interval shading
            plt.fill_between(
                N_values,
                accuracies - ci_margins,
                accuracies + ci_margins,
                color=model_colors[model_substring],
                alpha=0.2
            )
        if not plotted_something: continue
        plt.axhline(y=1./int(k), linestyle=':')

        # Customize plot labels and legend
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.xlabel("N")
        plt.title(f'Average Correctness vs. N (k={k} t={t}) {model_substring}')
        plt.legend(loc='upper left', fancybox=True, shadow=True)

        # Save and clear the figure
        plt.savefig(os.path.join(foldername, f"k{k}_t{t}_{model_substring}_T{temperature}.png"))
        plt.clf()

def plot_by_N(N, t):

    ks_to_data = {}

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(foldername, f"k*_N{N}_t{t}")):
        parsed_experimentname = re.search(r'k(\d+)', subfolder)
        if parsed_experimentname is None: continue
        k = parsed_experimentname.group(1)
        if k not in ks_to_data: ks_to_data[k] = {}
        for experiment_file in glob.glob(os.path.join(subfolder, "*.json")):
            if f"T{temperature}" not in experiment_file: continue
            modelname = re.search(r'(Llama-3.+)_T', experiment_file).group(1)
            for methodname in method_markers: 
                if methodname in experiment_file: break
            if methodname not in experiment_file: continue
            if (modelname, methodname) not in ks_to_data[k]: ks_to_data[k][(modelname, methodname)] = []
            results = json.load(open(experiment_file))
            ks_to_data[k][(modelname,methodname)].extend([ex["correct"] for ex in results])

    models = []
    methods = []
    ks = []
    accuracy = []
    random_baseline_nums = []
    confidence_margins = []

    for k, modeldata in ks_to_data.items():
        if len(modeldata) == 0: continue 
        min_examples = min(len(data) for data in modeldata.values())
        for (modelname, methodname), correctness in modeldata.items():
            sampled_correctness = random.sample(correctness, min_examples)
            models.append(modelname)
            methods.append(methodname)
            ks.append(int(k))
            correct_vals = sampled_correctness
            acc = sum(correct_vals) / len(correct_vals)
            accuracy.append(acc)
            
            # Compute 95% confidence margin for accuracy
            ci = stats.sem(correct_vals) * stats.t.ppf((1 + 0.95) / 2., len(correct_vals) - 1)
            confidence_margins.append(ci)
            random_baseline_nums.append(1./int(k))

    if len(models) == 0: return
    # Create a DataFrame for the new data
    data = {
        "Model": models,
        "Method": methods,
        "k": ks,
        "Accuracy": accuracy,
        "Confidence Margin": confidence_margins,
        "Random Baseline": random_baseline_nums,
    }
    df = pd.DataFrame(data).sort_values(by="k")

    # Organize data by model and method for plotting
    to_plot_data = {}
    for model_substring in model_colors:
        if model_substring not in to_plot_data: to_plot_data[model_substring] = {}
        for method in method_linestyle:
            subset = df[df['Model'].str.contains(model_substring) & df['Method'].str.contains(method)]
            to_plot_data[model_substring][method] = subset.sort_values(by='k')

    for method in method_linestyle:
        plt.figure(figsize=(10, 6))
        plotted_something = False
        for model_substring in model_colors:
            model_data = to_plot_data[model_substring][method]
            k_values = model_data['k']
            if len(k_values) == 0: continue
            accuracies = model_data['Accuracy']
            ci_margins = model_data['Confidence Margin']

            plt.plot(
                k_values, accuracies,
                color=model_colors[model_substring],
                marker=method_markers[method],
                linestyle=method_linestyle[method],
                label=model_substring
            )
            plotted_something = True

            # Add confidence interval shading
            plt.fill_between(
                k_values,
                accuracies - ci_margins,
                accuracies + ci_margins,
                color=model_colors[model_substring],
                alpha=0.2
            )
        if not plotted_something: continue
        plt.plot(df["k"], df["Random Baseline"], color="gray", label="Random Baseline", linestyle="--")

        # Customize plot labels and legend
        plt.xlim(xmin=0)
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.xlabel("k")
        plt.title(f'Average Correctness vs. k (N={N} t={t}) ({method})')
        plt.legend(loc='upper left', fancybox=True, shadow=True)

        # Save and clear the figure
        plt.savefig(os.path.join(foldername, f"N{N}_t{t}_{method}_T{temperature}.png"))
        plt.clf()

    for model_substring in model_colors:
        plotted_something = False
        plt.figure(figsize=(10, 6))
        for method in method_linestyle:
            model_data = to_plot_data[model_substring][method]
            k_values = model_data['k']
            if len(k_values) == 0: continue
            accuracies = model_data['Accuracy']
            ci_margins = model_data['Confidence Margin']

            plt.plot(
                k_values, accuracies,
                color=model_colors[model_substring],
                marker=method_markers[method],
                linestyle=method_linestyle[method],
                label=method
            )
            plotted_something = True

            # Add confidence interval shading
            plt.fill_between(
                k_values,
                accuracies - ci_margins,
                accuracies + ci_margins,
                color=model_colors[model_substring],
                alpha=0.2
            )
        if not plotted_something: continue
        plt.plot(df["k"], df["Random Baseline"], color="gray", label="Random Baseline", linestyle="--")

        # Customize plot labels and legend
        plt.xlim(xmin=0)
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.xlabel("k")
        plt.title(f'Average Correctness vs. k (N={N} t={t}) ({model_substring})')
        plt.legend(loc='upper left', fancybox=True, shadow=True)

        # Save and clear the figure
        plt.savefig(os.path.join(foldername, f"N{N}_t{t}_{model_substring}_T{temperature}.png"))
        plt.clf()

def plot_by_model(modelname, t, num_buckets=10):

    ks = []
    Ns = []
    gen_toks = []
    correct = []
    method_to_performance = {}

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(foldername, f"k*")):
        parsed_experimentname = re.search(rf'k(\d+)_N(\d+)_t{t}', subfolder)
        if parsed_experimentname is None: continue
        k = parsed_experimentname.group(1)
        N = parsed_experimentname.group(2)
        for experiment_file in glob.glob(os.path.join(subfolder, "*")):
            if f"T{temperature}" not in experiment_file: continue
            if modelname not in experiment_file: continue
            results = json.load(open(experiment_file))
            for methodname in method_markers: 
                if methodname in experiment_file: break
            if methodname not in experiment_file: continue
            ks.extend([k for _ in results])
            Ns.extend([N for _ in results])
            gen_toks.extend([ex["generated_tokens"] for ex in results])
            correct.extend([ex["correct"] for ex in results])
            if methodname not in method_to_performance: method_to_performance[methodname] = []
            method_to_performance[methodname].extend([(ex["correct"], ex["generated_tokens"]) for ex in results])

    # Create a DataFrame for the new data
    data = {
        "k": ks,
        "N": Ns,
        "No gen toks": gen_toks,
        "Correct?": correct,
    }
    main_df = pd.DataFrame(data)

    # Calculate average correctness in specified number of buckets
    plt.figure(figsize=(10, 6))


    # Function to create buckets based on number of buckets
    def calculate_buckets_samerange(df, groupby_key):
        if len(df) == 0: return None
        min_len, max_len = df['No gen toks'].min(), df['No gen toks'].max()
        
        bins = np.linspace(min_len, max_len, num_buckets + 1)
        df['Length Bucket'] = pd.cut(df['No gen toks'], bins, include_lowest=True)
        
        bucket_avg = df.groupby([groupby_key, 'Length Bucket'])['Correct?'].mean().reset_index()
        bucket_avg['Bucket Center'] = bucket_avg['Length Bucket'].apply(lambda x: x.mid)
        return bucket_avg

    def calculate_buckets_samesize(df, groupby_key):
        # Use qcut to create equal-sized buckets by count
        df['Length Bucket'] = pd.qcut(df['No gen toks'], q=num_buckets, duplicates='drop')
        bucket_avg = df.groupby([groupby_key, 'Length Bucket'])['Correct?'].mean().reset_index()
        
        # Calculate the bucket center by averaging bin edges
        bucket_avg['Bucket Center'] = bucket_avg['Length Bucket'].apply(lambda x: (x.left + x.right) / 2)
        
        return bucket_avg

    for groupby_key in {"k", "N"}:
        buckets = {}
        sorted_groupby_key = sorted(list(set(data[groupby_key])))
        plotted_something = False
        for idx, key_option in enumerate(sorted_groupby_key):
            # Get bucketed averages for each kN
            buckets[key_option] = calculate_buckets_samesize(main_df[main_df[groupby_key].str.contains(key_option)].sort_values(by="No gen toks"), groupby_key)

            # Plot the average correctness for each kN
            if buckets[key_option] is not None: 
                plt.plot(buckets[key_option]['Bucket Center'], buckets[key_option]['Correct?'], color=(1.*idx/len(set(data[groupby_key])), 0, 0), label=key_option)
                plotted_something = True
        for method in method_to_performance:
            avg_correctness = sum(ex[0] for ex in method_to_performance[method]) / len(method_to_performance[method])
            avg_toks = sum(ex[1] for ex in method_to_performance[method]) / len(method_to_performance[method])
            plt.plot(avg_toks, avg_correctness, label = method, marker=method_markers[method])
            plotted_something = True

        if not plotted_something: continue
        # Customize plot labels and legend
        plt.xlim(xmin=0)
        plt.ylim(0, 1)
        plt.ylabel('Average Correctness')
        plt.xlabel("No. of Generated Tokens (Binned)")
        plt.title(f'Average Correctness vs. No. of Generated Tokens ({modelname}, grouped by {groupby_key} Buckets={num_buckets})')
        plt.legend(loc='upper right', fancybox=True, shadow=True)

        # Save and clear the figure
        plt.savefig(os.path.join(foldername, f"{modelname}_{num_buckets}buckets_groupedby{groupby_key}_T{temperature}.png"))
        plt.clf()



if __name__ == "__main__":
    plot_requested_vs_generated()
    all_ks = set()
    all_Ns = set()
    all_ts = set()
    for subfolder in glob.glob(os.path.join(foldername, "k*")):
        parsed_foldername = re.search(r'k(\d+)_N(\d+)_t(\d+)', subfolder)
        if parsed_foldername is None: continue
        k = parsed_foldername.group(1)
        N = parsed_foldername.group(2)
        t = parsed_foldername.group(3)
        all_ks.add(k)
        all_Ns.add(N)
        all_ts.add(t)
    for k in all_ks:
        for t in all_ts:
            if t > k: continue
            plot_by_k(k, t)
            for N in all_Ns:
                plot(k, N, t, num_buckets=5)
    for N in all_Ns:
        for t in all_ts:
            plot_by_N(N, t)
    
    for modelname in ["Llama-3.1-8B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.2-1B-Instruct"]:
        for t in all_ts:
            plot_by_model(modelname, t, num_buckets=5)


