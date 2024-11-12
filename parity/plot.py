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
import random
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

global foldername

def plot_requested_vs_generated():
    global foldername
    models = []
    gen_toks = []
    req_L = []

    # see correlation between requested L and actual length
    for subfolder in glob.glob(os.path.join(foldername, "*")):
        for experiment_file in glob.glob(os.path.join(subfolder, "*")):
            modelname = re.search(r'(Llama-3.+)_T', experiment_file).group(1)
            requested_length = re.search(r'request_(.+)\.json', experiment_file).group(1)
            results = json.load(open(experiment_file))

            models.extend([modelname for _ in results])
            req_L.extend([requested_length for _ in results])
            gen_toks.extend([ex["generated_tokens"] for ex in results])

    data = {
        "Model": models,
        "No gen toks": gen_toks,
        "Requested length": req_L,
    }
    # Create a DataFrame for the new data
    df = pd.DataFrame(data)

    # Plot box plots for each model size and requested length category
    plt.figure(figsize=(12, 6))
    
    # Separate data by model size
    model_data = {
        '3.2-1B': df[df['Model'].str.contains('3.2-1B')],
        '3.2-3B': df[df['Model'].str.contains('3.2-3B')],
        '3.1-8B': df[df['Model'].str.contains('3.1-8B')]
    }

    # Define colors for each model
    colors = {'3.2-1B': 'red', '3.2-3B': 'blue', '3.1-8B': 'purple'}

    # Get sorted unique requested lengths for x-axis labels
    unique_lengths = sorted(df['Requested length'].unique())
    tick_positions = []

    # Plot box plots for each requested length category for each model
    for i, (model, model_df) in enumerate(model_data.items(), start=1):
        grouped = model_df.groupby('Requested length')['No gen toks'].apply(list)
        if len(grouped) == 0: continue
        positions = [unique_lengths.index(req_len) + (i - 2) * 0.2 for req_len in grouped.index]
        # tick_positions = positions if i == 1 else tick_positions
        plt.boxplot(
            grouped, positions=positions, widths=0.15, patch_artist=True,
            boxprops=dict(facecolor=colors[model], color=colors[model]),
            medianprops=dict(color='black'), showfliers=False
        )

    # Set x-axis labels
    plt.xticks(ticks=positions, labels=unique_lengths, rotation=45)
    plt.xlabel('Requested length (no. words)')
    plt.ylabel("Actual length (no. tokens)")
    plt.title(f'Box Plot of Generated Tokens by Requested Length and Model')
    
    # Legend for the models
    legend_elements = [
        Line2D([0], [0], color=colors[model], lw=2, label=model) for model in colors
    ]
    plt.legend(handles=legend_elements, loc='upper left', fancybox=True, shadow=True)
    
    plt.savefig(os.path.join(foldername, "lengthvsrequested_boxplot.png"))
    plt.clf()

def plot(k, N, num_buckets=10):
    global foldername

    models = []
    gen_toks = []
    correct = []

    # Load data from experiment files
    for experiment_file in glob.glob(os.path.join(foldername, f"k{k}_N{N}", "*")):
        modelname = re.search(r'(Llama-3.+)_T', experiment_file).group(1)
        results = json.load(open(experiment_file))

        models.extend([modelname for _ in results])
        gen_toks.extend([ex["generated_tokens"] for ex in results])
        correct.extend([ex["correct"] for ex in results])

    # Create a DataFrame for the new data
    if len(models) == 0: return
    data = {
        "Model": models,
        "No gen toks": gen_toks,
        "Correct?": correct,
    }
    main_df = pd.DataFrame(data)

    # Calculate average correctness in specified number of buckets
    plt.figure(figsize=(10, 6))

    # Function to create buckets based on number of buckets
    def calculate_buckets(df):
        if len(df) == 0: return None
        min_len, max_len = df['No gen toks'].min(), df['No gen toks'].max()
        
        # Handle case where all 'No gen toks' values are identical
        if min_len == max_len:
            df['Length Bucket'] = pd.cut(df['No gen toks'], bins=[min_len - 0.1, max_len + 0.1], include_lowest=True)
        else:
            bins = np.linspace(min_len, max_len, num_buckets + 1)
            df['Length Bucket'] = pd.cut(df['No gen toks'], bins, include_lowest=True)
        
        bucket_avg = df.groupby(['Model', 'Length Bucket'])['Correct?'].mean().reset_index()
        bucket_avg['Bucket Center'] = bucket_avg['Length Bucket'].apply(lambda x: x.mid)
        return bucket_avg


    # Get bucketed averages for each model
    model_1B_buckets = calculate_buckets(main_df[main_df['Model'].str.contains('3.2-1B')].sort_values(by="No gen toks"))
    model_3B_buckets = calculate_buckets(main_df[main_df['Model'].str.contains('3.2-3B')].sort_values(by="No gen toks"))
    model_8B_buckets = calculate_buckets(main_df[main_df['Model'].str.contains('3.1-8B')].sort_values(by="No gen toks"))

    # Plot the average correctness for each model size
    if model_1B_buckets is not None: plt.plot(model_1B_buckets['Bucket Center'], model_1B_buckets['Correct?'], color="red", label="3.2 1B")
    if model_3B_buckets is not None: plt.plot(model_3B_buckets['Bucket Center'], model_3B_buckets['Correct?'], color="blue", label="3.2 3B")
    if model_8B_buckets is not None: plt.plot(model_8B_buckets['Bucket Center'], model_8B_buckets['Correct?'], color="purple", label="3.1 8B")

    plt.axhline(y=1./int(k), linestyle=':')
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel('Average Correctness')
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.title(f'Average Correctness vs. No. of Generated Tokens (k={k}, N={N}, Buckets={num_buckets})')
    plt.legend(loc='upper left', fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(os.path.join(foldername, f"k{k}_N{N}_avg_correctness_{num_buckets}buckets.png"))
    plt.clf()

def plot_by_k(k):
    global foldername

    Ns_to_data = {}

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(foldername, f"k{k}_N*")):
        N = re.search(r'_N(\d+)', subfolder).group(1)
        if N not in Ns_to_data: Ns_to_data[N] = {}
        for experiment_file in glob.glob(os.path.join(subfolder, "*.json")):
            modelname = re.search(r'(Llama-3.+)_T', experiment_file).group(1)
            if modelname not in Ns_to_data[N]: Ns_to_data[N][modelname] = []
            results = json.load(open(experiment_file))

            Ns_to_data[N][modelname].extend([ex["correct"] for ex in results])

    models = []
    Ns = []
    accuracy = []
    for N, modeldata in Ns_to_data.items():
        for modelname, correctness in modeldata.items():
            models.append(modelname)
            Ns.append(int(N))
            accuracy .append(sum(correctness) / len(correctness))

    # Create a DataFrame for the new data
    data = {
        "Model": models,
        "N": Ns,
        "Accuracy": accuracy,
    }
    df = pd.DataFrame(data)

    # Calculate average correctness in specified number of buckets
    plt.figure(figsize=(10, 6))
    model_1B = df[df['Model'].str.contains('3.2-1B')].sort_values(by='N')
    model_3B = df[df['Model'].str.contains('3.2-3B')].sort_values(by='N')
    model_8B = df[df['Model'].str.contains('3.1-8B')].sort_values(by='N')

    # Plot the average correctness for each model size
    plt.scatter(model_1B['N'], model_1B['Accuracy'], color="red", label="3.2 1B")
    plt.scatter(model_3B['N'], model_3B['Accuracy'], color="blue", label="3.2 3B")
    plt.scatter(model_8B['N'], model_8B['Accuracy'], color="purple", label="3.1 8B")
    
    plt.axhline(y=1./int(k), linestyle=':')

    # Customize plot labels and legend
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel("N")
    plt.title(f'Average Correctness vs. N (k={k})')
    plt.legend(loc='upper left', fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(os.path.join(foldername, f"k{k}_scatter.png"))
    plt.clf()

def plot_by_N(N):
    global foldername

    ks_to_data = {}

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(foldername, f"k*")):
        parsed_experimentname = re.search(r'k(\d+)_N(\d+)', subfolder)
        if parsed_experimentname is None: continue
        k = parsed_experimentname.group(1)
        this_N = parsed_experimentname.group(2)
        if int(this_N) != int(N): continue
        if k not in ks_to_data: ks_to_data[k] = {}
        for experiment_file in glob.glob(os.path.join(subfolder, "*.json")):
            modelname = re.search(r'(Llama-3.+)_T', experiment_file).group(1)
            if modelname not in ks_to_data[k]: ks_to_data[k][modelname] = []
            results = json.load(open(experiment_file))

            ks_to_data[k][modelname].extend([ex["correct"] for ex in results])

    models = []
    ks = []
    accuracy = []
    random_baseline_nums = []
    for k, modeldata in ks_to_data.items():
        for modelname, correctness in modeldata.items():
            models.append(modelname)
            ks.append(int(k))
            accuracy.append(sum(correctness) / len(correctness))
            random_baseline_nums.append(1./int(k))

    # Create a DataFrame for the new data
    data = {
        "Model": models,
        "k": ks,
        "Accuracy": accuracy,
        "random": random_baseline_nums,
    }
    df = pd.DataFrame(data)

    # Calculate average correctness in specified number of buckets
    plt.figure(figsize=(10, 6))
    model_1B = df[df['Model'].str.contains('3.2-1B')].sort_values(by='k')
    model_3B = df[df['Model'].str.contains('3.2-3B')].sort_values(by='k')
    model_8B = df[df['Model'].str.contains('3.1-8B')].sort_values(by='k')

    # Plot the average correctness for each model size
    plt.scatter(model_1B['k'], model_1B['Accuracy'], color="red", label="3.2 1B")
    plt.scatter(model_3B['k'], model_3B['Accuracy'], color="blue", label="3.2 3B")
    plt.scatter(model_8B['k'], model_8B['Accuracy'], color="purple", label="3.1 8B")
    
    plt.plot(ks, random_baseline_nums, color="gray", label="random", linestyle="--")

    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel("k")
    plt.title(f'Average Correctness vs. k (N={N})')
    plt.legend(loc='upper left', fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(os.path.join(foldername, f"N{N}_scatter.png"))
    plt.clf()

def plot_by_model(modelname, num_buckets=10):
    global foldername

    kNs = []
    gen_toks = []
    correct = []
    method_to_performance = {}

    # Load data from experiment files
    for subfolder in glob.glob(os.path.join(foldername, f"k*")):
        parsed_experimentname = re.search(r'k(\d+)_N(\d+)', subfolder)
        if parsed_experimentname is None: continue
        k = parsed_experimentname.group(1)
        N = parsed_experimentname.group(2)
        for experiment_file in glob.glob(os.path.join(subfolder, "*")):
            if modelname not in experiment_file: continue
            results = json.load(open(experiment_file))
            if "detail" in experiment_file:
                method = "detailed" 
            if "none" in experiment_file:
                method = "none" 
            if "brief" in experiment_file:
                method = "brief" 

            kNs.extend([f"k{k}_N{N}" for _ in results])
            gen_toks.extend([ex["generated_tokens"] for ex in results])
            correct.extend([ex["correct"] for ex in results])
            if method not in method_to_performance: method_to_performance[method] = []
            method_to_performance[method].extend([(ex["correct"], ex["generated_tokens"]) for ex in results])

    # Create a DataFrame for the new data
    data = {
        "kN": kNs,
        "No gen toks": gen_toks,
        "Correct?": correct,
    }
    main_df = pd.DataFrame(data)

    # Calculate average correctness in specified number of buckets
    plt.figure(figsize=(10, 6))


    # Function to create buckets based on number of buckets
    def calculate_buckets_samerange(df):
        if len(df) == 0: return None
        min_len, max_len = df['No gen toks'].min(), df['No gen toks'].max()
        
        bins = np.linspace(min_len, max_len, num_buckets + 1)
        df['Length Bucket'] = pd.cut(df['No gen toks'], bins, include_lowest=True)
        
        bucket_avg = df.groupby(['kN', 'Length Bucket'])['Correct?'].mean().reset_index()
        bucket_avg['Bucket Center'] = bucket_avg['Length Bucket'].apply(lambda x: x.mid)
        return bucket_avg

    def calculate_buckets_samesize(df):
        # Use qcut to create equal-sized buckets by count
        df['Length Bucket'] = pd.qcut(df['No gen toks'], q=num_buckets, duplicates='drop')
        bucket_avg = df.groupby(['kN', 'Length Bucket'])['Correct?'].mean().reset_index()
        
        # Calculate the bucket center by averaging bin edges
        bucket_avg['Bucket Center'] = bucket_avg['Length Bucket'].apply(lambda x: (x.left + x.right) / 2)
        
        return bucket_avg


    kNs_buckets = {}
    sorted_kNs = []
    for kN in set(kNs):
        parsed_kN = re.search(r'k(\d+)_N(\d+)', kN)
        k = int(parsed_kN.group(1))
        N = int(parsed_kN.group(2))
        sorted_kNs.append((k, N))
    sorted_kNs = sorted(sorted_kNs, key=lambda ex: ex[1])
    for idx, kN in enumerate(sorted_kNs):
        kN_str = f"k{kN[0]}_N{kN[1]}"
        # Get bucketed averages for each kN
        kNs_buckets[kN_str] = calculate_buckets_samesize(main_df[main_df['kN'].str.contains(kN_str)].sort_values(by="No gen toks"))

        # Plot the average correctness for each kN
        if kNs_buckets[kN_str] is not None: 
            plt.plot(kNs_buckets[kN_str]['Bucket Center'], kNs_buckets[kN_str]['Correct?'], color=(1.*idx/len(set(kNs)), 0, 0), label=kN)
    for method in method_to_performance:
        avg_correctness = sum(ex[0] for ex in method_to_performance[method]) / len(method_to_performance[method])
        avg_toks = sum(ex[1] for ex in method_to_performance[method]) / len(method_to_performance[method])
        plt.plot(avg_toks, avg_correctness, label = method, marker="^" if method == "none" else "." )

    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel('Average Correctness')
    plt.xlabel("No. of Generated Tokens (Binned)")
    plt.title(f'Average Correctness vs. No. of Generated Tokens ({modelname}, Buckets={num_buckets})')
    plt.legend(loc='upper right', fancybox=True, shadow=True)

    # Save and clear the figure
    plt.savefig(os.path.join(foldername, f"{modelname}_{num_buckets}buckets.png"))
    plt.clf()

if __name__ == "__main__":
    global foldername
    foldername = sys.argv[1]
    plot_requested_vs_generated()
    for subfolder in glob.glob(os.path.join(foldername, "k*")):
        parsed_foldername = re.search(r'k(\d+)_N(\d+)', subfolder)
        if parsed_foldername is None: continue
        k = parsed_foldername.group(1)
        N = parsed_foldername.group(2)
        plot_by_k(k)
        plot_by_N(N)
        plot(k, N)
    for modelname in ["Llama-3.1-8B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.2-1B-Instruct"]:
        plot_by_model(modelname, num_buckets=10)


