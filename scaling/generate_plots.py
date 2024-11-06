foldername = "zeroshot_outputs"

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import json
import re
import glob
import ipdb
import traceback
import os

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook


# everything at once: each table is a domain
def plot_all(methodmodels, avg_gen_toks, pass_at_k, logprobs_correctness, domain, title_prefix=""):
    if len(methodmodels) == 0: return
    data = {
        "Method": methodmodels,
        "Avg no gen toks": avg_gen_toks,
        "pass_at_k": pass_at_k,
        "logprobs_correctness": logprobs_correctness,
    }

    # Create a DataFrame for the new data
    df = pd.DataFrame(data)

    # Plotting the new data grouped by model size
    plt.figure(figsize=(10, 6))
    
    # Grouped by model size
    model_1B_sampling = df[(df['Method'].str.contains('3.2-1B')) & (df['Method'].str.contains('_N'))].sort_values(by='Avg no gen toks')
    model_3B_sampling = df[df['Method'].str.contains('3.2-3B') & df['Method'].str.contains('_N')].sort_values(by='Avg no gen toks')
    model_8B_sampling = df[df['Method'].str.contains('3.1-8B') & df['Method'].str.contains('_N')].sort_values(by='Avg no gen toks')
    model_1B_beam = df[df['Method'].str.contains('3.2-1B') & df['Method'].str.contains('beam')].sort_values(by='Avg no gen toks')
    model_3B_beam = df[df['Method'].str.contains('3.2-3B') & df['Method'].str.contains('beam')].sort_values(by='Avg no gen toks')
    model_8B_beam = df[df['Method'].str.contains('3.1-8B') & df['Method'].str.contains('beam')].sort_values(by='Avg no gen toks')
    model_1B_direct = df[df['Method'].str.contains('3.2-1B') & (df['Method'].str.contains('basic') | df['Method'].str.contains('greedy'))].sort_values(by='Avg no gen toks')
    model_3B_direct = df[df['Method'].str.contains('3.2-3B') & (df['Method'].str.contains('basic') | df['Method'].str.contains('greedy'))].sort_values(by='Avg no gen toks')
    model_8B_direct = df[df['Method'].str.contains('3.1-8B') & (df['Method'].str.contains('basic') | df['Method'].str.contains('greedy'))].sort_values(by='Avg no gen toks')
    model_1B_cotdecoding = df[df['Method'].str.contains('3.2-1B') & df['Method'].str.contains('cotdecoding')].sort_values(by='Avg no gen toks')
    model_3B_cotdecoding = df[df['Method'].str.contains('3.2-3B') & df['Method'].str.contains('cotdecoding')].sort_values(by='Avg no gen toks')
    model_8B_cotdecoding = df[df['Method'].str.contains('3.1-8B') & df['Method'].str.contains('cotdecoding')].sort_values(by='Avg no gen toks')


    plt.plot(model_1B_direct['Avg no gen toks'], model_1B_direct["pass_at_k"], color="red", marker="^", label="direct")
    plt.plot(model_3B_direct['Avg no gen toks'], model_3B_direct["pass_at_k"], color="blue", marker="^")
    plt.plot(model_8B_direct['Avg no gen toks'], model_8B_direct["pass_at_k"], color="purple", marker="^")
    for correctness_measure in ["pass_at_k", "logprobs_correctness"]:
        # Plot for each model size and method. color represents model size and line style represents method
        color = "red" # if correctness_measure == "pass_at_k" else "lightcoral"
        plt.plot(model_1B_sampling['Avg no gen toks'], model_1B_sampling[correctness_measure], color=color, marker=".", label='3.2 1B')
        plt.plot(model_1B_beam['Avg no gen toks'], model_1B_beam[correctness_measure], color=color, marker="X", label="beam")
        plt.plot(model_1B_cotdecoding['Avg no gen toks'], model_1B_cotdecoding[correctness_measure], color=color, marker='d', label="cotdecoding")

        color = "blue" # if correctness_measure == "pass_at_k" else "cornflowerblue"
        plt.plot(model_3B_sampling['Avg no gen toks'], model_3B_sampling[correctness_measure], color=color, marker=".", label="BON")
        plt.plot(model_3B_beam['Avg no gen toks'], model_3B_beam[correctness_measure], color=color, marker="X", label='3.2 3B')
        plt.plot(model_3B_cotdecoding['Avg no gen toks'], model_3B_cotdecoding[correctness_measure], color=color, marker='d', label="cotdecoding")

        color = "purple" # if correctness_measure == "pass_at_k" else "violet"
        plt.plot(model_8B_sampling['Avg no gen toks'], model_8B_sampling[correctness_measure], color=color, marker=".", label='3.1 8B')
        plt.plot(model_8B_beam['Avg no gen toks'], model_8B_beam[correctness_measure], color=color, marker="X")
        plt.plot(model_8B_cotdecoding['Avg no gen toks'], model_8B_cotdecoding[correctness_measure], color=color, marker='d', label="cotdecoding")

        # Adding labels and title
        plt.xlabel('Average No. of Generated Tokens')
        plt.ylabel(correctness_measure)
        plt.ylim(0, 1.2)
        plt.xlim(xmin=0)
        # plt.xlim(0, 3500)
        plt.title(f'{correctness_measure} vs. Avg No. of Generated Tokens ({domain}) {title_prefix}')

        # Create custom legends
        legend_element = [
            Line2D([0], [0], color='red', lw=2, label='3.2 1B pass@k'),
            Line2D([0], [0], color='blue', lw=2, label='3.2 3B pass@k'),
            Line2D([0], [0], color='purple', lw=2, label='3.1 8B pass@k'),
            Line2D([0], [0], color='white', lw=0, label=''),
            Line2D([0], [0], color='lightcoral', lw=2, label='3.2 1B logprob-ranked'),
            Line2D([0], [0], color='cornflowerblue', lw=2, label='3.2 3B logprob-ranked'),
            Line2D([0], [0], color='violet', lw=2, label='3.1 8B logprob-ranked'),
            Line2D([0], [0], color='white', lw=0, label=''),
            Line2D([0], [0], marker='.', color='w', markerfacecolor='black', markersize=8, label='sampling'),
            Line2D([0], [0], marker="X", color='w', markerfacecolor='black', markersize=8, label='beam'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, label='direct'),
            Line2D([0], [0], marker='d', color='w', markerfacecolor='black', markersize=8, label='cotdecoding')
        ]

        # Add legends for model and method
        if not title_prefix:
            plt.legend(handles=legend_element, loc='upper center', ncol=3, bbox_to_anchor=(0.25, 1.1), fancybox=True, shadow=True)
        # plt.legend(title='Scaling Test-time compute')
        plt.savefig(os.path.join(foldername, f"{domain}_{correctness_measure}{f'_{title_prefix}' if title_prefix else ''}.png"))
        plt.clf()

domains = sys.argv[1:]
N_buckets = [(0,5), (5, 10), (10, 15), (15, 20), (20, 25)]
L_buckets = [(0,3), (3, 6), (6, 9), (9, 12), (12, 15)]

for domain in domains:

    methodmodels = []
    avg_gen_toks = []
    pass_at_k = []
    logprobs_correctness = []

    # ind sample level
    # gen_toks = []
    # correctness = []
    # logprob_confidence = []

    if domain == "arrayworld":
        data_groups = {}
        for N in N_buckets:
            for L in L_buckets:
                data_groups[(N, L)] = {}
    if domain == "indexing":
        data_groups = {}
        for N in N_buckets:
            data_groups[N] = {}
    if domain == "idx_management":
        data_groups = {}
        for L in L_buckets:
            data_groups[L] = {}

    for outputfile in glob.glob(f"{foldername}/{domain}*.json"):
        parsed_filename = re.search(rf'{domain}_(.+)_(Llama.+)\.json', outputfile)
        if not parsed_filename: breakpoint()
        method = parsed_filename.group(1)
        model = parsed_filename.group(2)

        methodmodels.append(f"{model} - {method}")

        outputs = json.load(open(outputfile))
        logprobs_correctness.append(len([ex for ex in outputs if ex['generations'][ex["ranking"]["len_norm_logprobs"][0][0]]["correct"]]) / len(outputs))
        pass_at_k.append(len([ex for ex in outputs if any(gen["correct"] for gen in ex['generations'])]) / len(outputs))
        if "_beams" in method:
            num_beams = int(re.search(r'_beams(\d+)', method).group(1))
            avg_gen_toks.append(num_beams * sum(gen['generated_tokens'] for op in outputs for gen in op["generations"]) / len(outputs))
        else:
            avg_gen_toks.append(sum(gen['generated_tokens'] for op in outputs for gen in op["generations"]) / len(outputs))
            # gen_toks.extend([gen['generated_tokens'] for op in outputs for gen in op["generations"]])
            # logprob_confidence.extend([ex["ranking"]["len_norm_logprobs"][][1] gen['generated_tokens'] for op in outputs for gen in op["generations"]])

        for ex in outputs:
            if domain in {"arrayworld", "indexing"}:
                code_lines = ex['input_example']['code'].splitlines(keepends=True)
                array = re.search(r'array = (.+)\n', code_lines[0]).group(1)
                this_N = len(eval(array))
                for N in N_buckets:
                    if this_N < N[-1]: break
            if domain in {"arrayworld", "idx_management"}:
                code_lines = ex['input_example']['code'].splitlines(keepends=True)
                this_L = len(code_lines)
                if domain == "arrayworld": this_L -= 2  # -2 for init and end, doesnt make a huge difference
                for L in L_buckets:
                    if this_L < L[-1]: break

            if domain == "arrayworld":
                if model not in data_groups[(N, L)]: data_groups[(N, L)][model] = {}
                if method not in data_groups[(N, L)][model]: data_groups[(N, L)][model][method] = []
                data_groups[(N, L)][model][method].append(ex)
            if domain == "indexing":
                if model not in data_groups[N]: data_groups[N][model] = {}
                if method not in data_groups[N][model]: data_groups[N][model][method] = []
                data_groups[N][model][method].append(ex)
            if domain == "idx_management":
                if model not in data_groups[L]: data_groups[L][model] = {}
                if method not in data_groups[L][model]: data_groups[L][model][method] = []
                data_groups[L][model][method].append(ex)

    plot_all(methodmodels, avg_gen_toks, pass_at_k, logprobs_correctness, domain)

    for bucket_key, bucket_info in data_groups.items():
        if domain == "arrayworld":
            bucket_title_prefix = f"N={bucket_key[0]} L={bucket_key[1]}"
        elif domain == "indexing":
            bucket_title_prefix = f"N={bucket_key}"
        elif domain == "idx_management":
            bucket_title_prefix = f"L={bucket_key}"
        else: continue
        bucket_methodmodels = []
        bucket_avg_gen_toks = []
        bucket_pass_at_k = []
        bucket_logprobs_correctness = []
        for model_name, experiment_info in bucket_info.items():
            for method_name, outputs in experiment_info.items():
                bucket_methodmodels.append(f"{model_name} - {method_name}")
                bucket_logprobs_correctness.append(len([ex for ex in outputs if ex['generations'][ex["ranking"]["len_norm_logprobs"][0][0]]["correct"]]) / len(outputs))
                bucket_pass_at_k.append(len([ex for ex in outputs if any(gen["correct"] for gen in ex['generations'])]) / len(outputs))
                if "_beams" in method:
                    num_beams = int(re.search(r'_beams(\d+)', method).group(1))
                    bucket_avg_gen_toks.append(num_beams * sum(gen['generated_tokens'] for op in outputs for gen in op["generations"]) / len(outputs))
                else:
                    bucket_avg_gen_toks.append(sum(gen['generated_tokens'] for op in outputs for gen in op["generations"]) / len(outputs))
        plot_all(bucket_methodmodels, bucket_avg_gen_toks, bucket_pass_at_k, bucket_logprobs_correctness, domain, bucket_title_prefix)