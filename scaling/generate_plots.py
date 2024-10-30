import pandas as pd
import matplotlib.pyplot as plt
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

methodmodels = []
avg_gen_toks = []
correctness = []

domain = sys.argv[1]
for outputfile in glob.glob(f"outputs/{domain}*.json"):
    parsed_filename = re.search(r'.*_(fs_basic|fs_cot_temp\d\d_N\d+|fs_cot_beams\d+)_(.+)\.json', outputfile)
    if not parsed_filename: breakpoint()
    method = parsed_filename.group(1)
    model = parsed_filename.group(2)

    methodmodels.append(f"{model} - {method}")

    outputs = json.load(open(outputfile))
    if "_N" in method:
        correctness.append(len([ex for ex in outputs if any(gen["correct"] for gen in ex['generations'])]) / len(outputs))
        avg_gen_toks.append(sum(gen['generated_tokens'] for op in outputs for gen in op["generations"]) / len(outputs))
    elif "_beams" in method:
        correctness.append(len([ex for ex in outputs if ex['generations'][0]["correct"]]) / len(outputs))
        num_beams = int(re.search(r'_beams(\d+)', method).group(1))
        avg_gen_toks.append(num_beams * sum(op["generations"][0]['generated_tokens'] for op in outputs) / len(outputs))
    else:
        avg_gen_toks.append(sum(op['generated_tokens'] for op in outputs) / len(outputs))
        correctness.append(len([ex for ex in outputs if ex["correct"]]) / len(outputs))

# everything at once: each table is a domain
def plot_all():

    data = {
        "Method": methodmodels,
        "Avg no gen toks": avg_gen_toks,
        "Correctness": correctness
    }

    # Create a DataFrame for the new data
    df = pd.DataFrame(data)

    # Plotting the new data grouped by model size
    plt.figure(figsize=(10, 6))

    # Grouped by model size
    model_1B_best_of_N = df[(df['Method'].str.contains('3.2-1B')) & (df['Method'].str.contains('_N'))]
    model_3B_best_of_N = df[df['Method'].str.contains('3.2-3B') & df['Method'].str.contains('_N')]
    model_8B_best_of_N = df[df['Method'].str.contains('3.1-8B') & df['Method'].str.contains('_N')]
    model_1B_beam = df[df['Method'].str.contains('3.2-1B') & df['Method'].str.contains('_beam')]
    model_3B_beam = df[df['Method'].str.contains('3.2-3B') & df['Method'].str.contains('_beam')]
    model_8B_beam = df[df['Method'].str.contains('3.1-8B') & df['Method'].str.contains('_beam')]
    model_1B_direct = df[df['Method'].str.contains('3.2-1B') & df['Method'].str.contains('_basic')]
    model_3B_direct = df[df['Method'].str.contains('3.2-3B') & df['Method'].str.contains('_basic')]
    model_8B_direct = df[df['Method'].str.contains('3.1-8B') & df['Method'].str.contains('_basic')]

    model_1B = df[df['Method'].str.contains('3.2-1B')]
    model_3B = df[df['Method'].str.contains('3.2-3B')]
    model_8B = df[df['Method'].str.contains('3.1-8B')]

    # Plot for each model size and method. color represents model size and line style represents method
    plt.scatter(model_1B_best_of_N['Avg no gen toks'], model_1B_best_of_N['Correctness'], color='red', marker=".", label='3.2 1B')
    plt.scatter(model_1B_beam['Avg no gen toks'], model_1B_beam['Correctness'], color='red', marker="x")
    plt.scatter(model_1B_direct['Avg no gen toks'], model_1B_direct['Correctness'], color='red', marker="^")
    plt.plot(model_1B['Avg no gen toks'], model_1B['Correctness'], color='red')

    plt.scatter(model_3B_best_of_N['Avg no gen toks'], model_3B_best_of_N['Correctness'], color='blue', marker=".", label='3.2 3B')
    plt.scatter(model_3B_beam['Avg no gen toks'], model_3B_beam['Correctness'], color='blue', marker="x")
    plt.scatter(model_3B_direct['Avg no gen toks'], model_3B_direct['Correctness'], color='blue', marker="^")
    plt.plot(model_3B['Avg no gen toks'], model_3B['Correctness'], color='blue')

    plt.scatter(model_8B_best_of_N['Avg no gen toks'], model_8B_best_of_N['Correctness'], color='purple', marker=".", label='3.1 8B')
    plt.scatter(model_8B_beam['Avg no gen toks'], model_8B_beam['Correctness'], color='purple', marker="x")
    plt.scatter(model_8B_direct['Avg no gen toks'], model_8B_direct['Correctness'], color='purple', marker="^")
    plt.plot(model_8B['Avg no gen toks'], model_8B['Correctness'], color='purple')

    # Adding labels and title
    plt.xlabel('Average No. of Generated Tokens')
    plt.ylabel('Correctness')
    plt.ylim(0, 1)
    plt.title(f'Correctness vs. Avg No. of Generated Tokens ({domain})')
    plt.legend(title='Scaling Test-time compute')
    plt.show()

# grouped by model: each table is a domain and model, showing difference in methods
plot_all()

def plot_by_model():
    for (model, color) in [("3.2-1B", 'red'), ("3.2-3B", 'blue'), ("3.1-8B", 'purple')]:
        
        methods = [mm for mm in methodmodels if model in mm]
        relevant_avg_gen_toks = [avg_gen_toks[idx] for idx, mm in enumerate(methodmodels) if model in mm]
        relevant_correctness = [correctness[idx] for idx, mm in enumerate(methodmodels) if model in mm]
        if len(relevant_avg_gen_toks) == 0: continue
        data = {
            "Method": methods,
            "Avg no gen toks": relevant_avg_gen_toks,
            "Correctness": relevant_correctness
        }

        # Create a DataFrame for the new data
        df = pd.DataFrame(data)

        # Plotting the new data grouped by model size
        plt.figure(figsize=(10, 6))

        # Grouped by model size
        best_of_N = df[df['Method'].str.contains('_N')]
        beam = df[df['Method'].str.contains('_beam')]
        direct = df[df['Method'].str.contains('_basic')]

        # Plot for each model size and method. color represents model size and line style represents method
        plt.scatter(best_of_N['Avg no gen toks'], best_of_N['Correctness'], color=color, marker=".", label="best-of-N")
        plt.plot(best_of_N['Avg no gen toks'], best_of_N['Correctness'], color=color, linestyle='-')
        plt.scatter(beam['Avg no gen toks'], beam['Correctness'], color=color, marker="x", label="beam")
        plt.plot(beam['Avg no gen toks'], beam['Correctness'], color=color, linestyle='--')
        plt.scatter(direct['Avg no gen toks'], direct['Correctness'], color=color, marker="^", label="greedy direct")
        plt.plot(direct['Avg no gen toks'], direct['Correctness'], color=color, linestyle=':')

        # Adding labels and title
        plt.xlabel('Average No. of Generated Tokens')
        plt.ylabel('Correctness')
        plt.ylim(0, 1)
        plt.title(f'Correctness vs. Avg No. of Generated Tokens ({domain} {model})')
        plt.legend(title='Scaling Test-time compute')
        plt.show()
# plot_by_model()

# grouped by method: each table is a domain and method, showing difference in models
def plot_by_method():
    for (method, marker, linestyle) in [("fs_basic", "^", ':'), ("_N", ".", '-'), ("_beam", "x", '--')]:
        
        models = [mm for mm in methodmodels if method in mm]
        relevant_avg_gen_toks = [avg_gen_toks[idx] for idx, mm in enumerate(methodmodels) if method in mm]
        relevant_correctness = [correctness[idx] for idx, mm in enumerate(methodmodels) if method in mm]
        if len(relevant_avg_gen_toks) == 0: continue
        data = {
            "Model": models,
            "Avg no gen toks": relevant_avg_gen_toks,
            "Correctness": relevant_correctness
        }

        # Create a DataFrame for the new data
        df = pd.DataFrame(data)

        # Plotting the new data grouped by model size
        plt.figure(figsize=(10, 6))

        # Grouped by model size
        llama_1b = df[df['Model'].str.contains('3.2-1B')]
        llama_3b = df[df['Model'].str.contains('3.2-3B')]
        llama_8b = df[df['Model'].str.contains('3.1-8B')]

        # Plot for each model size and method. color represents model size and line style represents method
        plt.scatter(llama_1b['Avg no gen toks'], llama_1b['Correctness'], color='red', marker=marker, label="3.2 1B")
        plt.plot(llama_1b['Avg no gen toks'], llama_1b['Correctness'], color='red', linestyle=linestyle)
        plt.scatter(llama_3b['Avg no gen toks'], llama_3b['Correctness'], color='blue', marker=marker, label="3.2 3B")
        plt.plot(llama_3b['Avg no gen toks'], llama_3b['Correctness'], color='blue', linestyle=linestyle)
        plt.scatter(llama_8b['Avg no gen toks'], llama_8b['Correctness'], color='purple', marker=marker, label="3.1 8B")
        plt.plot(llama_8b['Avg no gen toks'], llama_8b['Correctness'], color='purple', linestyle=linestyle)

        # Adding labels and title
        plt.xlabel('Average No. of Generated Tokens')
        plt.ylabel('Correctness')
        plt.ylim(0, 1)
        plt.title(f'Correctness vs. Avg No. of Generated Tokens ({domain} {method})')
        plt.legend(title='Scaling Test-time compute')
        plt.show()
# plot_by_method()