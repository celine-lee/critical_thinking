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

# Plot for each model size and method. color represents model size and line style represents method
plt.scatter(model_1B_best_of_N['Avg no gen toks'], model_1B_best_of_N['Correctness'], color='red', label='3.2 1B')
plt.plot(model_1B_best_of_N['Avg no gen toks'], model_1B_best_of_N['Correctness'], linestyle='-')
plt.scatter(model_3B_best_of_N['Avg no gen toks'], model_3B_best_of_N['Correctness'], color='blue', label='3.2 3B')
plt.plot(model_3B_best_of_N['Avg no gen toks'], model_3B_best_of_N['Correctness'], linestyle='-')
plt.scatter(model_8B_best_of_N['Avg no gen toks'], model_8B_best_of_N['Correctness'], color='purple', label='3.1 8B')
plt.plot(model_8B_best_of_N['Avg no gen toks'], model_8B_best_of_N['Correctness'], linestyle='-')

plt.scatter(model_1B_beam['Avg no gen toks'], model_1B_beam['Correctness'], color='red', label='3.2 1B')
plt.plot(model_1B_beam['Avg no gen toks'], model_1B_beam['Correctness'], linestyle='--')
plt.scatter(model_3B_beam['Avg no gen toks'], model_3B_beam['Correctness'], color='blue', label='3.2 3B')
plt.plot(model_3B_beam['Avg no gen toks'], model_3B_beam['Correctness'], linestyle='--')
plt.scatter(model_8B_beam['Avg no gen toks'], model_8B_beam['Correctness'], color='purple', label='3.1 8B')
plt.plot(model_8B_beam['Avg no gen toks'], model_8B_beam['Correctness'], linestyle='--')

plt.scatter(model_1B_direct['Avg no gen toks'], model_1B_direct['Correctness'], color='red', label='3.2 1B')
plt.plot(model_1B_direct['Avg no gen toks'], model_1B_direct['Correctness'], linestyle=':')
plt.scatter(model_3B_direct['Avg no gen toks'], model_3B_direct['Correctness'], color='blue', label='3.2 3B')
plt.plot(model_3B_direct['Avg no gen toks'], model_3B_direct['Correctness'], linestyle=':')
plt.scatter(model_8B_direct['Avg no gen toks'], model_8B_direct['Correctness'], color='purple', label='3.1 8B')
plt.plot(model_8B_direct['Avg no gen toks'], model_8B_direct['Correctness'], linestyle=':')

# Adding labels and title
plt.xlabel('Average No. of Generated Tokens')
plt.ylabel('Correctness')
plt.ylim(0, 1)
plt.title(f'Correctness vs. Avg No. of Generated Tokens ({domain})')
plt.legend(title='Scaling Test-time compute')
plt.show()

# grouped by model: each table is a domain and model, showing difference in methods
# grouped by method: each table is a domain and method, showing difference in models