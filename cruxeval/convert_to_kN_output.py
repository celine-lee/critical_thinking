import os
import json
import glob
import re

# Predefined bucket ranges for k and N
k_ranges = [(10, 25), (25, 50), (50, 120)]
N_ranges = [(1, 3), (3, 10), (10, 40)]

def find_bucket(value, ranges):
    """Finds the bucket that the value falls into and returns the bucket's midpoint."""
    for lower, upper in ranges:
        if lower <= value < upper:
            return (lower + upper) // 2  # Midpoint
    return None  # Should not happen if all values fit into ranges

# Process each JSON file in the `outputs_straightlined` directory
for experiment_file in glob.glob("outputs_straightlined/*.json"):
    # Extract model name from filename (assumes the format includes `_T`)
    match = re.search(r"([^\/]+)_T", experiment_file)
    if not match:
        continue  # Skip if filename doesn't match expected format
    modelname = match.group(1)

    sorted_ex = {}

    with open(experiment_file, "r") as f:
        results = json.load(f)
        for ex in results:
            # Keep the required fields
            new_ex = {
                "query": ex["query"],
                "model_generation": ex["model_generation"],
                "generated_tokens": ex["generated_tokens"],
                "pred_answer": ex["pred_answer"],
                "true_answer": ex["true_answer"],
                "correct": ex["correct"],
                "id": ex["id"],
            }

            # Extract original values
            k = ex["k"]
            N = ex["N"]

            # Find the bucket midpoints
            mid_k = find_bucket(k, k_ranges)
            mid_N = find_bucket(N, N_ranges)

            # Only store examples if they fall into a valid bucket
            if mid_k is not None and mid_N is not None:
                bucket_key = (mid_k, mid_N)
                if bucket_key not in sorted_ex:
                    sorted_ex[bucket_key] = []
                sorted_ex[bucket_key].append(new_ex)

    # Write bucketed examples into structured directories
    for (mid_k, mid_N), examples in sorted_ex.items():
        folder = os.path.join("outputs_straightlined", f"k{mid_k}_N{mid_N}")
        os.makedirs(folder, exist_ok=True)

        write_to_file = os.path.join(folder, f"{modelname}_T0.0.json")
        with open(write_to_file, "w") as out_f:
            json.dump(examples, out_f, indent=4)

print("Sorting and bucketing complete.")
