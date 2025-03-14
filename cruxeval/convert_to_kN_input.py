import os
import json

# Predefined bucket ranges for k and N
k_ranges = [(10, 25), (25, 50), (50, 120)]
N_ranges = [(1, 3), (3, 10), (10, 40)]

# Load data
data_file = "synth_cruxeval_profiled_straightlined.json"
with open(data_file, "r") as f:
    examples = json.load(f)

def find_bucket(value, ranges):
    """Finds the bucket that the value falls into and returns the bucket's midpoint."""
    for lower, upper in ranges:
        if lower <= value < upper:
            return (lower + upper) // 2  # Midpoint
    return None  # Should not happen if all values fit into ranges

# Assign each example to its (k, N) bucket
sorted_ex = {}
for ex in examples:
    k = ex["ast_size"]
    N = sum(ex["line_execution_counts"].values())

    # Find the corresponding bucket midpoints
    mid_k = find_bucket(k, k_ranges)
    mid_N = find_bucket(N, N_ranges)

    if mid_k is not None and mid_N is not None:
        bucket_key = (mid_k, mid_N)
        if bucket_key not in sorted_ex:
            sorted_ex[bucket_key] = []
        sorted_ex[bucket_key].append(ex)

# Write out the bucketed examples to separate files
os.makedirs("synth_cruxeval_straightlined", exist_ok=True)
for (mid_k, mid_N), examples in sorted_ex.items():
    write_to_file = os.path.join("synth_cruxeval_straightlined", f"k{mid_k}_N{mid_N}.json")
    with open(write_to_file, "w") as out_f:
        json.dump(examples, out_f, indent=4)

# Print bucket assignments
print("Bucket assignments:")
for key in sorted_ex:
    print(key, "→", len(sorted_ex[key]), "examples")
