import os
import json
import re
from datasets import load_dataset

# Predefined bucket ranges for k (number of unique entities) and N (number of operations)
# These can be adjusted based on the distribution of k and N in the dataset.
k_ranges = [(2, 4), (4, 6), (6, 10), (10, 20)]
N_ranges = [(1, 3), (3, 5), (5, 10), (10, 20)]

def find_bucket(value, ranges):
    """Finds the bucket that a value falls into and returns the bucket's midpoint."""
    for lower, upper in ranges:
        if lower <= value < upper:
            return (lower + upper) // 2
    return None

def process_split(split_name, dataset):
    """
    Processes a split of the GSM8K dataset to extract k and N,
    then buckets and saves the examples.
    """
    processed_examples = []
    print(f"Processing '{split_name}' split...")
    for i, ex in enumerate(dataset[split_name]):
        answer_text = ex['answer']
        final_answer = re.search(r'####\s*([\d\.]+)', answer_text).group(1).strip()
        
        # Find all calculations enclosed in <<...>>
        expressions = re.findall(r'<<(.+?)>>', answer_text)
        
        if not expressions:
            continue

        operation_count = 0
        entities = set()
        
        for expr in expressions:
            # Count arithmetic operations for 'N'
            operation_count += len(re.findall(r'[+\-*/]', expr))
            
            # Find all numeric literals (integers and floats) for 'k'
            numbers = re.findall(r'\d+\.?\d*', expr)
            for num_str in numbers:
                try:
                    entities.add(float(num_str))
                except ValueError:
                    # This should not happen with the given regex, but it's safe to have.
                    print(f"Warning: Could not convert '{num_str}' to float in example {i}.")
                    pass
                
        k = len(entities)
        N = operation_count
        
        if k > 0 and N > 0:
            processed_ex = {
                'id': f'gsm8k_{split_name}_{i}',
                'question': ex['question'],
                'answer': final_answer,
                'k': k,
                'N': N
            }
            processed_examples.append(processed_ex)

    # Bucket the processed examples
    bucketed_examples = {}
    for ex in processed_examples:
        mid_k = find_bucket(ex['k'], k_ranges)
        mid_N = find_bucket(ex['N'], N_ranges)
        
        if mid_k is not None and mid_N is not None:
            bucket_key = (mid_k, mid_N)
            if bucket_key not in bucketed_examples:
                bucketed_examples[bucket_key] = []
            bucketed_examples[bucket_key].append(ex)

    # Write bucketed examples to separate JSON files
    output_dir = f"gsm8k/examples/{split_name}"
    os.makedirs(output_dir, exist_ok=True)
    for (mid_k, mid_N), examples in bucketed_examples.items():
        filename = os.path.join(output_dir, f"k{mid_k}_N{mid_N}.json")
        with open(filename, "w") as f:
            json.dump(examples, f, indent=4)

    print(f"Bucket assignments for '{split_name}':")
    for key in sorted(bucketed_examples.keys()):
        print(f"  k={key[0]}, N={key[1]} -> {len(bucketed_examples[key])} examples")
    print("-" * 20)

def main():
    """Main function to load dataset and process splits."""
    # Load the openai/gsm8k dataset
    print("Loading 'openai/gsm8k' dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    process_split('train', dataset)
    process_split('test', dataset)
    print("Processing complete.")

if __name__ == "__main__":
    main()