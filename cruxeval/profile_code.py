import sys
import ast
import json
import traceback
import ipdb
from datasets import load_dataset

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

# Load dataset
dataset = load_dataset("cruxeval-org/cruxeval")

def count_ast_nodes(code):
    """Count the number of nodes in the AST of the given code."""
    try:
        tree = ast.parse(code)
        return sum(1 for _ in ast.walk(tree))
    except Exception:
        return None  # Return None if parsing fails

global lines

def trace_f(frame, event, arg):
    """Trace function to count line executions."""
    if event == 'call': return trace_f
    # if frame.f_code.co_name != "f":
    #     return
    elif event == 'line':
        lineno = frame.f_lineno - 1
        if lineno not in {0, len(lines) - 1}: 
            current_line = lines[lineno ]
            line_counts[lineno] = line_counts.get(lineno, 0) + 1
    return trace_f

results = []

# Iterate through examples
for example in dataset["test"]: 
    code = example["code"]
    input_data = example["input"]
    uid = example['id']
    code_to_execute = code + f"\n\nanswer = f({input_data})"
    lines = code_to_execute.split("\n")
    # Prepare environment
    global_env = {}
    local_env = {}
    
    # Reset line counts
    line_counts = {}

    # Set trace
    sys.settrace(trace_f)
    
    try:
        # Execute the code with input safely
        exec(code_to_execute, global_env, local_env)
    except Exception as e:
        error_msg = str(e)
    else:
        error_msg = None
    finally:
        sys.settrace(None)  # Always disable trace after execution

    if error_msg: continue

    if isinstance(local_env['answer'], str):
        answer = f"'{local_env['answer']}'"
    else:
        answer = str(local_env['answer'])
    # Collect results
    result = {
        "code": code,
        "input": input_data,
        "output": answer,
        "line_execution_counts": line_counts,
        "ast_size": count_ast_nodes(code),
        "error": error_msg,
        "id": uid,
    }
    results.append(result)

# Save results as JSON
with open("cruxeval_profiled.json", "w") as f:
    json.dump(results, f, indent=4)

print("Profiling completed. Results saved in 'cruxeval_profiled.json'.")
