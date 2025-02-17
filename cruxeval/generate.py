import sys
import json
import random
import re
import ast
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
import sys
import ipdb
import traceback


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

fs_insn = """You will be given a desired function AST size between [TASK] and [/TASK] tags. Following the examples given, write a Python function of the given AST size and several varying test inputs for that function."""
fs_template = """[TASK]
ast_size = {ast_size}
[/TASK]
[PYTHON]
{fn_def}
[/PYTHON]
[TEST]
{test_cases}
[/TEST]"""
query_template = """[TASK]
ast_size = {ast_size}
[/TASK]
[PYTHON]
"""

def load_model(model_name, quantize=True):
    config = AutoConfig.from_pretrained(model_name)

    bnb_config = None
    if quantize and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    for param in model.parameters():
        param._requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", truncation_side="left"
    )

    return model, tokenizer


def count_ast_nodes(code):
    """Count the number of nodes in the AST of the given code."""
    try:
        tree = ast.parse(code)
        return sum(1 for _ in ast.walk(tree))
    except Exception:
        return None  # Return None if parsing fails

global lines
global line_counts

def trace_f(frame, event, arg):
    """Trace function to count line executions."""
    if event == 'call': return trace_f
    elif event == 'line':
        lineno = frame.f_lineno - 1
        if lineno not in {0, len(lines) - 1}: 
            current_line = lines[lineno]
            line_counts[lineno] = line_counts.get(lineno, 0) + 1
    return trace_f

def get_kN_info(example):
    global lines
    global line_counts
    code = example["code"]
    input_data = example["input"]
    uid = example['id']
    code_to_execute = code + f"\n\nanswer = f({input_data})"
    lines = code_to_execute.split("\n")
    global_env = {}
    local_env = {}
    line_counts = {}

    sys.settrace(trace_f)
    
    try:
        exec(code_to_execute, global_env, local_env)
        error_msg = None
    except Exception as e:
        error_msg = str(e)
    finally:
        sys.settrace(None)

    if error_msg:
        return None

    answer = repr(local_env['answer'])  # Preserve string formatting correctly
    return {
        "code": code,
        "input": input_data,
        "output": answer,
        "line_execution_counts": line_counts,
        "ast_size": count_ast_nodes(code),
        "error": error_msg,
        "id": uid,
    }

def map_to(value, ranges):
    """Map a value to a range."""
    for r in ranges:
        if r[0] <= value < r[1]:
            return r
    return ranges[-1]

def map_to(value, value_ranges):
    if value < min(value_ranges)[0]: return min(value_ranges)
    if value > max(value_ranges)[1]: return max(value_ranges)
    for value_range in value_range:
        if value in range(*value_range):
            return value_range
    return None

def create_examples(model_name, existing, k_ranges, N_ranges, num_ex_per=100):
    """Generates synthetic examples using VLLM."""
    model, tokenizer = load_model(model_name)
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 600,
        "stop_strings": ["[/TEST]"],
        "pad_token_id": tokenizer.eos_token_id,
        "num_return_sequences": 4,

    }
    
    dataset = {(kr, Nr): [] for kr in k_ranges for Nr in N_ranges}
    for ex in existing:
        k_range = map_to(ex["ast_size"], k_ranges)
        N_range = map_to(sum(ex["line_execution_counts"].values()), N_ranges)
        dataset[(k_range, N_range)].append(ex)
    
    uid_counter = len(existing)

    for kr_idx, kr in enumerate(k_ranges):
        if len(dataset[(kr, N_ranges[0])]) >= num_ex_per:
            continue
        
        all_kr_ex = [ex for Nr in N_ranges for ex in dataset[(kr, Nr)]]

        for Nr_idx, Nr in enumerate(N_ranges):
            while len(dataset[(kr, Nr)]) < num_ex_per:
                prompt = fs_insn + "\n\n"
                other_samples = random.sample(all_kr_ex, min(4, len(all_kr_ex)))

                fs_ex = []
                for other_ex in other_samples:
                    ast_size = other_ex["ast_size"]
                    fn_def = other_ex["code"]
                    test_cases = f"assert f({other_ex['input']}) == {other_ex['output']}"
                    fs_ex.append(fs_template.format(ast_size=ast_size, fn_def=fn_def, test_cases=test_cases))
                
                random.shuffle(fs_ex)
                prompt += "\n".join(fs_ex) + '\n' + query_template.format(ast_size=random.choice(range(Nr[0], Nr[1])))
                print(prompt)
                input_ids = self.tokenizer(
                    [prompt],
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                ).to(device)
                outputs = model.generate(input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask, **generation_config)

                for output in self.tokenizer.decode(outputs.sequences):
                    print(output)
                    generated_code_match = re.search(r'\[PYTHON\]\n(.*?)\[\/PYTHON\]', output.text, re.DOTALL)
                    generated_test_match = re.search(r'\[TEST\]\s*assert f\((.+?)\)\s*==', output.text)

                    if not generated_code_match or not generated_test_match:
                        continue

                    generated_code = generated_code_match.group(1).strip()
                    generated_test = generated_test_match.group(1).strip()

                    example = {
                        "code": generated_code,
                        "input": generated_test,
                        "id": f"new_synth_{uid_counter}"
                    }
                    
                    full_ex = get_kN_info(example)
                    if full_ex is None:
                        continue

                    new_ex_k = map_to(full_ex["ast_size"], k_ranges)
                    new_ex_N = map_to(sum(full_ex["line_execution_counts"].values()), N_ranges)

                    if (new_ex_k, new_ex_N) not in dataset:
                        continue

                    dataset[(new_ex_k, new_ex_N)].append(full_ex)
                    uid_counter += 1

    return dataset

def make_straightline(file_to_straightline):
    straightlined_dataset = []
    for ex in json.load(open(file_to_straightline)):
        code_lines = ex['code'].splitlines()
        helpful_parses = re.search(r'([\s\S]*)def f\((.*)\):\s*\n([\s\S]+)', ex['code'])
        starter_code = helpful_parses.group(1)
        function_arguments = helpful_parses.group(2)
        code = helpful_parses.group(3)
        starting_space = re.search(r'\s*', code[:code.find('\n')]).group(0)
        code_lines = [re.sub(r'return\s+(.*)', r'f = \1', cl[len(starting_space):]) for cl in code.splitlines()] # assuming that doesnt dedent...

        # get variables from fn header and initialize them with some value
        try:
            variable_matches = get_complete_paren(function_arguments)
        except: import pdb; pdb.set_trace()
        if variable_matches:
            varnames = [re.search(r'[a-zA-Z_][a-zA-Z_0-9]*', varmatch).group(0) for varmatch in variable_matches.split(',')]
            code_setup = ", ".join(varnames) + " = " + ex['input'] + '\n'
        else:
            code_setup = ""
        modified_code = code_setup + "\n".join(code_lines)

        straightlined_dataset.append(ex | {"straightlined_code": modified_code})

    with open(file_to_straightline.replace(".json", "_straightlined.json"), 'w') as wf:
        json.dump(straightlined_dataset, wf, indent=4)

if __name__ == "__main__":
    # Load existing dataset
    with open("cruxeval_profiled.json") as f:
        existing = json.load(f)

    k_ranges = [(0, 30), (30, 60), (60, 120)]
    N_ranges = [(0, 3), (3, 10), (10, 40)]
    model_name = "codellama/CodeLlama-34b-hf"

    dataset = create_examples(model_name, existing, k_ranges, N_ranges, num_ex_per=100)

    output_filename = "synth_cruxeval_profiled.json"
    with open(output_filename, 'w') as wf:
        json.dump([ex for ex_list in dataset.values() for ex in ex_list], wf, indent=4)

    make_straightline(output_filename)
