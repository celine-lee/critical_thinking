import re
from tqdm import tqdm
import os
import json
import ast
import subprocess
from contextlib import contextmanager
import signal

from prompts import *
from datasets import load_dataset

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException('Timeout')
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def execute_code(code):
    with open("DELETETHIS.py", 'w') as wf:
        wf.write(code)
    with time_limit(5):
        try:
            collected_print = subprocess.check_output(['python', 'DELETETHIS.py'], timeout=15).decode('utf-8')
            return collected_print
        except:
            return None
    return None


def probe_execution(code, start_index, end_index, probe_str):
    code_lines = code.rstrip().splitlines(keepends=True)
    line_idx = 0
    while len(''.join(code_lines[:line_idx])) < start_index:
        line_idx += 1
    prefix_lines = code_lines[:line_idx-1] # get before line of entity to probe
    spaces_in_last = ""
    if len(prefix_lines):
        spaces_in_last = re.search(r'\s*', prefix_lines[-1]).group(0)
        if (prefix_lines[-1].strip()[:4] in {"for ", "for(", "def "} or prefix_lines[-1].strip()[:3] in {"if ", "if("} or prefix_lines[-1].strip()[:6] in {"while ", "while("} or prefix_lines[-1].strip()[:5] in {"else ", "else:", "elif ", "elif("}) and (prefix_lines[-1].strip()[-1] == ":"):
            spaces_in_last += '    ' 
    with open("DELETETHIS.py", 'w') as wf:
        wf.write(''.join(prefix_lines) + probe_str.format(spaces_in_last=spaces_in_last, to_evaluate=code[start_index:end_index]))
    with time_limit(5):
        try:
            collected_print = subprocess.check_output(['python', 'DELETETHIS.py'], timeout=15).decode('utf-8')
            return collected_print
        except:
            # Tends to happen when variable in listcomp or lambda fn or something or timeout exception
            return None
    return None

def get_boolean_truth_values(code):
    conditional_evaluation = {}
    code_lines = code.splitlines(keepends=True)
    try: 
        tree = ast.parse(code)
    except: import pdb; pdb.set_trace()
    loop_indices = []
    for node in ast.walk(tree):
        if 'lineno' not in dir(node): continue
        start_index = sum(len(line) for line in code_lines[:node.lineno-1]) + node.col_offset
        end_index = sum(len(line) for line in code_lines[:node.end_lineno-1]) + node.end_col_offset
        if (start_index, end_index) in conditional_evaluation: continue
        if any(start_index >= other_s_i and end_index <= other_e_i for (other_s_i, other_e_i) in loop_indices): continue
        if isinstance(node, ast.For) or isinstance(node, ast.While):
            loop_indices.append((start_index, end_index))
        elif isinstance(node, ast.BoolOp):
            for child in node.values:
                child_start_index = sum(len(line) for line in code_lines[:child.lineno-1]) + child.col_offset
                child_end_index = sum(len(line) for line in code_lines[:child.end_lineno-1]) + child.end_col_offset
                truth_value = probe_execution(code, child_start_index, child_end_index, conditional_probe_str)
                if truth_value: 
                    # print(f"{code[child_start_index:child_end_index]} --> {truth_value}")
                    conditional_evaluation[(child_start_index, child_end_index)] = truth_value
            truth_value = probe_execution(code, start_index, end_index, conditional_probe_str)
            if truth_value: conditional_evaluation[(start_index, end_index)] = truth_value
        elif isinstance(node, ast.If):
            child_start_index = sum(len(line) for line in code_lines[:node.test.lineno-1]) + node.test.col_offset
            child_end_index = sum(len(line) for line in code_lines[:node.test.end_lineno-1]) + node.test.end_col_offset
            truth_value = probe_execution(code, child_start_index, child_end_index, conditional_probe_str)
            if truth_value: 
                # print(f"{code[child_start_index:child_end_index]} --> {truth_value}")
                conditional_evaluation[(child_start_index, child_end_index)] = truth_value
        elif isinstance(node, ast.Compare):
            truth_value = probe_execution(code, start_index, end_index, conditional_probe_str)
            if truth_value: 
                # print(f"{code[child_start_index:child_end_index]} --> {truth_value}")
                conditional_evaluation[(start_index, end_index)] = truth_value

    return conditional_evaluation


def get_types(code):
    var_types = {}
    code_lines = code.splitlines(keepends=True)
    try: 
        tree = ast.parse(code)
    except: import pdb; pdb.set_trace()
    for node in ast.walk(tree):
        if 'lineno' not in dir(node): continue
        start_index = sum(len(line) for line in code_lines[:node.lineno-1]) + node.col_offset
        end_index = sum(len(line) for line in code_lines[:node.end_lineno-1]) + node.end_col_offset
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load): continue
            extracted_types = probe_execution(code, start_index, end_index, vartype_probe_str)
            if extracted_types: 
                this_var_type = re.search(r'TYPE:<<(.+?)>>', extracted_types)
                if this_var_type:
                    var_types[(start_index, end_index)] = this_var_type.group(1)
    return var_types

##########################################################################################################################################


def get_complete_paren(code_str):
    if code_str.count("(") == 0 and code_str.count(")") == 0: return code_str
    cursor = 0
    open_p = 1
    while cursor < len(code_str):
        if code_str[cursor] in {'"', "'"}:
            cursor += code_str[cursor+1:].find(code_str[cursor]) + 1 + 1
            continue
        if code_str[cursor] == "(": open_p += 1
        if code_str[cursor] == ")": 
            open_p -= 1
            if open_p == 0:
                return code_str[:cursor]
        cursor += 1
    return code_str

def get_conditionals_and_types_data():
    examples = []

    ds = load_dataset("cruxeval-org/cruxeval")
    for ex in tqdm(ds['test']):
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
        # prompt = f"{op_insn}{modified_code}\nassert f == "
        # cot_prompt = f"{op_insn_cot}{modified_code}\nassert f == ??\n\n"
        conditional_evaluation = get_boolean_truth_values(modified_code)
        types = get_types(modified_code)
        if len(types) == 0 and len(conditional_evaluation) == 0: continue
        examples.append((
            modified_code,
            ex['output'], 
            {f"({s_i}, {e_i})": tv for (s_i, e_i), tv in conditional_evaluation.items()},
            {f"({s_i}, {e_i})": tv for (s_i, e_i), tv in types.items()},
            ))

    with open("cot/data/cruxeval.json", "w") as wf:
        json.dump(examples, wf, indent=4)
# get_conditionals_and_types_data()




##########################################################################################################################################

from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_logic_data(model_name, num_examples):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    icl_fns = list(fs_code)

    datapoints = []
    if os.path.exists("data/logic.json"): datapoints = json.load(open("data/logic.json"))

    num_tvs = 0
    while num_tvs < num_examples:
        successful_generations = []

        # prompt model to make candidates
        prompt = code_generation_insn + "\n\n".join(random.sample(icl_fns, 3)) + "\n\n```\n"
        input_ids = tokenizer(prompt, return_tensors='pt')
        generations = model.generate(**input_ids, tokenizer=tokenizer, return_dict_in_generate=True, num_return_sequences=4, max_length = 1024, stop_strings=["```"], do_sample=True, temperature=0.9)
        for generated_code in tokenizer.batch_decode(generations.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True):
            code = re.search(r'([\s\S]+?)```', generated_code)
            if code is None: continue
            code = code.group(1).strip()

            # filter based on simple heuristics
            if len(code) > 300: continue # CRUXeval does 75 < code < 300
            if len(code) < 75: continue
            if not code.splitlines(keepends=True)[-1].strip().startswith("assert answer == "): continue
            if "input(" in code: continue # no user interaction
            if re.search(r'\d+\.\d+', code): continue # no floating points
            if "random" in code: continue # we want determinism

            # get original model solution
            model_solution = re.search(r'assert answer == ([\s\S]*)', code.splitlines()[-1]).group(1)
            code_line = f"answer = {model_solution}\nprint(answer[0] if isinstance(answer, tuple) else answer)"
            model_solution = execute_code(code_line)
            
            # try to execute candidate code
            code = ''.join(code.splitlines(keepends=True)[:-1]).strip()
            code_output = execute_code(code + "\nprint(f'\"{answer}\"' if isinstance(answer, str) else answer)")
            if code_output is None: continue
            code_output = code_output.strip()
            icl_fns.append(f"```\n{code}\nassert answer == {code_output}\n```")
            conditionals = {f"({s_i}, {e_i})": tv for (s_i, e_i), tv in get_boolean_truth_values(code + "\nassert answer == \"dummy\"").items()}
            num_tvs += len(conditionals)

            datapoints.append({
                "code": code,
                "truth_states": conditionals,
                "true_answer": code_output,
                f"{model_name}_answer": model_solution.strip() if model_solution else model_solution,
            })

        with open("data/logic.json", 'w') as wf: json.dump(datapoints, wf, indent=4)

    return datapoints
generate_logic_data("meta-llama/Meta-Llama-3.1-8B-Instruct", 300)