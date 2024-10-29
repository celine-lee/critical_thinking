import re
from tqdm import tqdm
import os
import json
import ast
import subprocess
from contextlib import contextmanager
import signal
import random

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
    while len(''.join(code_lines[:line_idx])) <= start_index:
        line_idx += 1
    prefix_lines = code_lines[:line_idx-1] # get before line of entity to probe
    spaces_in_last = ""
    if len(prefix_lines):
        spaces_in_last = re.search(r'\s*', prefix_lines[-1]).group(0)
        pattern = r'^\s*(if|else|elif|for|while|def|class|try|except|finally|with)\b.*:\s*$'
        if re.match(pattern,  prefix_lines[-1]):
            spaces_in_last += '    ' 
    with open("DELETETHIS.py", 'w') as wf:
        wf.write(''.join(prefix_lines) + probe_str.format(spaces_in_last=spaces_in_last, to_evaluate=code[start_index:end_index]))
    with time_limit(5):
        try:
            collected_print = subprocess.check_output(['python', 'DELETETHIS.py'], timeout=15).decode('utf-8')
            if collected_print.strip() == '': return None
            return collected_print
        except:
            # Tends to happen when variable in listcomp or lambda fn or something or timeout exception
            return None
    return None



def parse_arrayworld_tree(code, force_static_array=True):
    idx_values = {} 
    conditional_valuations = {}
    code_lines = code.splitlines(keepends=True)
    end_of_first_line = len(code_lines[0])
    first_idx_initialization = re.search(r'idx\s*=', code)
    if first_idx_initialization is None: return None
    first_idx_initialization = first_idx_initialization.start()
    try: 
        tree = ast.parse(code)
    except: return None
    for node in ast.walk(tree):
        if 'lineno' not in dir(node): continue
        start_index = sum(len(line) for line in code_lines[:node.lineno-1]) + node.col_offset
        end_index = sum(len(line) for line in code_lines[:node.end_lineno-1]) + node.end_col_offset
        if start_index < end_of_first_line: continue
        if (start_index, end_index) in conditional_valuations: continue
        if (start_index, end_index) in idx_values: continue
        if isinstance(node, ast.For) or isinstance(node, ast.While) or isinstance(node, ast.FunctionDef) or isinstance(node, ast.Import):
            return None

        if isinstance(node, ast.Assign):
            for child in node.targets:
                if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name) and child.value.id == 'array': 
                    if force_static_array: 
                        return None
                if isinstance(child, ast.Name) and child.id == 'idx': # is is format "idx = ..." then dont probe this.
                    node.is_assigned_to = True

        elif isinstance(node, ast.AugAssign):
            child = node.target
            if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name) and child.value.id == 'array': 
                if force_static_array: 
                    return None
            
        # if using the idx variable, get its value
        elif isinstance(node, ast.Name):
            if node.id == 'idx': 
                if "is_assigned_to" in dir(node): continue 
                if start_index <= first_idx_initialization: continue
                var_value = probe_execution(code, start_index, end_index, varval_probe_str)
                if var_value is None: 
                    return None
                regex_str = rf"{code[start_index:end_index]}:([\s\S]+)"
                var_value = re.search(regex_str, var_value)
                if var_value is None:
                    breakpoint()
                var_value = var_value.group(1).strip()
                if not (var_value.isdigit() or var_value.lstrip('-').isdigit()): 
                    return None
                idx_values[(start_index, end_index)] = var_value
        elif isinstance(node, ast.BoolOp):
            for child in node.values:
                child_start_index = sum(len(line) for line in code_lines[:child.lineno-1]) + child.col_offset
                child_end_index = sum(len(line) for line in code_lines[:child.end_lineno-1]) + child.end_col_offset
                truth_value = probe_execution(code, child_start_index, child_end_index, conditional_probe_str)
                if truth_value is None: 
                    continue
                conditional_valuations[(child_start_index, child_end_index)] = re.search(r'truthval:(.+)', truth_value).group(1).strip()
            truth_value = probe_execution(code, start_index, end_index, conditional_probe_str)
            if truth_value is None: 
                continue
            conditional_valuations[(start_index, end_index)] = re.search(r'truthval:(.+)', truth_value).group(1).strip()
        elif isinstance(node, ast.If):
            child_start_index = sum(len(line) for line in code_lines[:node.test.lineno-1]) + node.test.col_offset
            child_end_index = sum(len(line) for line in code_lines[:node.test.end_lineno-1]) + node.test.end_col_offset
            truth_value = probe_execution(code, child_start_index, child_end_index, conditional_probe_str)
            if truth_value is None: 
                continue
            conditional_valuations[(child_start_index, child_end_index)] = re.search(r'truthval:(.+)', truth_value).group(1).strip()
        elif isinstance(node, ast.Compare):
            truth_value = probe_execution(code, start_index, end_index, conditional_probe_str)
            if truth_value is None: 
                continue
            conditional_valuations[(start_index, end_index)] = re.search(r'truthval:(.+)', truth_value).group(1).strip()
    return idx_values, conditional_valuations

def generate_arrayworld_data(model_name, num_examples, N=20):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    modelname = re.search(r'/(.+)', model_name).group(1)

    icl_fns = list(array_world_examples)

    datapoints = []
    if os.path.exists("data/array_world.json"): datapoints = json.load(open("data/array_world.json"))

    while len(datapoints) < num_examples:

        # prompt model to make candidates
        prompt = arrayworld_generation_insn.format(N=N) + "\n\n".join(f"```\n{ex_code.strip()}\n```" for ex_code in random.sample(icl_fns, 3)) + "\n\n```\n"
        input_ids = tokenizer(prompt, return_tensors='pt')
        generations = model.generate(**input_ids, tokenizer=tokenizer, return_dict_in_generate=True, num_return_sequences=4, min_new_tokens=30, max_length = 1024, stop_strings=["```"], do_sample=True, temperature=0.9)
        for generated_code in tokenizer.batch_decode(generations.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True):
            code = re.search(r'([\s\S]+?)```', generated_code)
            if code is None: continue
            code = code.group(1).strip()

            # filter based on simple heuristics
            if len(code) > 800: continue # CRUXeval does 75 < code < 300
            if len(code) < 100: continue
            if not code.splitlines(keepends=True)[-1].strip().startswith("assert answer == "): continue
            if "input(" in code: continue # no user interaction
            if re.search(r'\d+\.\d+', code): continue # no floating points
            if "random" in code: continue # we want determinism
            code_no_assert = ''.join(code.splitlines(keepends=True)[:-1]).strip()

            # try to execute candidate code
            code_output = execute_code(code_no_assert + "\nprint(f'\"{answer}\"' if isinstance(answer, str) else answer)")
            if code_output is None: continue
            code_output = code_output.strip()

            # get value of idx throughout the program
            parsed_arrayworld = parse_arrayworld_tree(code_no_assert)
            if parsed_arrayworld is None: continue 
            (idx_values, conditional_values) = parsed_arrayworld

            # get original model solution
            model_solution = re.search(r'assert answer == ([\s\S]*)', code.splitlines()[-1]).group(1)
            code_line = f"answer = {model_solution}\nprint(answer[0] if isinstance(answer, tuple) else answer)"
            model_solution = execute_code(code_line)

            print("+============= CODE ===============")
            print(code)
            print('-------- world states --------')
            for idxes, value in sorted({**idx_values, **conditional_values}.items(), key=lambda it: it[1], reverse=True):
                print(f"{idxes}: {code[idxes[0]:idxes[1]]} --> {value}")

            datapoints.append({
                "code": code,
                "idx_values": {f"({s_i}, {e_i})": i_v for (s_i, e_i), i_v in idx_values.items()},
                "conditional_values": {f"({s_i}, {e_i})": tv for (s_i, e_i), tv in conditional_values.items()},
                "true_answer": code_output,
                f"{model_name}_answer": model_solution.strip() if model_solution else model_solution,
            })
            icl_fns.append(f"{code_no_assert}\nassert answer == {code_output}")

            with open(f"data/arrayworld_{modelname}_N{N}.json", 'w') as wf: json.dump(datapoints, wf, indent=4)

    return datapoints
# generate_arrayworld_data("meta-llama/Meta-Llama-3.1-8B-Instruct", 300, N=20)

def make_arrayworld_data_uniform(filename=f"data/arrayworld_Meta-Llama-3.1-8B-Instruct_N20.json"):
    # TODO think about whether this is unfair to be reusing programs
    data = json.load(open(filename))
    idx_val_to_exs = {}
    for ex in data:
        code_lines = ex['code'].splitlines(keepends=True)
        array = re.search(r'array = (.+)', code_lines[0]).group(1)
        array_length = len(eval(array))
        last_idx_line = len(code_lines) - 1
        while not code_lines[last_idx_line].lstrip().startswith("answer = array[idx]"):
            last_idx_line -= 1
            if last_idx_line == -1: break
        if last_idx_line == -1: continue

        idx_values = ex['idx_values']
        for si_ei, answer_idx_val in idx_values.items():
            parsed_idxes = re.search(r'(\d+), (\d+)', si_ei)
            s_i = int(parsed_idxes.group(1))
            e_i = int(parsed_idxes.group(2))
            if sum(len(line) for line in code_lines[:last_idx_line]) <= s_i and sum(len(line) for line in code_lines[:last_idx_line+1]) >= e_i:
                break
        answer_idx_val = int(answer_idx_val)       
        if answer_idx_val < 0: answer_idx_val = array_length + answer_idx_val
        if answer_idx_val not in idx_val_to_exs: idx_val_to_exs[answer_idx_val] = []
        idx_val_to_exs[answer_idx_val].append((ex, si_ei, last_idx_line))

    print("All lengths: ", {idx_val: len(exs) for idx_val, exs in idx_val_to_exs.items()})
    num_per_idx_val = max(len(exs) for exs in idx_val_to_exs.values())
    print("Make them all: ", num_per_idx_val)
    for idx_val in idx_val_to_exs:
        while len(idx_val_to_exs[idx_val]) < num_per_idx_val:
            # get another program and modify it to what we want.
            while True:
                random_other_idx_val = random.choice(list(idx_val_to_exs.keys()))
                (random_other_program, other_si_ei, answer_idx_val) = random.choice(idx_val_to_exs[random_other_idx_val])
                if other_si_ei is None: continue
                break
            other_answer_idx_val = random_other_program['idx_values'][other_si_ei]
            diff = int(idx_val) -  int(other_answer_idx_val)
            new_line = f"idx = idx + {diff}\n" if diff > 0 else f"idx = idx - {-1 * diff}\n"
            old_code_lines = random_other_program['code'].splitlines(keepends=True)
            new_code = ''.join(old_code_lines[:answer_idx_val] + [new_line] + old_code_lines[answer_idx_val:])

            # collect information from new program
            code_no_assert = ''.join(new_code.splitlines(keepends=True)[:-1]).strip()

            # try to execute candidate code
            code_output = execute_code(code_no_assert + "\nprint(f'\"{answer}\"' if isinstance(answer, str) else answer)")
            if code_output is None: continue
            code_output = code_output.strip()
            new_code = f"{code_no_assert}\nassert answer == {code_output}"

            # get value of idx throughout the program
            parsed_arrayworld = parse_arrayworld_tree(code_no_assert)
            if parsed_arrayworld is None: continue 
            (idx_values, conditional_values) = parsed_arrayworld

            idx_val_to_exs[idx_val].append(({
                "code": new_code,
                "idx_values": {f"({s_i}, {e_i})": i_v for (s_i, e_i), i_v in idx_values.items()},
                "conditional_values": {f"({s_i}, {e_i})": tv for (s_i, e_i), tv in conditional_values.items()},
                "true_answer": code_output,
            }, None, None))


    all_examples = [ex[0] for exs in idx_val_to_exs.values() for ex in exs]
    random.shuffle(all_examples)
    with open(filename, "w") as wf:
        json.dump(all_examples, wf, indent=4)

# make_arrayworld_data_uniform()


def make_indexing_from_arrayworld(input_filename="uniformed_arrayworld_N20.json", output_filename="indexing_array_N20.json"):
    data = json.load(open(input_filename))
    new_data = []
    for ex in data:
        code_lines = ex['code'].splitlines(keepends=True)
        array = re.search(r'array = (.+)', code_lines[0]).group(1)
        last_idx_line = len(code_lines) - 1
        while not code_lines[last_idx_line].lstrip().startswith("answer = array[idx]"):
            last_idx_line -= 1
            if last_idx_line == -1: break
        if last_idx_line == -1: continue

        idx_values = ex['idx_values']
        for si_ei, answer_idx_val in idx_values.items():
            parsed_idxes = re.search(r'(\d+), (\d+)', si_ei)
            s_i = int(parsed_idxes.group(1))
            e_i = int(parsed_idxes.group(2))
            if sum(len(line) for line in code_lines[:last_idx_line]) <= s_i and sum(len(line) for line in code_lines[:last_idx_line+1]) >= e_i:
                break
        new_data.append({
            "code": f"array = {array}\nidx = {answer_idx_val}\nanswer = array[idx]\n{code_lines[-1].strip()}", 
            "true_answer": ex['true_answer']
        }) 


    with open(output_filename, "w") as wf:
        json.dump(new_data, wf, indent=4)

# make_indexing_from_arrayworld()

def make_idx_management(input_filename="uniformed_arrayworld_N{N}.json", output_filename="idx_management_N{N}.json", N=20):
    data = json.load(open(input_filename.format(N=N)))
    idx_val_to_exs = {}
    for ex in data:
        code_lines = ex['code'].splitlines(keepends=True)

        use_array_lines = set(line_idx for line_idx, line in enumerate(code_lines) if re.search(r'array', line))
        new_code_no_assert = "".join(cl for l_idx, cl in enumerate(code_lines) if l_idx not in use_array_lines|{0,len(code_lines)-1}).strip()

        # try to execute candidate code
        code_output = execute_code(new_code_no_assert + "\nprint(idx)")
        if code_output is None: 
            new_code_no_assert = f"idx = {random.randint(0, N)}\n" + new_code_no_assert
            code_output = execute_code(new_code_no_assert + "\nprint(idx)")
        if code_output is None: continue
            
        code_output = code_output.strip()
        new_code = f"{new_code_no_assert}\nassert idx == {code_output}"

        final_idx_val = int(code_output)
        if final_idx_val not in idx_val_to_exs: idx_val_to_exs[final_idx_val] = []
        idx_val_to_exs[final_idx_val].append({
            "code": new_code,
            "true_answer": code_output
        })
        

    print("All lengths: ", {idx_val: len(exs) for idx_val, exs in idx_val_to_exs.items()})
    num_per_idx_val = max(len(exs) for idx_val, exs in idx_val_to_exs.items() if idx_val > -5 and idx_val < N)
    print("Make them all: ", num_per_idx_val)
    for idx_val in idx_val_to_exs:
        while len(idx_val_to_exs[idx_val]) < num_per_idx_val:
            # get another program and modify it to what we want.
            random_other_idx_val = random.choice(list(idx_val_to_exs.keys()))
            random_other_program = random.choice(idx_val_to_exs[random_other_idx_val])
            try: 
                other_answer_idx_val = re.search(r'assert idx == (-?\d+)', random_other_program['code']).group(1)
            except:
                breakpoint()
            diff = int(idx_val) -  int(other_answer_idx_val)
            new_line = f"idx = idx + {diff}\n" if diff > 0 else f"idx = idx - {-1 * diff}\n"
            old_code_lines = random_other_program['code'].splitlines(keepends=True)
            new_code = ''.join(old_code_lines[:-1] + [new_line] + old_code_lines[-1:])

            # collect information from new program
            code_no_assert = ''.join(new_code.splitlines(keepends=True)[:-1]).strip()

            # try to execute candidate code
            code_output = execute_code(code_no_assert + "\nprint(idx)")
            if code_output is None: continue
            code_output = code_output.strip()
            new_code = f"{code_no_assert}\nassert idx == {code_output}"

            idx_val_to_exs[idx_val].append({
                "code": new_code,
                "true_answer": code_output,
            })


    all_examples = [ex for exs in idx_val_to_exs.values() for ex in exs]
    random.shuffle(all_examples)
    with open(output_filename.format(N=N), "w") as wf:
        json.dump(all_examples, wf, indent=4)

make_idx_management()