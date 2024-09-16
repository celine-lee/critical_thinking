import re
from tqdm import tqdm
import json
import ast
import subprocess
from contextlib import contextmanager
import signal

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

conditional_probe_str = """
{spaces_in_last}condval = bool({to_evaluate})
{spaces_in_last}print(condval)
{spaces_in_last}exit()
"""
vartype_probe_str = """
{spaces_in_last}varval = eval("{to_evaluate}")
{spaces_in_last}print(f"TYPE:<<{{type(varval).__name__}}>>")
{spaces_in_last}exit()
"""
def get_execution_run(code, start_index, end_index, probe_str):
    code_lines = code.rstrip().splitlines(keepends=True)
    prefix_lines = []
    for line in code_lines: 
        prefix_lines.append(line)
        if len(''.join(prefix_lines)) >= end_index: break 
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
                truth_value = get_execution_run(code, child_start_index, child_end_index, conditional_probe_str)
                if truth_value: 
                    # print(f"{code[child_start_index:child_end_index]} --> {truth_value}")
                    conditional_evaluation[(child_start_index, child_end_index)] = truth_value
            truth_value = get_execution_run(code, start_index, end_index, conditional_probe_str)
            if truth_value: conditional_evaluation[(start_index, end_index)] = truth_value
        elif isinstance(node, ast.If):
            child_start_index = sum(len(line) for line in code_lines[:node.test.lineno-1]) + node.test.col_offset
            child_end_index = sum(len(line) for line in code_lines[:node.test.end_lineno-1]) + node.test.end_col_offset
            truth_value = get_execution_run(code, child_start_index, child_end_index, conditional_probe_str)
            if truth_value: 
                # print(f"{code[child_start_index:child_end_index]} --> {truth_value}")
                conditional_evaluation[(child_start_index, child_end_index)] = truth_value

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
            extracted_types = get_execution_run(code, start_index, end_index, vartype_probe_str)
            if extracted_types: 
                this_var_type = re.search(r'TYPE:<<(.+?)>>', extracted_types)
                if this_var_type:
                    var_types[(start_index, end_index)] = this_var_type.group(1)
    return var_types

def get_conditionals_and_types_data():
    cot_examples = []
    examples = []

    ds = load_dataset("cruxeval-org/cruxeval")
    op_insn = """Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete. Do NOT output a description for the assert.

n = 17
f = n
assert f == 17

"""
    op_insn_cot = """You are given a function and an input. Complete the assertion with the output of executing the function on the input. First, reason step by step before arriving at an answer. Then, surround the answer as an assertion with [ANSWER] and [/ANSWER] tags.

s = "hi"
f = s + "a"
assert f == ??

The code takes a string s and produces the concatenation of s with the string "a", then assigns the result to f.
To determine the output of executing the code with s set to "hi", we need to concatenate "hi" with "a".

Therefore, the output set to f is "hia".

[ANSWER]assert f == "hia"[/ANSWER]

"""
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
        prompt = f"{op_insn}{modified_code}\nassert f == "
        cot_prompt = f"{op_insn_cot}{modified_code}\nassert f == ??\n\n"
        conditional_evaluation = get_boolean_truth_values(modified_code)
        types = get_types(modified_code)
        if len(types) == 0 and len(conditional_evaluation) == 0: continue
        examples.append((
            prompt, 
            modified_code,
            ex['output'], 
            {str((s_i+len(op_insn), e_i+len(op_insn))): truth_val for (s_i, e_i), truth_val in conditional_evaluation.items()},
            {str((s_i+len(op_insn), e_i+len(op_insn))): var_type for (s_i, e_i), var_type in types.items()},
            ))
        cot_examples.append((
            cot_prompt, 
            modified_code,
            ex['output'], 
            {str((s_i+len(op_insn_cot), e_i+len(op_insn_cot))): truth_val for (s_i, e_i), truth_val in conditional_evaluation.items()},
            {str((s_i+len(op_insn_cot), e_i+len(op_insn_cot))): var_type for (s_i, e_i), var_type in types.items()}
            ))

    with open("cot/data/cruxeval.json", "w") as wf:
        json.dump(examples, wf, indent=4)
    with open("cot/data/cruxeval_cot.json", "w") as wf:
        json.dump(cot_examples, wf, indent=4)
get_conditionals_and_types_data()

