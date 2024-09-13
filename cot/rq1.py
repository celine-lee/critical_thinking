import re
from tqdm import tqdm
import json
import ast
import subprocess
import sys
sys.path.append("../src")
sys.path.append(".")
from src.probe import LinearProbe

from datasets import load_dataset

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
def get_execution_run(code, conditional_start_index, conditional_end_index):
    code_lines = code.rstrip().splitlines(keepends=True)
    prefix_lines = []
    for line in code_lines: 
        if len(''.join(prefix_lines)) + len(line) < conditional_start_index: 
            prefix_lines.append(line)
    spaces_in_last = re.search(r'\s*', prefix_lines[-1]).group(0)
    if (prefix_lines[-1].strip()[:4] in {"for ", "for(", "def "} or prefix_lines[-1].strip()[:3] in {"if ", "if("} or prefix_lines[-1].strip()[:6] in {"while ", "while("} or prefix_lines[-1].strip()[:5] in {"else ", "else:", "elif ", "elif("}) and (prefix_lines[-1].strip()[-1] == ":"):
        spaces_in_last += '    ' 
    with open("DELETETHIS.py", 'w') as wf:
        wf.write(''.join(prefix_lines) + conditional_probe_str.format(spaces_in_last=spaces_in_last, to_evaluate=code[conditional_start_index:conditional_end_index]))
    try:
        collected_print = subprocess.check_output(['python', 'DELETETHIS.py'], timeout=15).decode('utf-8')
        return collected_print
    except SyntaxError:
        return None
    except:
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

                truth_value = get_execution_run(code, child_start_index, child_end_index)
                if truth_value: conditional_evaluation[(child_start_index, child_end_index)] = truth_value
            truth_value = get_execution_run(code, start_index, end_index)
            if truth_value: conditional_evaluation[(start_index, end_index)] = truth_value
        elif isinstance(node, ast.If):
            child_start_index = sum(len(line) for line in code_lines[:node.test.lineno-1]) + node.test.col_offset
            child_end_index = sum(len(line) for line in code_lines[:node.test.end_lineno-1]) + node.test.end_col_offset
            truth_value = get_execution_run(code, child_start_index, child_end_index)
            if truth_value: conditional_evaluation[(child_start_index, child_end_index)] = truth_value

    return conditional_evaluation

cot_examples = []
examples = []

ds = load_dataset("cruxeval-org/cruxeval")
op_insn = """Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete. Do NOT output a description for the assert.

def f(n):
return n
assert f(17) == 17

"""
op_insn_cot = """You are given a function and an input. Complete the assertion with the output of executing the function on the input. First, reason step by step before arriving at an answer. Then, surround the answer as an assertion with [ANSWER] and [/ANSWER] tags.

def f(s):
return s + "a"
assert f("hi") == ??

The function f takes a string s as input and returns the concatenation of s with the string "a".
To determine the output of executing the function f on the input "hi", we need to concatenate "hi" with "a".

Therefore, the output of executing the function f on the input "hi" is "hia".

[ANSWER]assert f("hi") == "hia"[/ANSWER]

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
    conditional_evaluation = get_boolean_truth_values(modified_code)
    if not conditional_evaluation: continue
    # print(f"===\n{modified_code}\n")
    # for (s_i, e_i), truth_val in conditional_evaluation.items():
    #     print(f"{modified_code[s_i:e_i]} is {truth_val}")

    prompt = f"{op_insn}{modified_code}\nassert f == "
    examples.append((prompt, ex['output'], {str((s_i+len(op_insn), e_i+len(op_insn))): truth_val for (s_i, e_i), truth_val in conditional_evaluation.items()}))
    cot_prompt = f"{op_insn_cot}{modified_code}\nassert f == ??"
    cot_examples.append((cot_prompt, ex['output'], {str((s_i+len(cot_prompt), e_i+len(cot_prompt))): truth_val for (s_i, e_i), truth_val in conditional_evaluation.items()}))

with open("cot/data/cruxeval_conditionals.json", "w") as wf:
    json.dump(examples, wf, indent=4)
with open("cot/data/cruxeval_conditionals_cot.json", "w") as wf:
    json.dump(cot_examples, wf, indent=4)


# from datasets import load_dataset, Dataset
# humaneval_dataset = load_dataset("openai_humaneval")
# cruxeval_dataset = load_dataset("cruxeval-org/cruxeval")

# liveness_probe_str = """
# {spaces_in_last}all_variables_hehaw = dir()
# {spaces_in_last}for name in all_variables_hehaw:
# {spaces_in_last}    if not name.startswith("__"):
# {spaces_in_last}        varval = eval(name)
# {spaces_in_last}        print(f"NAME:<<{{name}}>>TYPE:<<{{type(varval).__name__}}>>")
# {spaces_in_last}exit()
# """

# import random
# import subprocess
# import re

# def get_complete_paren(code_str):
#     if code_str.count("(") == 0 and code_str.count(")") == 0: return code_str
#     cursor = 0
#     open_p = 1
#     while cursor < len(code_str):
#         if code_str[cursor] in {'"', "'"}:
#             cursor += code_str[cursor+1:].find(code_str[cursor]) + 1 + 1
#             continue
#         if code_str[cursor] == "(": open_p += 1
#         if code_str[cursor] == ")": 
#             open_p -= 1
#             if open_p == 0:
#                 return code_str[:cursor]
#         cursor += 1
#     return code_str
            


# def get_var_types():
#     for ex in humaneval_dataset['test']:
#         # get appropriate imports
#         import_statements = []
#         if "import" in ex['prompt']:
#             import_statements = [import_line.strip() for import_line in ex['prompt'][:ex['prompt'].find("def ")].splitlines() if "import " in import_line and re.search(r'import ([^\s]+)', import_line).group(1) in ex['canonical_solution'] ]

#         # pull code out of canonical soln and remove returns
#         raw_code = ex['canonical_solution']
#         starting_space = re.search(r'\s*', raw_code[:raw_code.find('\n')]).group(0)
#         code_lines = [re.sub(r'return\s+(.*)', ex['entry_point']+r' = \1', cl[len(starting_space):]) for cl in raw_code.splitlines()] # assuming that doesnt dedent...

#         # get variables from fn header and initialize them with some value
#         random_var_initialization = get_complete_paren(random.choice(re.findall(r"candidate\((.*)\)", ex['test'])))
#         variable_matches = get_complete_paren(re.search(r'def [a-zA-Z_][a-zA-Z_0-9]*\((.*)\)', ex['prompt']).group(1))
#         varnames = [re.search(r'[a-zA-Z_][a-zA-Z_0-9]*', varmatch).group(0) for varmatch in variable_matches.split(',')]
#         code_setup = ", ".join(varnames) + " = " + random_var_initialization + '\n'

#         for last_line_no in random.sample(range(1, len(code_lines)+1), k=min(4, len(code_lines))):
#             spaces_in_last = re.search(r'\s*', code_lines[last_line_no-1]).group(0)
#             if (code_lines[last_line_no-1].strip()[:4] in {"for ", "for(", "def "} or code_lines[last_line_no-1].strip()[:3] in {"if ", "if("} or code_lines[last_line_no-1].strip()[:6] in {"while ", "while("} or code_lines[last_line_no-1].strip()[:5] in {"else ", "else:", "elif ", "elif("}) and (code_lines[last_line_no-1].strip()[-1] == ":"):
#                 spaces_in_last += '    ' 
#             with open("DELETETHIS.py", 'w') as wf:
#                 wf.write('\n'.join(import_statements) + code_setup + '\n'.join(code_lines[:last_line_no]) + liveness_probe_str.format(spaces_in_last=spaces_in_last))
#             try:
#                 collected_print = subprocess.check_output(['python', 'DELETETHIS.py'], timeout=15).decode('utf-8')
#             except SyntaxError:
#                 continue
#             except:
#                 continue

#             vars_already_captured = set()
#             for varname, vartype  in re.findall(r'NAME:<<(.+?)>>TYPE:<<(.+?)>>', collected_print):
#                 if varname in vars_already_captured: continue
#                 vars_already_captured.add(varname)
#                 yield {"code_prefix": '\n'.join(import_statements) + code_setup + '\n'.join(code_lines[:last_line_no]), "full_code": '\n'.join(import_statements) + code_setup + '\n'.join(code_lines), "varname": varname, "vartype": vartype}

#     for ex in cruxeval_dataset['test']:
#         raw_code = ex['code']
#         starting_space = re.search(r'\s*', raw_code[:raw_code.find('\n')]).group(0)
#         code_lines = [re.sub(r'return\s+(.*)', r'f = \1', cl[len(starting_space):]) for cl in raw_code.splitlines()] # assuming that doesnt dedent...

#         # get variables from fn header and initialize them with some value
#         variable_matches = get_complete_paren(re.search(r'def f\((.*)\)', raw_code).group(1))
#         if variable_matches:
#             varnames = [re.search(r'[a-zA-Z_][a-zA-Z_0-9]*', varmatch).group(0) for varmatch in variable_matches.split(',')]
#             code_setup = ", ".join(varnames) + " = " + ex['input'] + '\n'
#         else:
#             code_setup = ""

#         for last_line_no in random.sample(range(1, len(code_lines)+1), k=min(4, len(code_lines))):
#             spaces_in_last = re.search(r'\s*', code_lines[last_line_no-1]).group(0)
#             if (code_lines[last_line_no-1].strip()[:4] in {"for ", "for(", "def "} or code_lines[last_line_no-1].strip()[:3] in {"if ", "if("} or code_lines[last_line_no-1].strip()[:6] in {"while ", "while("} or code_lines[last_line_no-1].strip()[:5] in {"else ", "else:", "elif ", "elif("}) and (code_lines[last_line_no-1].strip()[-1] == ":"):
#                 spaces_in_last += '    ' 
#             with open("DELETETHIS.py", 'w') as wf:
#                 wf.write(code_setup + '\n'.join(code_lines[:last_line_no]) + liveness_probe_str.format(spaces_in_last=spaces_in_last))
#             try:
#                 collected_print = subprocess.check_output(['python', 'DELETETHIS.py'], timeout=15).decode('utf-8')
#             except SyntaxError:
#                 continue
#             except:
#                 continue

#             vars_already_captured = set()
#             for varname, vartype  in re.findall(r'NAME:<<(.+?)>>TYPE:<<(.+?)>>', collected_print):
#                 if varname in vars_already_captured: continue
#                 vars_already_captured.add(varname)
#                 yield {"code_prefix": code_setup + '\n'.join(code_lines[:last_line_no]), "full_code": code_setup + '\n'.join(code_lines), "varname": varname, "vartype": vartype}
                

# vars_ds = Dataset.from_generator(get_var_types)
# vars_ds.push_to_hub("celinelee/humaneval_cruxeval_types")
