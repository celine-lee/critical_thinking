
from prompts import scratchpad_examples, code_solving_insn, fs_code, scratchpad_query_template, scratchpad_template, cot_query_template

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import json
import re
import sys
import ipdb
import traceback
import ast


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

def get_code_str_from_tree_node(ast_node, og_code):
    if 'lineno' not in dir(ast_node):
        print(f"Cannot get codestr for node {ast_node}")
        return None
    code_lines = og_code.splitlines(keepends=True)
    start_index = sum(len(line) for line in code_lines[:ast_node.lineno-1]) + ast_node.col_offset
    end_index = sum(len(line) for line in code_lines[:ast_node.end_lineno-1]) + ast_node.end_col_offset
    return (start_index, end_index, og_code[start_index:end_index])


def fs_basic(model, tokenizer, examples, max_batch_size):
    finish_assert_template = "```\n{code}\n```"
    code_solving_prompt = code_solving_insn + "\n\n".join(finish_assert_template.format(code=code) for code in fs_code)
    outputs = []
    correct = 0
    stop_strings = ['```']
    ex_idx = 0
    while ex_idx < len(examples):
        exs = examples[ex_idx:min(len(examples), ex_idx+max_batch_size)]
        queries = []
        for ex in exs:
            code_assert_prefix = re.search(r'^[\s\S]*assert answer == ', ex['code'], re.MULTILINE).group(0).strip()
            query = code_solving_prompt + f"\n\n```\n{code_assert_prefix}"
            queries.append(query)
        input_ids = tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to(device)
        model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=1024, tokenizer=tokenizer, stop_strings=stop_strings)
        model_predictions = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        for batch_idx, model_prediction in enumerate(model_predictions):
            assert_line = (f"assert answer == {model_prediction.strip().rstrip('`')}")
            predicted_answer = None
            if assert_line: 
                try:
                    tree = ast.parse(assert_line.strip())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assert):
                            if isinstance(node.test.ops[0], ast.Eq) and isinstance(node.test.left, ast.Name) and node.test.left.id == "answer":
                                answer_node = node.test.comparators[0]
                                (_, _, predicted_answer) = get_code_str_from_tree_node(answer_node, assert_line)
                                break
                    predicted_answer = predicted_answer.strip()
                except:
                    pass
            is_correct = eval(exs[batch_idx]['true_answer']) == eval(predicted_answer)
            outputs.append({
                "input_example": exs[batch_idx],
                "query": queries[batch_idx],
                "model_prediction": model_prediction,
                "total_compute_tokens": torch.sum(model_output.sequences[batch_idx] != tokenizer.pad_token_id).item(),
                "answer": predicted_answer,
                "true_answer": exs[batch_idx]['true_answer'],
                "correct": is_correct
            })

            correct += int(is_correct)
        ex_idx += max_batch_size
    return correct, outputs

def fs_scratchpad(model, tokenizer, examples, max_batch_size):
    outputs = []
    prompt = "\n\n".join(scratchpad_template.format(code=ex[0], trace=ex[1]) for ex in scratchpad_examples)
    stop_strings = ["[END]"]
    correct = 0
    ex_idx = 0
    while ex_idx < len(examples):
        exs = examples[ex_idx:min(len(examples), ex_idx+max_batch_size)]
        queries = [prompt + '\n\n' + scratchpad_query_template.format(code=ex['code']) for ex in exs]
        input_ids = tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to(device)
        model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=1024, tokenizer=tokenizer, stop_strings=stop_strings)
        model_predictions = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        for batch_idx, model_prediction in enumerate(model_predictions):
            trace_lines = model_prediction.splitlines(keepends=True)
            final_state_line = len(trace_lines) - 1
            while final_state_line >= 0 and re.search(r'state:.+"answer":', trace_lines[final_state_line]) is None:
                final_state_line -= 1
            try: 
                final_state = re.search(r'.*state: ({.*})', trace_lines[final_state_line]).group(1)
                final_state = eval(final_state)
            except: final_state = {}
            is_correct = "answer" in final_state and str(eval(exs[batch_idx]['true_answer'])) == str(final_state["answer"])
            outputs.append({
                "input_example": exs[batch_idx],
                "query": queries[batch_idx],
                "model_prediction": model_prediction,
                "total_compute_tokens": torch.sum(model_output.sequences[batch_idx] != tokenizer.pad_token_id).item(),
                "answer": str(final_state['answer']) if "answer" in final_state else None,
                "true_answer": exs[batch_idx]['true_answer'],
                "correct": is_correct
            })
            correct += int(is_correct)

        ex_idx += max_batch_size
        
    return correct, outputs

def os_cot(model, tokenizer, examples, max_batch_size):
    correct = 0
    outputs = []
    stop_strings = ['[/ANSWER]']
    ex_idx = 0
    while ex_idx < len(examples):
        exs = examples[ex_idx:min(len(examples), ex_idx+max_batch_size)]
        queries = [cot_query_template.format(code=ex['code']) for ex in exs]
        input_ids = tokenizer(queries, padding=True, truncation=True, return_tensors=True).to(device)
        model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=1024, tokenizer=tokenizer, stop_strings=stop_strings)
        model_predictions = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        for batch_idx, model_prediction in enumerate(model_predictions):
            assert_line = re.search(r'\[ANSWER\](.+?)\[/ANSWER\]', model_prediction)
            predicted_answer = None
            if assert_line: 
                try:
                    tree = ast.parse(assert_line.strip())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assert):
                            if isinstance(node.test.ops[0], ast.Eq) and isinstance(node.test.left, ast.Name) and node.test.left.id == "answer":
                                answer_node = node.test.comparators[0]
                                (_, _, predicted_answer) = get_code_str_from_tree_node(answer_node, assert_line)
                                break
                    predicted_answer = predicted_answer.strip()
                except:
                    pass
            is_correct = eval(exs[batch_idx]['true_answer']) == eval(predicted_answer)
            outputs.append({
                "query": queries[batch_idx],
                "model_prediction": model_prediction,
                "total_compute_tokens": torch.sum( model_output.sequences[batch_idx] != tokenizer.pad_token_id).item(),
                "answer": predicted_answer,
                "true_answer": exs[batch_idx]['true_answer'],
                "correct": is_correct
            })

            correct += int(is_correct)
         
        ex_idx += max_batch_size

    return correct, outputs


def run_experiment(model_name, examples_file, max_batch_size):
    examples = json.load(open(examples_file))
    examples = examples[:200]
    total_ex = len(examples)
    modelname = re.search(r'/(.+)', model_name).group(1)

    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(" ====== FS BASIC ===== ")
    fs_basic_correct, fs_basic_outputs = fs_basic(model, tokenizer, examples, max_batch_size)
    print(f"Correct: {fs_basic_correct} / {total_ex}")
    print(f"Avg compute tokens: {sum(op['total_compute_tokens'] for op in fs_basic_outputs) / total_ex:.2f}")
    with open(f"arrayworld_fs_basic_{modelname}.json", 'w') as wf:
        json.dump(fs_basic_outputs, wf, indent=4)

    print(" ====== FS SCRATCHPAD ===== ")
    fs_sp_correct, fs_sp_outputs = fs_scratchpad(model, tokenizer, examples, max_batch_size)
    print(f"Correct: {fs_sp_correct} / {total_ex}")
    print(f"Avg compute tokens: {sum(op['total_compute_tokens'] for op in fs_sp_outputs) / total_ex:.2f}")
    with open(f"arrayworld_fs_sp_{modelname}.json", 'w') as wf:
        json.dump(fs_sp_outputs, wf, indent=4)

    print(" ====== OS COT ===== ")
    os_cot_correct, os_cot_outputs = fs_basic(model, tokenizer, examples, max_batch_size)
    print(f"Correct: {os_cot_correct} / {total_ex}")
    print(f"Avg compute tokens: {sum(op['total_compute_tokens'] for op in os_cot_outputs) / total_ex:.2f}")
    with open(f"arrayworld_os_cot_{modelname}.json", 'w') as wf:
        json.dump(os_cot_outputs, wf, indent=4)

run_experiment("meta-llama/Meta-Llama-3.1-8B-Instruct", "arrayworld_Meta-Llama-3.1-8B-Instruct_N20.json", 8)