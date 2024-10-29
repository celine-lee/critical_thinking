
from prompts import self_ask_prompt, qa_basic_prompt, compgap_cot_prompt

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import json
import re
import sys
import ipdb
import traceback
import ast
import random


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook


def fs_selfask(model, tokenizer, examples, max_batch_size):
    outputs = []
    correct = 0
    stop_strings = ['\nQuestion:']
    ex_idx = 0
    while ex_idx < len(examples):
        exs = examples[ex_idx:min(len(examples), ex_idx+max_batch_size)]
        queries = []
        for ex in exs:
            query = self_ask_prompt.format(question=ex['Question'])
            queries.append(query)
        input_ids = tokenizer(queries, padding=True, truncation=True, max_length=2048,  return_tensors='pt').to(device)
        model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=1024, tokenizer=tokenizer, stop_strings=stop_strings)
        model_predictions = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        for batch_idx, model_prediction in enumerate(model_predictions):
            is_correct = False
            predicted_answer = re.search(r'So the final answer is: (.+)', model_prediction.strip())
            if predicted_answer:
                predicted_answer = predicted_answer.group(1).strip().rstrip(".")
                is_correct = predicted_answer.lower()  ==  exs[batch_idx]['Answer'].lower()
            outputs.append({
                "input_example": exs[batch_idx],
                "query": queries[batch_idx],
                "model_prediction": model_prediction,
                "total_compute_tokens": torch.sum(model_output.sequences[batch_idx] != tokenizer.pad_token_id).item(),
                "generated_tokens": torch.sum(model_output.sequences[batch_idx, input_ids.input_ids.shape[-1]:] != tokenizer.pad_token_id).item(),
                "answer": predicted_answer,
                "true_answer": exs[batch_idx]['Answer'],
                "correct": is_correct
            })

            correct += int(is_correct)
        ex_idx += max_batch_size
    return correct, outputs

def fs_basic(model, tokenizer, examples, max_batch_size):
    outputs = []
    correct = 0
    stop_strings = ['\nQuestion:']
    ex_idx = 0
    while ex_idx < len(examples):
        exs = examples[ex_idx:min(len(examples), ex_idx+max_batch_size)]
        queries = []
        for ex in exs:
            query = qa_basic_prompt.format(question=ex['Question'])
            queries.append(query)
        input_ids = tokenizer(queries, padding=True, truncation=True, max_length=2048,  return_tensors='pt').to(device)
        model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=1024, tokenizer=tokenizer, stop_strings=stop_strings)
        model_predictions = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        for batch_idx, model_prediction in enumerate(model_predictions):
            predicted_answer = model_prediction.strip()
            if predicted_answer.endswith("Question:"):
                predicted_answer = predicted_answer.split("Question:")[0].strip()
            is_correct =  predicted_answer.lower() ==  exs[batch_idx]['Answer'].lower()
            outputs.append({
                "input_example": exs[batch_idx],
                "query": queries[batch_idx],
                "model_prediction": model_prediction,
                "total_compute_tokens": torch.sum(model_output.sequences[batch_idx] != tokenizer.pad_token_id).item(),
                "generated_tokens": torch.sum(model_output.sequences[batch_idx, input_ids.input_ids.shape[-1]:] != tokenizer.pad_token_id).item(),
                "answer": predicted_answer,
                "true_answer": exs[batch_idx]['Answer'],
                "correct": is_correct
            })

            correct += int(is_correct)
        ex_idx += max_batch_size
    return correct, outputs

def fs_cot(model, tokenizer, examples, max_batch_size):
    outputs = []
    correct = 0
    stop_strings = ['\nQuestion:']
    ex_idx = 0
    while ex_idx < len(examples):
        exs = examples[ex_idx:min(len(examples), ex_idx+max_batch_size)]
        queries = []
        for ex in exs:
            query = compgap_cot_prompt.format(question=ex['Question'])
            queries.append(query)
        input_ids = tokenizer(queries, padding=True, truncation=True, max_length=2048,  return_tensors='pt').to(device)
        model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=1024, tokenizer=tokenizer, stop_strings=stop_strings)
        model_predictions = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        for batch_idx, model_prediction in enumerate(model_predictions):
            is_correct = False
            predicted_answer = re.search(r'So the final answer is: (.+)', model_prediction.strip())
            if predicted_answer:
                predicted_answer = predicted_answer.group(1).strip().rstrip(".")
                is_correct = predicted_answer.lower()  ==  exs[batch_idx]['Answer'].lower()
            outputs.append({
                "input_example": exs[batch_idx],
                "query": queries[batch_idx],
                "model_prediction": model_prediction,
                "total_compute_tokens": torch.sum(model_output.sequences[batch_idx] != tokenizer.pad_token_id).item(),
                "generated_tokens": torch.sum(model_output.sequences[batch_idx, input_ids.input_ids.shape[-1]:] != tokenizer.pad_token_id).item(),
                "answer": predicted_answer,
                "true_answer": exs[batch_idx]['Answer'],
                "correct": is_correct
            })

            correct += int(is_correct)
        ex_idx += max_batch_size
    return correct, outputs

def run_experiment(model_name, examples_file, max_batch_size):
    examples = json.load(open(examples_file))
    total_ex = len(examples)
    modelname = re.search(r'/(.+)', model_name).group(1)

    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(" ====== FS BASIC ===== ")
    fs_basic_correct, fs_basic_outputs = fs_basic(model, tokenizer, examples, max_batch_size)
    print(f"Correct: {fs_basic_correct} / {total_ex}")
    print(f"Avg tot tokens: {sum(op['total_compute_tokens'] for op in fs_basic_outputs) / total_ex:.2f}")
    print(f"Avg gen tokens: {sum(op['generated_tokens'] for op in fs_basic_outputs) / total_ex:.2f}")
    with open(f"outputs/compgap_fs_basic_{modelname}.json", 'w') as wf:
        json.dump(fs_basic_outputs, wf, indent=4)

    print(" ====== FS COT===== ")
    fs_cot_correct, fs_cot_outputs = fs_cot(model, tokenizer, examples, max_batch_size)
    print(f"Correct: {fs_cot_correct} / {total_ex}")
    print(f"Avg tot tokens: {sum(op['total_compute_tokens'] for op in fs_cot_outputs) / total_ex:.2f}")
    print(f"Avg gen tokens: {sum(op['generated_tokens'] for op in fs_cot_outputs) / total_ex:.2f}")
    with open(f"outputs/compgap_fs_cot_{modelname}.json", 'w') as wf:
        json.dump(fs_cot_outputs, wf, indent=4)

    print(" ====== FS SELF ASK ===== ")
    fs_sa_correct, fs_sa_outputs = fs_selfask(model, tokenizer, examples, max_batch_size)
    print(f"Correct: {fs_sa_correct} / {total_ex}")
    print(f"Avg tot tokens: {sum(op['total_compute_tokens'] for op in fs_sa_outputs) / total_ex:.2f}")
    print(f"Avg gen tokens: {sum(op['generated_tokens'] for op in fs_sa_outputs) / total_ex:.2f}")
    with open(f"outputs/compgap_fs_selfask_{modelname}.json", 'w') as wf:
        json.dump(fs_sa_outputs, wf, indent=4)


run_experiment("meta-llama/Llama-3.2-1B-Instruct", "data/bamboogle_prerelease.json", 12)
run_experiment("meta-llama/Llama-3.2-3B-Instruct", "data/bamboogle_prerelease.json", 12)
run_experiment("meta-llama/Llama-3.1-8B-Instruct", "data/bamboogle_prerelease.json", 12)