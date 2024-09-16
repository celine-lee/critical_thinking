import json
import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import re
import random
import sys
sys.path.append("../src")
sys.path.append(".")
from src.probe import LinearProbe
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_tokenized_data(property_data, prompt, code, suffix, tokenizer):
    sorted_property = []
    for s_i_e_i, tv in property_data.items():
        parsed_indices = re.search(r'\((\d+), (\d+)\)', s_i_e_i)
        s_i = int(parsed_indices.group(1))
        e_i = int(parsed_indices.group(2))
        sorted_property.append((s_i+len(prompt), e_i+len(prompt), tv.strip()))
    sorted_property.sort(key=lambda elt: elt[0])

    full_input = prompt + code + suffix
    input_ids = tokenizer(full_input, return_offsets_mapping=True, return_tensors='pt').to(device)

    data = []
    property_idx = 0
    property_tok_idx = [0, 1]
    tok_idx = 0
    while property_idx < len(sorted_property):
        while input_ids['offset_mapping'][0][tok_idx][0].item() <= sorted_property[property_idx][0]: 
            tok_idx += 1
            if tok_idx >= len(input_ids['offset_mapping'][0]): break
        if tok_idx: tok_idx -= 1
        # now token start >= this conditional start
        property_tok_idx[0] = tok_idx
        while input_ids['offset_mapping'][0][tok_idx][1].item() <= sorted_property[property_idx][1]: 
            tok_idx += 1
        # now token end > this conditional end
        property_tok_idx[1] = tok_idx
        data.append((input_ids.input_ids[0], list(property_tok_idx), sorted_property[property_idx][2]))
        property_idx += 1
    return data

def load_cruxeval_data(prompt, examples, suffix, tokenizer, train_split=0.8, max_num_ex=200):
    print(f"{len(examples)} total examples")
    train_examples = examples[:int(train_split*len(examples))]
    val_examples = examples[len(train_examples):]
    train_conditionals_data = [] # (input_ids, relevant_indices, label)
    train_types_data = [] # (input_ids, relevant_indices, label)
    for example in train_examples:
        (code, _, conditionals, types) = example
        train_conditionals_data.extend(get_tokenized_data(conditionals, prompt, code, suffix, tokenizer))
        train_types_data.extend(get_tokenized_data(types, prompt, code, suffix, tokenizer))
    random.shuffle(train_conditionals_data)
    random.shuffle(train_types_data)
    train_conditionals_data = train_conditionals_data[:max_num_ex]
    train_types_data = train_types_data[:max_num_ex]
    print(f"cond train ex: {len(train_conditionals_data)}")
    print(f"types train ex: {len(train_types_data)}")
    val_data = reindex_cruxeval_data(prompt, val_examples, suffix, tokenizer)
    print(f"cond test ex: {len(val_data[0])}")
    print(f"types test ex: {len(val_data[2])}")
    return (train_conditionals_data, train_types_data), (val_examples, val_data)

def reindex_cruxeval_data(prompt, examples, suffix, tokenizer):
    conditionals_data = [] # (input_ids, relevant_indices, label)
    types_data = [] # (input_ids, relevant_indices, label)
    ex_maps = []
    for idx, example in enumerate(examples):
        (code, _, conditionals, types) = example
        tokenized_data = get_tokenized_data(conditionals, prompt, code, suffix, tokenizer)
        cond_indices = [len(conditionals_data), len(conditionals_data)+len(tokenized_data)]
        conditionals_data.extend(tokenized_data)
        tokenized_data = get_tokenized_data(types, prompt, code, suffix, tokenizer)
        types_indices = [len(types_data), len(types_data)+len(tokenized_data)]
        types_data.extend(tokenized_data)
        ex_maps.append((cond_indices, types_indices))
    return (conditionals_data, types_data, ex_maps)

def rq1(model_name, layers, args):
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_rq1_experiment(examples, experiment_name, stop_strings, fav_layer=10):
        if experiment_name == "BASE": 
            prompt = op_insn
            suffix = "\nassert f == "
        else: 
            prompt = op_insn_cot
            suffix = "\nassert f == ??\n"
        train_data, (test_examples, test_data) = load_cruxeval_data(prompt, examples, suffix, tokenizer)
        print(experiment_name)
        # TODO Should we balance?
        print("CONDITIONALS")
        cond_probe = LinearProbe(model_config, model, tokenizer, layers, f"{experiment_name}_conditionals")
        if args['overwrite_cache']:
            cond_probe.train(train_data[0], test_data[0], args)
            cond_probe.save(args['model_save_folder'])
        else: cond_probe.load_saved(args['model_save_folder'])
        print("TYPES")
        types_probe = LinearProbe(model_config, model, tokenizer, layers, f"{experiment_name}_types")
        if args['overwrite_cache']:
            types_probe.train(train_data[1], test_data[1], args)
            types_probe.save(args['model_save_folder'])
        else: types_probe.load_saved(args['model_save_folder'])

        print("TASK")
        correct = 0
        conditionals_correct = [0, 0]
        types_correct = [0, 0]
        total_conditionals = [0, 0]
        total_types = [0, 0]
        (test_conditionals_data, test_types_data, test_ex_maps) = test_data
        cond_probe_predictions = cond_probe.predict(test_conditionals_data)
        types_probe_predictions = types_probe.predict(test_types_data)
        # For each test example...
        for ex_idx, (cond_indices, types_indices) in enumerate(test_ex_maps):
            original_example = test_examples[ex_idx]
            code = original_example[0]
            output = original_example[1]
            code_cond_indices = original_example[2]
            code_type_indices = original_example[3]
            print('-------------PROMPT--------------')
            first_promt = prompt + code + suffix
            print(first_promt)
            print('---------------------------')
            # printout the conditionals probe predictions
            for offset_idx, (input_ids, relevant_indices, label) in enumerate(test_conditionals_data[cond_indices[0]:cond_indices[1]]):
                print(f"{tokenizer.decode(input_ids[relevant_indices[0]:relevant_indices[1]])} is actually {label}") 
                print(f"\tL{fav_layer} pred: {cond_probe_predictions[fav_layer][cond_indices[0]+offset_idx]}") 
                if label.strip() == cond_probe_predictions[fav_layer][cond_indices[0]+offset_idx].strip(): conditionals_correct[0] += 1
                total_conditionals[0] += 1

            # printout the types probe predictions
            for offset_idx, (input_ids, relevant_indices, label) in enumerate(test_types_data[types_indices[0]:types_indices[1]]):
                print(f"{tokenizer.decode(input_ids[relevant_indices[0]:relevant_indices[1]])} is actually {label}") 
                print(f"\tL{fav_layer} pred: {types_probe_predictions[fav_layer][types_indices[0]+offset_idx]}") 
                if label.strip() == types_probe_predictions[fav_layer][types_indices[0]+offset_idx].strip(): types_correct[0] += 1
                total_types[0] += 1

            # printout the task prediction
            input_ids = tokenizer(first_promt, return_tensors='pt').to(device)
            model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=1024, tokenizer=tokenizer, stop_strings=stop_strings)
            predicted_output = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)[0]
            print(f"--> {predicted_output}")
            if experiment_name == "BASE":
                try:
                    predicted_output = re.search(r'(.+?)#', predicted_output).group(1).strip()
                    print(f"    --> {predicted_output}")
                except: pass

            # if we're doing COT REPROMPT, have it printout the original code again after reasoning before predicting again
            if experiment_name == "COT_REPROMPT": 
                new_suffix = "\n[ANSWER]assert f == "
                try:
                    original_prediction = re.search(r'\[ANSWER\][\s\S]*f == (.+?)\[/ANSWER\]', predicted_output).group(1).strip()
                    print(f"    --> {original_prediction}")
                except: pass
                if "[ANSWER]" in predicted_output:
                    predicted_output = predicted_output[:predicted_output.index("[ANSWER]")]
                re_prompted_example = [original_example]
                (reind_conditionals_data, reind_types_data, reind_ex_maps) = reindex_cruxeval_data(first_promt + predicted_output, re_prompted_example, new_suffix, tokenizer)
                (reind_cond_indices, reind_types_indices) = reind_ex_maps[0]
                reind_cond_probe_predictions = cond_probe.predict(reind_conditionals_data)
                reind_types_probe_predictions = types_probe.predict(reind_types_data)
                print('======= RE PROMPT =========')
                re_prompt = first_promt + predicted_output + code + new_suffix
                print(re_prompt)
                print('---------------------------')
                # printout the conditionals probe predictions
                for offset_idx, (input_ids, relevant_indices, label) in enumerate(reind_conditionals_data[reind_cond_indices[0]:reind_cond_indices[1]]):
                    print(f"{tokenizer.decode(input_ids[relevant_indices[0]:relevant_indices[1]])} is actually {label}") 
                    print(f"\tL{fav_layer} pred: {cond_probe_predictions[fav_layer][reind_cond_indices[0]+offset_idx]}") 
                    if label.strip() == cond_probe_predictions[fav_layer][reind_cond_indices[0]+offset_idx].strip(): conditionals_correct[1] += 1
                    total_conditionals[1] += 1

                # printout the types probe predictions
                for offset_idx, (input_ids, relevant_indices, label) in enumerate(reind_types_data[reind_types_indices[0]:reind_types_indices[1]]):
                    print(f"{tokenizer.decode(input_ids[relevant_indices[0]:relevant_indices[1]])} is actually {label}") 
                    print(f"\tL{fav_layer} pred: {types_probe_predictions[fav_layer][reind_types_indices[0]+offset_idx]}") 
                    if label.strip() == types_probe_predictions[fav_layer][reind_types_indices[0]+offset_idx].strip(): types_correct[1] += 1
                    total_types[1] += 1

                # printout the task prediction
                input_ids = tokenizer(re_prompt, return_tensors='pt').to(device)
                model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=300, tokenizer=tokenizer, stop_strings=["[/ANSWER]"])
                predicted_output = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)[0]
                print(f"--> {predicted_output}")
                try:
                    predicted_output = re.search(r'(.+?)\[/ANSWER\]', predicted_output).group(1).strip()
                    print(f"    --> {predicted_output}")
                except: pass

            print(f"Should be: {output}")
            if predicted_output.strip() == output.strip():
                correct += 1
        print(f"{experiment_name}: {correct}/{len(test_ex_maps)}")
        print(f"conditionals before: {conditionals_correct[0]}/{total_conditionals[0]}")
        print(f"conditionals after: {conditionals_correct[1]}/{total_conditionals[1]}")
        print(f"types before: {types_correct[0]}/{total_types[0]}")
        print(f"types after: {types_correct[1]}/{total_types[1]}")

    cot_examples = json.load(open("cot/data/cruxeval.json"))
    run_rq1_experiment(cot_examples, "COT_REPROMPT", ["[/ANSWER]"])

    
train_args = {
    "overwrite_cache": False,
    "save_folder": "data",
    "model_save_folder": "saved_probes"
}
rq1("meta-llama/Meta-Llama-3.1-8B-Instruct", [8, 10, 12, 16, 24], train_args)
