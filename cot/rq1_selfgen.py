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

from prompts import *

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
        import pdb; pdb.set_trace()
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

def load_data_base(prompt, examples, suffix, tokenizer, train_split=0.8, max_num_ex=200):
    print(f"{len(examples)} total examples")
    train_examples = examples[:int(train_split*len(examples))]
    val_examples = examples[len(train_examples):]

    train_conditionals_data = [] # (input_ids, relevant_indices, label)
    for example in train_examples:
        code = example['code']
        conditionals = example['truth_states']
        train_conditionals_data.extend(get_tokenized_data(conditionals, prompt, code, suffix, tokenizer))
    random.shuffle(train_conditionals_data)
    train_conditionals_data = train_conditionals_data[:max_num_ex]
    print(f"cond train ex: {len(train_conditionals_data)}")

    val_conditionals_data = [] # (input_ids, relevant_indices, label)
    val_ex_maps = []
    for idx, example in enumerate(val_examples):
        code = example['code']
        conditionals = example['truth_states']
        true_answer = example['true_answer']
        tokenized_data = get_tokenized_data(conditionals, prompt, code, suffix, tokenizer)
        cond_indices = [len(val_conditionals_data), len(val_conditionals_data)+len(tokenized_data)]
        val_conditionals_data.extend(tokenized_data)
        val_ex_maps.append(cond_indices)
    random.shuffle(val_conditionals_data)
    val_conditionals_data = val_conditionals_data[:max_num_ex]
    print(f"cond val ex: {len(val_conditionals_data)}")
    return train_conditionals_data, (val_examples, val_conditionals_data, val_ex_maps)

def rq1(model_name, layers, args):
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_rq1_base(examples, fav_layer=10):
        suffix = "\nassert answer == "
        stop_strings = ["```"]
        experiment_name = "base"
        if not os.path.exists(f"experiment_outputs/{experiment_name}/{model_name}"):
            os.makedirs(f"experiment_outputs/{experiment_name}/{model_name}", exist_ok=True)
        train_conditionals_data, (val_examples, val_conditionals_data, val_ex_maps) = load_data_base(base_predict_insn, examples, suffix, tokenizer)
        cond_probe = LinearProbe(model_config, model, tokenizer, layers, f"{experiment_name}_conditionals")
        if args['overwrite_cache']:
            cond_probe.train(train_conditionals_data, val_conditionals_data, args)
            cond_probe.save(args['model_save_folder'])
        else: cond_probe.load_saved(args['model_save_folder'])

        summary_metrics = {
            "all": [],
            "gen_good_ws_bad": [],
            "gen_bad_ws_good": [],
            "none": [],
        }
        cond_probe_predictions = cond_probe.predict(val_conditionals_data)
        # For each test example...
        for ex_idx, cond_indices in enumerate(val_ex_maps):
            original_example = val_examples[ex_idx]
            code = original_example['code']
            output = original_example['true_answer']

            prompt = base_predict_insn + code + suffix
            prediction_info = {
                "code": code,
                "true_output": output,
                "prompt": prompt
            }
            # the conditionals probe predictions
            probe_predictions = {}
            got_all_probe_predictions_correct = True
            for offset_idx, (input_ids, relevant_indices, label) in enumerate(val_conditionals_data[cond_indices[0]:cond_indices[1]]):
                entity_string = tokenizer.decode(input_ids[relevant_indices[0]:relevant_indices[1]])
                probe_prediction = cond_probe_predictions[fav_layer][cond_indices[0]+offset_idx]
                probe_predictions[f"({relevant_indices[0]}, {relevant_indices[1]})"] = {
                    "entity_string": entity_string,
                    "probe_prediction": probe_prediction,
                    "actual_value": label
                }
                if label.strip() != probe_prediction.strip(): 
                    got_all_probe_predictions_correct = False
            prediction_info["probe_predictions"] = probe_predictions

            # printout the task prediction
            input_ids = tokenizer(prompt, return_tensors='pt').to(device)
            model_output = model.generate(**input_ids, return_dict_in_generate=True, max_new_tokens=1024, tokenizer=tokenizer, stop_strings=stop_strings)
            predicted_output = tokenizer.batch_decode(model_output.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)[0]
            prediction_info[f"prediction::{model_name}"] = {
                "completion": predicted_output
            }
            try:
                predicted_output = re.search(r'(.+?)#', predicted_output).group(1).strip()
            except: pass
            prediction_info[f"prediction::{model_name}"]["extracted_prediction"] = predicted_output
            generation_is_correct = predicted_output.strip() == output.strip()

            if generation_is_correct and got_all_probe_predictions_correct:
                summary_metrics["all"].append(prediction_info)
            if generation_is_correct and (not got_all_probe_predictions_correct):
                summary_metrics["gen_good_ws_bad"].append(prediction_info)
            if (not generation_is_correct) and got_all_probe_predictions_correct:
                summary_metrics["gen_bad_ws_good"].append(prediction_info)
            if (not generation_is_correct) and (not got_all_probe_predictions_correct):
                summary_metrics["none"].append(prediction_info)
            
            with open(f"experiment_outputs/{experiment_name}/{model_name}/layer{fav_layer}.json", 'w') as wf:
                json.dump(summary_metrics, wf, indent=4)
                
            for key in summary_metrics:
                print(f"{key}: {len(summary_metrics[key])}")

        print("FINAL BASE SUMMARY")
        for key in summary_metrics:
            print(f"{key}: {len(summary_metrics[key])}")

    examples = json.load(open("cot/data/logic.json"))
    run_rq1_base(examples)
    
train_args = {
    "overwrite_cache": False,
    "save_folder": "data",
    "model_save_folder": "saved_probes"
}
rq1("meta-llama/Meta-Llama-3.1-8B-Instruct", [8, 10, 12, 16, 24], train_args)
