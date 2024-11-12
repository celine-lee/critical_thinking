import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
import json
import re
import sys
import ipdb
import traceback
import os
import ast
from tqdm import tqdm
import numpy as np
import random
import glob
import pandas as pd


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook



foldername = "parity_outputs_describelength"

def random_parity(k, N):
    digit_options = list(range(k))
    digits = random.choices(digit_options, k=N)
    parity = sum(digits) % k
    return [f"{dig}" for dig in digits], parity

system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""

generation_instructions = {
    "request_L": "Provide your thought process using at least {L} words, then provide the answer following this template: [ANSWER]\nY = YOUR ANSWER\n[/ANSWER]",
    "request_descriptor": "{descriptor}Provide the answer following this template: [ANSWER]\nY = YOUR ANSWER\n[/ANSWER]",
}

def check_parity_even_odd(pred_answer, true_parity):
    if pred_answer.strip().lower() not in {"even", "odd"}: return False
    if true_parity % 2: return pred_answer.strip().lower() == "odd"
    return pred_answer.strip().lower() == "even"


end_queries = {
    "parity": ("What is Y?", lambda pred_answer, true_parity: pred_answer.strip().isdigit() and (int(pred_answer.strip()) == true_parity)),
    "even_odd": ("Is Y even or odd?", lambda pred_answer, true_parity: check_parity_even_odd),
}

query_template = """I want to perform the following sum modulo {k}:
Y = ({string_add}) mod {k}
{end_query}

"""

stop_strings = ["[/ANSWER]"]

def make_prompt(length_control_metadata, k, digits, end_query):
    (length_control_mode, length_control_kwargs) = length_control_metadata
    prompt = system_instruction + "\n\n"
    if length_control_mode:
        if length_control_mode in generation_instructions:

            if length_control_mode == "request_descriptor":
                if length_control_kwargs["descriptor"] == "brief":
                    length_control_kwargs["descriptor"] = "Provide your thought process very briefly. "
                elif length_control_kwargs["descriptor"] == "detail":
                    length_control_kwargs["descriptor"] = "Provide your thought process in detail. "
                elif length_control_kwargs["descriptor"] == "None":
                    length_control_kwargs["descriptor"] = ""
            prompt += generation_instructions[length_control_mode].format(**length_control_kwargs) + "\n\n"
        elif length_control_mode == "eos_decay":
            pass # done during generation not prompting
        else:
            raise NotImplementedError
    prompt += query_template.format(k=k, string_add = " + ".join(digits), end_query=end_query)
    return prompt

class ParityExperiment:

    def __init__(self, model_name, max_batch_size, length_control_metadata, n_samples = 100, temperature=0.9):
        model_config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.modelname = re.search(r"/(.+)", model_name).group(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.n_samples = n_samples
        (length_control_mode, length_control_kwargs) = length_control_metadata
        self.length_control_mode = length_control_mode
        self.length_control_kwargs = length_control_kwargs
        self.filename = f"{self.modelname}_T{self.temperature}_{self.length_control_mode}_{''.join(self.length_control_kwargs.values())}"


    def extract_answers(
        self, input_ids, model_output, answer_template_regex=r'Y = (\d+)'
    ):
        input_idx = 0
        query_len = len(self.tokenizer.decode(input_ids.input_ids[input_idx], skip_special_tokens=True))
        model_predictions = self.tokenizer.batch_decode(
            model_output.sequences, skip_special_tokens=True,
        )

        new_queries = []
        gens_need_augmenting = []
        extracted_answers_and_indices = [None for _ in model_predictions]
        for gen_idx, model_prediction in enumerate(model_predictions):
            parsed_answer = None
            for parsed_answer in re.finditer(answer_template_regex, model_prediction):
                pass # only get the last
            if parsed_answer is None:
                for ss in stop_strings:
                    if model_prediction.endswith(ss):
                        model_prediction = model_prediction[:-len(ss)]
                gens_need_augmenting.append(gen_idx)
                new_queries.append(model_prediction + "[ANSWER]\nY = ")
            else:
                answer = parsed_answer.group(1).rstrip(" .")
                string_index_start = parsed_answer.start() + parsed_answer.group(0).index(answer)
                string_index_end = string_index_start + len(answer)
                answer_indices = self.get_token_indices(
                    model_output.sequences[gen_idx],
                    answer,
                    string_index_start,
                    string_index_end,
                    init_tok_offset=input_ids.input_ids.shape[-1],
                    init_char_offset=query_len
                )
                num_generated_tokens = torch.sum(
                    model_output.sequences[gen_idx, input_ids.input_ids.shape[-1] :]
                    != self.tokenizer.pad_token_id
                ).item()
                total_compute_tokens = torch.sum(
                    model_output.sequences[gen_idx] != self.tokenizer.pad_token_id
                ).item()
                extracted_answers_and_indices[gen_idx] = (
                    answer,
                    answer_indices,
                    num_generated_tokens,
                    total_compute_tokens,
                    model_prediction[query_len:],
                )

        if len(new_queries) == 0: return extracted_answers_and_indices
        new_input_ids = self.tokenizer(
            new_queries,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)

        generation_config = self.get_generation_config_final_answer()
        model_output = self.model.generate(
            input_ids=new_input_ids.input_ids,
            attention_mask=new_input_ids.attention_mask,
            **generation_config,
        )
        for new_idx, orig_idx in enumerate(gens_need_augmenting):
            model_prediction = self.tokenizer.decode(model_output.sequences[new_idx], skip_special_tokens=True)
            new_query_len = len(new_queries[new_idx])
            parsed_answer = None
            for parsed_answer in re.finditer(answer_template_regex, model_prediction):
                pass # only get the last
            if parsed_answer is None:
                answer = None
                answer_indices = None
            else:
                answer = parsed_answer.group(1).rstrip(" .")
                string_index_start = parsed_answer.start() + parsed_answer.group(0).index(answer)
                string_index_end = string_index_start + len(answer)
                answer_indices = self.get_token_indices(
                    model_output.sequences[new_idx],
                    answer,
                    string_index_start,
                    string_index_end,
                    init_tok_offset=new_input_ids.input_ids.shape[-1],
                    init_char_offset=new_query_len
                )
            num_generated_tokens = torch.sum(
                model_output.sequences[new_idx, input_ids.input_ids.shape[-1] :]
                != self.tokenizer.pad_token_id
            ).item()
            total_compute_tokens = torch.sum(
                model_output.sequences[new_idx] != self.tokenizer.pad_token_id
            ).item()
            extracted_answers_and_indices[orig_idx] = (
                answer,
                answer_indices,
                num_generated_tokens,
                total_compute_tokens,
                model_prediction[query_len:],
            )
        return extracted_answers_and_indices

    def get_generation_config(self):
        num_gens = min(self.max_batch_size, self.n_samples)
        return {
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": 1200,
            "tokenizer": self.tokenizer,
            "stop_strings": stop_strings,
            "num_return_sequences": num_gens,
            "do_sample": True,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def get_generation_config_final_answer(self):
        return {
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": 10,
            "tokenizer": self.tokenizer,
            "stop_strings": stop_strings,
            "num_return_sequences": 1,
            "do_sample": False,
            "temperature": None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def get_token_indices(
        self, output_ids, answer_string, string_index_start, string_index_end, init_tok_offset=0, init_char_offset=0
    ):
        output_tokens = self.tokenizer.convert_ids_to_tokens(output_ids)
        start_tok_idx = init_tok_offset
        curr_offset = init_char_offset
        while start_tok_idx < len(output_tokens):
            output_tok = output_tokens[start_tok_idx]
            if output_tok in self.tokenizer.all_special_tokens:
                start_tok_idx += 1
                continue
            curr_offset += len(output_tok)
            if curr_offset >= string_index_start:
                break
            start_tok_idx += 1
        end_tok_idx = start_tok_idx + 1
        while answer_string not in self.tokenizer.decode(output_ids[start_tok_idx:end_tok_idx]):
            end_tok_idx += 1
            if end_tok_idx > len(output_ids): breakpoint()
        return (start_tok_idx, end_tok_idx)

    def run_experiment(self, k, N, end_query, check_answer):
        subfolder = os.path.join(foldername, f"k{k}_N{N}")
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print(filename)
        if os.path.exists(filename): return json.load(open(filename))

        ex_digits, true_parity = random_parity(k, N)

        prompt = make_prompt((self.length_control_mode, self.length_control_kwargs), k, ex_digits, end_query)
        input_ids = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)
        
        results = []
        n_gens_remaining = self.n_samples
        while n_gens_remaining > 0:
            generation_config = self.get_generation_config()
            model_output = self.model.generate(
                input_ids=input_ids.input_ids,
                attention_mask=input_ids.attention_mask,
                **generation_config,
            )
            extracted_answers_and_indices = self.extract_answers(input_ids, model_output)

            for gen_idx, (pred_answer, _, num_generated_tokens, total_compute_tokens, model_generation) in enumerate(extracted_answers_and_indices):
                is_correct = pred_answer is not None and check_answer(pred_answer, true_parity)
                results.append(
                    {
                        "query": prompt,
                        "model_generation": model_generation,
                        "total_compute_tokens": total_compute_tokens,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": true_parity,
                        "correct": is_correct,
                    }
                )
            n_gens_remaining -= generation_config["num_return_sequences"]

        with open(filename, "w") as wf:
            json.dump(results, wf, indent=4)

        return results

def run():
    models = [
        ("meta-llama/Llama-3.2-1B-Instruct", 28),
        ("meta-llama/Llama-3.2-3B-Instruct", 20),
        ("meta-llama/Llama-3.1-8B-Instruct", 16),
    ]
    length_control_mode = "request_descriptor"
    # length_control_mode = "request_L"
    query_mode = "parity"
    (end_query, Y_to_query) = end_queries[query_mode] 
    n_samples = 100

    for (model_name, batch_size) in models:
        for descriptor in {"detail", "brief", "None"}:
            experiment = ParityExperiment(model_name, batch_size, (length_control_mode, {"descriptor": descriptor}), n_samples=n_samples)
            for k in range(2, 6, 2):
                for N in range(8, 25, 4):
                    # for L in range(0, N*10, 2*N):
                    #     results = experiment.run_experiment(k, N, end_query, Y_to_query, (length_control_mode, {"L": L + 10}), n_samples)
                    results = experiment.run_experiment(k, N, end_query, Y_to_query)

if __name__ == "__main__":
    if "run" in sys.argv: run()