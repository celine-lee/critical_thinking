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

foldername = "outputs/k1_N1_t1"


system_instruction = """You are a smart and helpful AI assistant. Please help me answer the following question."""


query_template = """Question: {question}"""

generation_instructions = {
    "request_descriptor": "{descriptor} provide just the final answer following this template: [ANSWER]\nThe answer is YOUR ANSWER\n[/ANSWER]",
}

stop_strings = ["[/ANSWER]"]

def make_prompt(length_control_metadata, question):
    (length_control_mode, length_control_kwargs) = length_control_metadata
    prompt = system_instruction + "\n\n"
    if length_control_mode in generation_instructions:
        if length_control_mode == "request_descriptor":
            if length_control_kwargs["descriptor"] == "brief":
                length_control_kwargs["descriptor"] = "Provide your thought process succinctly, then"
            elif length_control_kwargs["descriptor"] == "detail":
                length_control_kwargs["descriptor"] = "Provide your thought process in detail, then"
            elif length_control_kwargs["descriptor"] == "none":
                length_control_kwargs["descriptor"] = "Do not generate any other text, only"
        prompt += generation_instructions[length_control_mode].format(**length_control_kwargs) + "\n\n"
    else:
        raise NotImplementedError
    prompt += query_template.format(question=question) + "\n"
    return prompt

class Experiment:
    def __init__(self, model, model_name, max_batch_size, length_control_metadata, num_qs, n_samples_per_q, temperature):
        self.model = model
        self.modelname = re.search(r"/(.+)", model_name).group(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.n_samples_per_q = n_samples_per_q
        (length_control_mode, length_control_kwargs) = length_control_metadata
        if length_control_mode in {"detail", "states_long"}: 
            self.max_batch_size = int(self.max_batch_size * 0.1)
        if length_control_mode in {"brief", "states_short"}:
            self.max_batch_size = int(self.max_batch_size * 0.5)
        self.length_control_mode = length_control_mode
        self.length_control_kwargs = length_control_kwargs
        self.filename = f"{self.modelname}_T{self.temperature}_{num_qs}qs_{self.length_control_mode}_{''.join(self.length_control_kwargs.values())}"

        self.examples = load_dataset('mandarjoshi/trivia_qa', 'rc', split='validation').select(range(num_qs))


    def extract_answers(
        self, input_ids, model_output
    ):
        reprompt_string = "[ANSWER]\nThe answer is "

        model_predictions = self.tokenizer.batch_decode(
            model_output.sequences, skip_special_tokens=True,
        )
        new_queries = []
        gens_need_augmenting = []
        extracted_answers_and_indices = [None for _ in model_predictions]
        for gen_idx, model_prediction in enumerate(model_predictions):
            query_len = len(self.tokenizer.decode(input_ids.input_ids[gen_idx], skip_special_tokens=True))
            parsed_answer = None
            for parsed_answer in re.finditer(r'The answer is:?(.+)[\.\[\n]?', model_prediction):
                pass # only get the last
            if parsed_answer is None or (parsed_answer.start() < query_len):
                gens_need_augmenting.append(gen_idx)
                new_queries.append(model_prediction + reprompt_string)
                num_generated_tokens = torch.sum(
                    model_output.sequences[gen_idx, input_ids.input_ids.shape[-1] :]
                    != self.tokenizer.pad_token_id
                ).item()
                answer = None
                answer_indices = None
            else:
                answer = parsed_answer.group(1)
                for ss in stop_strings:
                    if answer.endswith(ss): answer = answer[:-len(ss)]
                answer = answer.rstrip(" \n.")
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
            for parsed_answer in re.finditer(r'The answer is (.+)', model_prediction):
                pass # only get the last
            if parsed_answer is None or (parsed_answer.start() < new_query_len - len(reprompt_string)):
                answer = None
                answer_indices = None
            else:
                answer = parsed_answer.group(1)
                for ss in stop_strings:
                    if answer.endswith(ss): answer = answer[:-len(ss)]
                answer = answer.rstrip(" \n.")
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
            (_, _, prev_num_generated_tokens, _, prev_generated) = extracted_answers_and_indices[orig_idx]
            num_generated_tokens = torch.sum(
                model_output.sequences[new_idx, new_input_ids.input_ids.shape[-1] :]
                != self.tokenizer.pad_token_id
            ).item()
            total_compute_tokens = torch.sum(
                model_output.sequences[new_idx] != self.tokenizer.pad_token_id
            ).item()
            extracted_answers_and_indices[orig_idx] = (
                answer,
                answer_indices,
                prev_num_generated_tokens + num_generated_tokens,
                total_compute_tokens,
                prev_generated + reprompt_string + model_prediction[new_query_len:],
            )
        return extracted_answers_and_indices

    def get_generation_config(self):
        return {
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": 1200 if self.length_control_mode in {"brief", "states_short"} else (2048 if self.length_control_mode in {"states_long", "detail"} else 648),
            "tokenizer": self.tokenizer,
            "stop_strings": stop_strings,
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def get_generation_config_final_answer(self):
        return {
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": 50,
            "tokenizer": self.tokenizer,
            "stop_strings": stop_strings,
            "num_return_sequences": 1,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
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
        while answer_string not in self.tokenizer.decode(output_ids[start_tok_idx:end_tok_idx], skip_special_tokens=True):
            end_tok_idx += 1
            if end_tok_idx > len(output_ids): breakpoint()
        return (start_tok_idx, end_tok_idx)

    def load_existing_results(self):
        target_num_qs = len(self.examples)
        max_num_qs_before = 0
        results = []
        for filename in glob.glob(os.path.join(foldername, f"{self.modelname}_T{self.temperature}_*_{self.length_control_mode}_{''.join(self.length_control_kwargs.values())}.json")):
            num_qs = re.search(r'_(\d+)qs_', filename)
            if num_qs is None: continue
            num_qs = num_qs.group(1)
            if int(num_qs) > target_num_qs: 
                more_results = json.load(open(filename))
                results = [ex for ex in more_results if ex["question_idx"] < target_num_qs]
                return results
            if int(num_qs) > max_num_qs_before:
                results = json.load(open(filename))
                max_num_qs_before = int(num_qs)
        return results

    def run_experiment(self):
        os.makedirs(foldername, exist_ok=True)
        filename = f"{foldername}/{self.filename}.json"
        print(filename)
        results = self.load_existing_results()

        gens_per_q = {idx: len([elt for elt in results if elt["question_idx"] == idx]) for idx in range(len(self.examples))}

        q_idx = 0
        while q_idx < len(self.examples):
            gen_idx_to_q_idx_mapping = {}
            inputs = []
            while len(inputs) < self.max_batch_size:
                n_gens_remaining = self.n_samples_per_q - gens_per_q[q_idx]
                prompt = make_prompt((self.length_control_mode, self.length_control_kwargs), self.examples[q_idx]['question'])
                if len(inputs) + n_gens_remaining > self.max_batch_size:
                    n_gens = self.max_batch_size - len(inputs)
                    for idx in range(len(inputs),len(inputs)+n_gens):
                        gen_idx_to_q_idx_mapping[idx] = q_idx
                else:
                    n_gens = n_gens_remaining
                    for idx in range(len(inputs),len(inputs)+n_gens):
                        gen_idx_to_q_idx_mapping[idx] = q_idx
                    q_idx += 1
                    if q_idx == len(self.examples): break
                inputs.extend([prompt] * n_gens)
            if len(inputs) == 0: continue

            input_ids = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
                return_offsets_mapping=True,
            ).to(device)
            
            generation_config = self.get_generation_config()
            model_output = self.model.generate(
                input_ids=input_ids.input_ids,
                attention_mask=input_ids.attention_mask,
                **generation_config,
            )
            extracted_answers_and_indices = self.extract_answers(input_ids, model_output)

            for gen_idx, (pred_answer, _, num_generated_tokens, total_compute_tokens, model_generation) in enumerate(extracted_answers_and_indices):
                q_idx = gen_idx_to_q_idx_mapping[gen_idx]
                q_example = self.examples[q_idx]
                if pred_answer is None:
                    is_correct = False
                else:
                    is_correct = re.sub(r'[^\w\s]', '', pred_answer.lower().strip()) in q_example['answer']['normalized_aliases']
                results.append(
                    {
                        "question_idx": q_idx,
                        "query": inputs[gen_idx],
                        "model_generation": model_generation,
                        "total_compute_tokens": total_compute_tokens,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": q_example["answer"]['normalized_aliases'],
                        "correct": is_correct,
                    }
                )
                gens_per_q[q_idx] += 1

            if q_idx % 10 < 2:
                with open(filename, "w") as wf:
                    json.dump(results, wf, indent=4)

        return results

def run():
    models = [
        ("meta-llama/Llama-3.1-8B-Instruct", 6),
        ("meta-llama/Llama-3.2-3B-Instruct", 6),
        ("meta-llama/Llama-3.2-1B-Instruct", 8),
    ]
    length_control_mode = "request_descriptor"
    num_qs = int(sys.argv[1])
    n_samples_per_q = int(sys.argv[2])
    temperature = sys.argv[3]

    for (model_name, batch_size) in models:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        for descriptor in ["none", "detail", "brief"]:
            experiment = Experiment(model, model_name, batch_size, (length_control_mode, {"descriptor": descriptor}), num_qs=num_qs, n_samples_per_q=n_samples_per_q, temperature=float(temperature))
            results = experiment.run_experiment()

if __name__ == "__main__":
    run()
