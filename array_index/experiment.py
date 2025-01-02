import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
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

foldername = "outputs"

def normalize_state(idx, k):
    while idx < 0: idx += k
    while idx >= k: idx -= k
    return idx

def random_dfa(k, t):
    # assume half split t+, t-
    states = list(range(k))
    edges = {state: list(range(state-(t//2), state+(t//2))) for state in states}
    return states, edges

def random_walk(edges, N):
    curr_state = 0
    turns = []
    while len(turns) < N:
        edge = random.choice(edges[curr_state])
        if edge < 0:
            turns.append(f"pointer = pointer - {-edge}")
        else:
            turns.append(f"pointer = pointer + {edge}")
        curr_state = normalize_state(curr_state + edge, len(edges))
    return turns, curr_state


system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""


query_template = """You are given a length-{k} array and must track the index of a 0-indexed pointer to the array. The pointer undergoes several modifications. The pointer wraps around the length of the array on both ends, so when it reaches {k} it becomes 0, when it reaches {k_plus_one} it becomes 1, when it reaches -1 it becomes {k_minus_1}, etc. What is the index of the pointer after all the modifications are complete? Provide the answer in the range [0, {k}).

pointer = 0
{sequence}
"""

generation_instructions = {
    "request_descriptor": "{descriptor} provide your final answer following this template: [ANSWER]\npointer == YOUR ANSWER\n[/ANSWER]",
    "none": "Provide your final answer following this template: [ANSWER]\npointer == YOUR ANSWER\n[/ANSWER]",
}

stop_strings = ["[/ANSWER]"]

def prompt_with_chat_template(tokenizer, length_control_metadata, states, edges, turns):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    (length_control_mode, length_control_kwargs) = length_control_metadata
    prompt = query_template.format(k=len(states), k_plus_one=len(states)+1, k_minus_1=len(states)-1, sequence="\n".join(turns)) + "\n"
    if length_control_mode in generation_instructions:
        if length_control_mode == "request_descriptor":
            if length_control_kwargs["descriptor"] == "brief":
                length_control_kwargs["descriptor"] = "Provide your thought process succinctly, then"
            elif length_control_kwargs["descriptor"] == "detail":
                length_control_kwargs["descriptor"] = "Provide your thought process in detail, then"
            elif length_control_kwargs["descriptor"] == "none":
                length_control_kwargs["descriptor"] = "Do not generate any other text, only"
            elif length_control_kwargs["descriptor"] == "states_short":
                length_control_kwargs["descriptor"] = "Do not generate any other text, only identify the location after each clue, then"
            elif length_control_kwargs["descriptor"] == "states_long":
                length_control_kwargs["descriptor"] = "Identify the location after each clue, then"
        elif length_control_mode == "none":
            length_control_kwargs = {}
        prompt += generation_instructions[length_control_mode].format(**length_control_kwargs)
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, length_control_metadata, states, edges, turns):
    if tokenizer.chat_template: return prompt_with_chat_template(tokenizer, length_control_metadata, states, edges, turns)
    (length_control_mode, length_control_kwargs) = length_control_metadata
    prompt = system_instruction + "\n\n"
    prompt += query_template.format(k=len(states), k_plus_one=len(states)+1, k_minus_1=len(states)-1, sequence="\n".join(turns)) + "\n"
    if length_control_mode in generation_instructions:
        if length_control_mode == "request_descriptor":
            if length_control_kwargs["descriptor"] == "brief":
                length_control_kwargs["descriptor"] = "Provide your thought process succinctly, then"
            elif length_control_kwargs["descriptor"] == "detail":
                length_control_kwargs["descriptor"] = "Provide your thought process in detail, then"
            elif length_control_kwargs["descriptor"] == "none":
                length_control_kwargs["descriptor"] = "Do not generate any other text, only"
            elif length_control_kwargs["descriptor"] == "states_short":
                length_control_kwargs["descriptor"] = "Do not generate any other text, only identify the location after each clue, then"
            elif length_control_kwargs["descriptor"] == "states_long":
                length_control_kwargs["descriptor"] = "Identify the location after each clue, then"
        elif length_control_mode == "none":
            length_control_kwargs = {}
        prompt += generation_instructions[length_control_mode].format(**length_control_kwargs) + "\n\n"
    else:
        raise NotImplementedError
    return prompt

class Experiment:
    def __init__(self, model, model_name, max_batch_size, length_control_metadata, n_samples, temperature, max_new_tokens=2400):
        self.model = model
        self.modelname = re.search(r"/(.+)", model_name).group(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.n_samples = n_samples
        (length_control_mode, length_control_kwargs) = length_control_metadata
        if length_control_mode in {"detail", "states_long"}: 
            self.max_batch_size = int(self.max_batch_size * 0.1)
        if length_control_mode in {"brief", "states_short"}:
            self.max_batch_size = int(self.max_batch_size * 0.5)
        self.length_control_mode = length_control_mode
        self.length_control_kwargs = length_control_kwargs
        self.filename = f"{self.modelname}_T{self.temperature}_{self.length_control_mode}_{''.join(self.length_control_kwargs.values())}"
        self.max_new_tokens = max_new_tokens

    def extract_answers(
        self, input_ids, model_output
    ):
        reprompt_string = "[ANSWER]\npointer == "
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
            for parsed_answer in re.finditer(r'pointer ==\s*(\d+)', model_prediction):
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
            for parsed_answer in re.finditer(r'pointer == (\d+)', model_prediction):
                pass # only get the last
            if parsed_answer is None or (parsed_answer.start() < query_len):
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
        num_gens = min(self.max_batch_size, self.n_samples)
        return {
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": self.max_new_tokens,
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

    def run_experiment(self, k, N, t):
        subfolder = os.path.join(foldername, f"k{k}_N{N}_t{t}")
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print(filename)
        results = []
        if os.path.exists(filename): results = json.load(open(filename))

        n_gens_remaining = self.n_samples - len(results)
        while n_gens_remaining > 0:
            states, edges = random_dfa(k, t)
            turns, true_final_location = random_walk(edges, N)

            prompt = make_prompt(self.tokenizer, (self.length_control_mode, self.length_control_kwargs), states, edges, turns)
            input_ids = self.tokenizer(
                [prompt],
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
                is_correct = pred_answer is not None and eval(pred_answer) == true_final_location
                results.append(
                    {
                        "query": prompt,
                        "model_generation": model_generation,
                        "total_compute_tokens": total_compute_tokens,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": true_final_location,
                        "correct": is_correct,
                    }
                )
            n_gens_remaining -= generation_config["num_return_sequences"]
                
            if n_gens_remaining % 10 < 2:
                with open(filename, "w") as wf:
                    json.dump(results, wf, indent=4)

        with open(filename, "w") as wf:
            json.dump(results, wf, indent=4)
        return results


def load_model(model_name, quantize=True):
    config = AutoConfig.from_pretrained(model_name)

    bnb_config = None
    if quantize and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
    )
    model.eval()

    for param in model.parameters():
        param._requires_grad = False

    return model


def run():
    models = [
        # ("Qwen/Qwen2.5-7B-Instruct", 6),
        # ("Qwen/Qwen2.5-14B-Instruct", 6),
        ("Qwen/Qwen2.5-32B-Instruct", 6),

        # ("allenai/OLMo-2-1124-13B-Instruct", 6),
        # ("meta-llama/Llama-3.1-8B-Instruct", 6),
        # ("mistralai/Ministral-8B-Instruct-2410", 6),
        # ("google/gemma-2-9b-it", 6),
        # ("allenai/OLMo-2-1124-7B-Instruct", 6 ),
        # ("mistralai/Mistral-7B-Instruct-v0.3", 6),
        # ("Qwen/CodeQwen1.5-7B-Chat", 6),
        # ("deepseek-ai/deepseek-coder-6.7b-instruct", 6),
        # ("meta-llama/CodeLlama-13b-Instruct-hf", 6),
        # ("google/codegemma-7b-it", 6),
        # the below models r so bad
        # ("meta-llama/Llama-2-13b-chat-hf", 6),
        # ("meta-llama/Llama-3.2-3B-Instruct", 20),
        # ("meta-llama/Llama-3.2-1B-Instruct", 28),
    ]
    n_samples = 200
    k_vals = [5, 10, 30]
    # k_vals = [5, 10, 15]
    t_vals = [2, 4, 6]
    # t_vals = [2, 3, 4]
    N_vals = [1, 10, 16, 24]
    # N_vals = [1, 6, 10, 16, 24, 32]
    temperature = sys.argv[1]

    for (model_name, batch_size) in models:
        model = load_model(model_name)
        experiment = Experiment(model, model_name, batch_size, ("none", {}), n_samples=n_samples, temperature=float(temperature))
        for k in k_vals:
            for t in t_vals:
                for N in N_vals:
                    results = experiment.run_experiment(k, N, t)
        del model

if __name__ == "__main__":
    run()
