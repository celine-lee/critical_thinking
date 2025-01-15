import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
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

generation_instruction = "Provide your final answer following this template: [ANSWER]\npointer == YOUR ANSWER\n[/ANSWER]"

stop_strings = ["[/ANSWER]"]

def prompt_with_chat_template(tokenizer, states, edges, turns):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = query_template.format(k=len(states), k_plus_one=len(states)+1, k_minus_1=len(states)-1, sequence="\n".join(turns)) + "\n"
    prompt += generation_instruction
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, states, edges, turns):
    if tokenizer.chat_template: return prompt_with_chat_template(tokenizer, states, edges, turns)
    prompt = system_instruction + "\n\n"
    prompt += query_template.format(k=len(states), k_plus_one=len(states)+1, k_minus_1=len(states)-1, sequence="\n".join(turns)) + "\n"
    prompt += generation_instruction + "\n\n"
    return prompt

class Experiment:
    def __init__(self, model, model_name, num_gens_per, n_samples, temperature, num_beams=3, max_new_tokens=2400):
        self.model = model
        self.modelname = re.search(r"/(.+)", model_name).group(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.num_gens_per = num_gens_per
        self.temperature = temperature
        self.n_samples = n_samples
        if temperature > 0:
            self.filename = f"{self.modelname}_T{self.temperature}_B{num_beams}_S{num_gens_per}"
        else:
            self.filename = f"{self.modelname}_T{self.temperature}"
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

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
            else:
                answer = parsed_answer.group(1).rstrip(" .")
                string_index_start = parsed_answer.start() + parsed_answer.group(0).index(answer)
                string_index_end = string_index_start + len(answer)
            num_generated_tokens = torch.sum(
                model_output.sequences[gen_idx, input_ids.input_ids.shape[-1] :]
                != self.tokenizer.pad_token_id
            ).item()
            total_compute_tokens = torch.sum(
                model_output.sequences[gen_idx] != self.tokenizer.pad_token_id
            ).item()
            extracted_answers_and_indices[gen_idx] = (
                answer,
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
            else:
                answer = parsed_answer.group(1).rstrip(" .")
                string_index_start = parsed_answer.start() + parsed_answer.group(0).index(answer)
                string_index_end = string_index_start + len(answer)
            (_, prev_num_generated_tokens, _, prev_generated) = extracted_answers_and_indices[orig_idx]
            num_generated_tokens = torch.sum(
                model_output.sequences[new_idx, new_input_ids.input_ids.shape[-1] :]
                != self.tokenizer.pad_token_id
            ).item()
            total_compute_tokens = torch.sum(
                model_output.sequences[new_idx] != self.tokenizer.pad_token_id
            ).item()
            extracted_answers_and_indices[orig_idx] = (
                answer,
                prev_num_generated_tokens + num_generated_tokens, # TODO include the reprompt string...
                total_compute_tokens,
                prev_generated + reprompt_string + model_prediction[new_query_len:],
            )
        return extracted_answers_and_indices

    def get_generation_config(self):
        if self.temperature == 0.:
            return {
                "return_dict_in_generate": True,
                "output_scores": True,
                "max_new_tokens": self.max_new_tokens,
                "no_repeat_ngram_size": 0, 
                "num_beams": self.num_beams,
                "tokenizer": self.tokenizer,
                "stop_strings": stop_strings,
                "num_return_sequences": 1,
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
        return {
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": self.max_new_tokens,
            "no_repeat_ngram_size": 0, 
            "tokenizer": self.tokenizer,
            "stop_strings": stop_strings,
            "num_return_sequences": self.num_gens_per,
            "do_sample": True,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_beams": self.num_beams,
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
            "top_k": None,
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
            # Reconstruct the decoded text to align tokenization and character-level offsets
            decoded_token = self.tokenizer.convert_tokens_to_string([output_tok])
            token_start_offset = curr_offset
            token_end_offset = curr_offset + len(decoded_token)

            # Check if this token contributes to or overlaps the target start index
            if token_start_offset <= string_index_start < token_end_offset:
                break
            # Update current character offset
            curr_offset = token_end_offset

        end_tok_idx = start_tok_idx + 1
        while answer_string not in self.tokenizer.decode(output_ids[start_tok_idx:end_tok_idx], skip_special_tokens=True):
            end_tok_idx += 1
            if end_tok_idx > len(output_ids): breakpoint()
        return (start_tok_idx, end_tok_idx)

    def run_experiment(self, k, N, t):
        if self.temperature == 0.: return self.run_experiment_greedy(k, N, t)
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

            prompt = make_prompt(self.tokenizer, states, edges, turns)
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


    def run_experiment_greedy(self, k, N, t):
        subfolder = os.path.join(foldername, f"k{k}_N{N}_t{t}")
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print("greedy:", filename)
        results = []
        if os.path.exists(filename): results = json.load(open(filename))

        n_gens_remaining = self.n_samples - len(results)
        while n_gens_remaining > 0:
            prompts = []
            true_final_locations = []
            while len(prompts) < self.num_gens_per:
                states, edges = random_dfa(k, t)
                turns, true_final_location = random_walk(edges, N)
                true_final_locations.append(true_final_location)

                prompt = make_prompt(self.tokenizer, states, edges, turns)
                prompts.append(prompt)

            input_ids = self.tokenizer(
                prompts,
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

            for gen_idx, prediction in enumerate(extracted_answers_and_indices):
                (pred_answer, num_generated_tokens, total_compute_tokens, model_generation) = prediction
                is_correct = pred_answer is not None and eval(pred_answer) == true_final_locations[gen_idx]
                results.append(
                    {
                        "query": prompts[gen_idx],
                        "model_generation": model_generation,
                        "total_compute_tokens": total_compute_tokens,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": true_final_locations[gen_idx],
                        "correct": is_correct,
                    }
                )
                n_gens_remaining -= 1
                
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
    temperature = sys.argv[1]
    num_beams = sys.argv[2]
    num_gens_per = sys.argv[3]
    models = [
        ("Qwen/Qwen2.5-7B-Instruct", num_gens_per),
        ("Qwen/Qwen2.5-14B-Instruct", num_gens_per),
        ("Qwen/Qwen2.5-32B-Instruct", num_gens_per),

        # ("mistralai/Ministral-8B-Instruct-2410", num_gens_per),
        # ("meta-llama/Llama-3.1-8B-Instruct", num_gens_per),
        # ("google/gemma-2-9b-it", num_gens_per),

        # ("allenai/OLMo-2-1124-13B-Instruct", num_gens_per),
        # ("allenai/OLMo-2-1124-7B-Instruct", 6 ),

        # ("mistralai/Mistral-7B-Instruct-v0.3", num_gens_per),
        # ("Qwen/CodeQwen1.5-7B-Chat", num_gens_per),
        # ("deepseek-ai/deepseek-coder-6.7b-instruct", num_gens_per),
        # ("meta-llama/CodeLlama-13b-Instruct-hf", num_gens_per),
        # ("google/codegemma-7b-it", num_gens_per),
        # the below models r so bad
        # ("meta-llama/Llama-2-13b-chat-hf", num_gens_per),
        # ("meta-llama/Llama-3.2-3B-Instruct", num_gens_per),
        # ("meta-llama/Llama-3.2-1B-Instruct", num_gens_per),
    ]
    n_samples = 100
    k_vals = [5, 10, 30]
    # k_vals = [5, 10, 15]
    t_vals = [2, 4, 6]
    # t_vals = [2, 3, 4]
    N_vals = [1, 10, 16, 24]
    # N_vals = [1, 6, 10, 16, 24, 32]

    for (model_name, num_gens_per) in models:
        model = load_model(model_name)
        experiment = Experiment(model, model_name, int(num_gens_per), n_samples=n_samples, temperature=float(temperature), num_beams=int(num_beams))
        for k in k_vals:
            for t in t_vals:
                for N in N_vals:
                    results = experiment.run_experiment(k, N, t)
        del model

if __name__ == "__main__":
    run()
