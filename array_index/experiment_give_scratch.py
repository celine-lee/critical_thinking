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

from experiment import normalize_state, random_dfa, random_walk, system_instruction, query_template, stop_strings



def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

foldername = "outputs_scratch_given"
response_starter = "[ANSWER]\npointer == "

stop_strings = ["[/ANSWER]"]

def make_query(states, edges, turns):
    query = query_template.format(k=len(states), k_plus_one=len(states)+1, k_minus_1=len(states)-1, sequence="\n".join(turns)) 
    query += "\nProvide your answer following this template: [ANSWER]\npointer == YOUR ANSWER\n[/ANSWER]"
    return query

def prompt_with_chat_template(tokenizer, length_control_metadata, states, edges, turns):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = make_query(states, edges, turns)

    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + length_control_metadata["pad_sequence"] * length_control_metadata["num_repeats"] + response_starter
    
def make_prompt(tokenizer, length_control_metadata, states, edges, turns):
    if tokenizer.chat_template: return prompt_with_chat_template(tokenizer, length_control_metadata, states, edges, turns)
    prompt = system_instruction + "\n\n" + make_query(states, edges, turns, length_control_metadata)
    prompt += length_control_metadata["pad_sequence"] * length_control_metadata["num_repeats"] + response_starter
    return prompt 

class Experiment:
    def __init__(self, model, model_name, max_batch_size, length_control_metadata, n_samples, max_new_tokens=2400):
        self.model = model
        self.modelname = re.search(r"/(.+)", model_name).group(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_batch_size = max_batch_size
        self.n_samples = n_samples
        self.length_control_metadata = length_control_metadata
        self.filename = f"{self.modelname}_{self.length_control_metadata['pad_sequence']}x{self.length_control_metadata['num_repeats']}"
        self.max_new_tokens = max_new_tokens

    def extract_answers(
        self, input_ids, model_output
    ):
        model_predictions = self.tokenizer.batch_decode(
            model_output.sequences, skip_special_tokens=True,
        )
        extracted_answers_and_indices = [None for _ in model_predictions]
        for gen_idx, model_prediction in enumerate(model_predictions):
            query_len = len(self.tokenizer.decode(input_ids.input_ids[gen_idx], skip_special_tokens=True))
            parsed_answer = None
            for parsed_answer in re.finditer(r'pointer ==\s*(\d+)', model_prediction):
                pass # only get the last
            if parsed_answer is None or (parsed_answer.start() < query_len - len(response_starter)): 
                breakpoint()
                continue
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

        return extracted_answers_and_indices

    def get_generation_config(self):
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
            "top_k": None,
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
            input_batch = []
            true_final_locations = []
            while len(input_batch) < self.max_batch_size:
                states, edges = random_dfa(k, t)
                turns, true_final_location = random_walk(edges, N)
                true_final_locations.append(true_final_location)

                prompt = make_prompt(self.tokenizer, self.length_control_metadata, states, edges, turns)
                input_batch.append(prompt)

            input_ids = self.tokenizer(
                input_batch,
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
                if prediction is None: continue
                (pred_answer, _, num_generated_tokens, total_compute_tokens, model_generation) = prediction
                is_correct = pred_answer is not None and eval(pred_answer) == true_final_locations[gen_idx]
                results.append(
                    {
                        "query": input_batch[gen_idx],
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
    models = [
        ("Qwen/Qwen2.5-7B-Instruct", 6),
        ("Qwen/Qwen2.5-14B-Instruct", 6),
        ("Qwen/Qwen2.5-32B-Instruct", 6),
        ("meta-llama/Llama-3.1-8B-Instruct", 6),
        ("mistralai/Ministral-8B-Instruct-2410", 6),
        ("google/gemma-2-9b-it", 6),
        ("allenai/OLMo-2-1124-7B-Instruct", 6 ),
        ("allenai/OLMo-2-1124-13B-Instruct", 6),

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
    n_samples = 100
    k_vals = [5, 10, 30]
    # k_vals = [5, 10, 15]
    t_vals = [2, 4, 6]
    # t_vals = [2, 3, 4]
    N_vals = [1, 10, 16, 24]
    # N_vals = [1, 6, 10, 16, 24, 32]
    pad_sequences = {" ": [0, 1000, 10000], "thinking...": [0, 100, 500]}

    for pad_sequence, num_repeats in pad_sequences.items():
        for (model_name, batch_size) in models:
            model = load_model(model_name)
            for n_repeats in num_repeats:
                experiment = Experiment(model, model_name, batch_size, {"pad_sequence": pad_sequence , "num_repeats": n_repeats}, n_samples=n_samples)
                for k in k_vals:
                    for t in t_vals:
                        for N in N_vals:
                            results = experiment.run_experiment(k, N, t)
            del model

if __name__ == "__main__":
    run()
