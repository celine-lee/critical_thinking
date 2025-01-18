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


from example_generator import random_dfa, do_run, generate_illegal_string, generate_hard_illegal_string, dfa_to_regex


temperature = sys.argv[1]
num_beams = sys.argv[2]
num_gens_per = sys.argv[3]
foldername = sys.argv[4]

system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""

query_template = """I have a lexer described by the regex r'{regex}'. According to the regex, is the following string valid? "{string}"
"""
if "cot" in foldername:
    generation_instruction = "Think step by step, then provide your final answer as True or False following this template: [ANSWER]\nYOUR ANSWER\n[/ANSWER]"
else:
    generation_instruction = "Provide your final answer as True or False following this template: [ANSWER]\nYOUR ANSWER\n[/ANSWER]"

stop_strings = ["[/ANSWER]"]

def prompt_with_chat_template(tokenizer, regex, string):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = query_template.format(regex=regex, string=string) + "\n"
    prompt += generation_instruction
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, regex, string):
    if tokenizer.chat_template: return prompt_with_chat_template(tokenizer, regex, string)
    prompt = system_instruction + "\n\n"
    prompt += query_template.format(regex=regex, string=string) + "\n"
    prompt += generation_instruction + "\n\n"
    return prompt

class Experiment:
    def __init__(self, model, model_name, num_gens_per, n_samples, temperature, num_beams=1, max_new_tokens=2400):
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
        self.max_batch_size = 4

    def extract_answers(
        self, input_ids, model_output
    ):
        reprompt_string = "[ANSWER]\n"
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
            for parsed_answer in re.finditer(r'\[ANSWER\]\s*(True|False)', model_prediction):
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
            for parsed_answer in re.finditer(r'\[ANSWER\]\s*(True|False)', model_prediction):
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

    def run_experiment(self, N, vocab, density):
        if self.temperature == 0.: return self.run_experiment_greedy(N, vocab, density)
        subfolder = os.path.join(foldername, f"N{N}_V{len(vocab)}_d{density:0.2f}")
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print(filename)
        results = []
        if os.path.exists(filename): results = json.load(open(filename))

        n_gens_remaining = self.n_samples - len(results)
        while n_gens_remaining > 0:
            edges, rip_node_view, actual_d = random_dfa(N, vocab, density)
            num_repeats_before_quit = 10
            while len(set(runs)) < num_gens_per // 2:
                run = do_run(rip_node_view, edges)
                if run is None: break
                runs.append(run)
                if runs.count(run) > num_repeats_before_quit: break
            if run is None: continue
            runs = list(set(runs))
            is_legal = [True for _ in runs]
            while len(set(runs)) < num_gens_per:
                illegal_run = generate_illegal_string(rip_node_view, vocab, max(len(x) for x in runs) + 2)
                runs.append(illegal_run)
                is_legal.append(False)

            resulting_regex = dfa_to_regex(rip_node_view, edges)
            for run_idx, run in enumerate(runs):
                prompt = make_prompt(self.tokenizer, resulting_regex, run)
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
                    is_correct = pred_answer is not None and eval(pred_answer) == is_legal[run_idx]
                    results.append(
                        {
                            "query": prompt,
                            "model_generation": model_generation,
                            "total_compute_tokens": total_compute_tokens,
                            "generated_tokens": num_generated_tokens,
                            "pred_answer": pred_answer,
                            "true_answer": is_legal[run_idx],
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


    def run_experiment_greedy(self, N, vocab, density):
        num_repeats_before_quit = 10
        subfolder = os.path.join(foldername, f"N{N}_V{len(vocab)}_d{density:0.2f}")
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print("greedy:", filename)
        results = []
        if os.path.exists(filename): results = json.load(open(filename))

        n_gens_remaining = self.n_samples - len(results)
        while n_gens_remaining > 0:
            prompts = []
            is_legal = []
            runs = []
            # generate a connected DFA
            while True:
                edges, node_view, actual_d = random_dfa(N, vocab, density)
                for _ in range(num_repeats_before_quit):
                    run = do_run(node_view, edges)
                    if run is not None: break
                if run is not None: break
            # print(edges, node_view, actual_d)
            # generate a few runs
            while len(set(runs)) < self.max_batch_size // 2:
                run = do_run(node_view, edges)
                if run is None: continue
                runs.append(run)
                if runs.count(run) > num_repeats_before_quit: 
                    break
            runs = list(set(runs))
            # print(runs)
            is_legal.extend([True for _ in runs])
            while len(set(runs)) < self.max_batch_size:
                illegal_run = generate_illegal_string(node_view, vocab, max(len(x) for x in runs) + 2)
                if illegal_run is None: continue
                runs.append(illegal_run)
                is_legal.append(False)
            # print(runs)
            resulting_regex = dfa_to_regex(node_view, edges)
            # print(resulting_regex)
            for run in set(runs):
                prompt = make_prompt(self.tokenizer, resulting_regex, run)
                prompts.append(prompt)

            input_ids = self.tokenizer(
                prompts[:self.max_batch_size],
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
                is_correct = pred_answer is not None and eval(pred_answer) == is_legal[gen_idx]
                results.append(
                    {
                        "query": prompts[gen_idx],
                        "model_generation": model_generation,
                        "total_compute_tokens": total_compute_tokens,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": is_legal[gen_idx],
                        "correct": is_correct,
                        "regex": resulting_regex
                    }
                )
                n_gens_remaining -= 1
            
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
    global num_gens_per
    global temperature
    global num_beams
    models = [
        "mistralai/Ministral-8B-Instruct-2410",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "allenai/OLMo-2-1124-13B-Instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
    ]
    n_samples = 50
    N_vals = [2, 3]
    vocab = ["a", "b", "c"]
    density = 0.8

    for model_name in models:
        model = load_model(model_name)
        experiment = Experiment(model, model_name, int(num_gens_per), n_samples=n_samples, temperature=float(temperature), num_beams=int(num_beams))
        for N in N_vals:
            results = experiment.run_experiment(N, vocab, density)
        del model

if __name__ == "__main__":
    run()
