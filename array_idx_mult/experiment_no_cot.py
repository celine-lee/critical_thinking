
import json
import re
import sys
import os
from tqdm import tqdm
import numpy as np

import sys
import ipdb
import traceback

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

from generate_dfas import *
from llm_string_consts import *

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

foldername = "outputs_nocot"

class ExperimentNoCOT:
    def __init__(self, model, model_name, num_gens_per, n_samples, temperature, num_beams=1, max_new_tokens=50, max_batch_size=8):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.num_gens_per = num_gens_per
        self.temperature = temperature
        self.n_samples = n_samples
        modelname = re.search(r"/(.+)", model_name).group(1)
        if temperature > 0:
            self.filename = f"{modelname}_T{self.temperature}_B{num_beams}_S{num_gens_per}"
        else:
            self.filename = f"{modelname}_T{self.temperature}"
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.max_batch_size = max(1, max_batch_size // num_gens_per)

    def extract_answers(
        self, input_ids, model_output
    ):
        model_predictions = self.tokenizer.batch_decode(
            model_output.sequences, skip_special_tokens=True,
        )
        extracted_answers = [None for _ in model_predictions]
        for gen_idx, model_prediction in enumerate(model_predictions):
            input_idx = gen_idx // self.num_gens_per
            query_len = len(self.tokenizer.decode(input_ids.input_ids[input_idx], skip_special_tokens=True))
            parsed_answer = None
            for parsed_answer in re.finditer(r'pointer ==\s*(\d+)', model_prediction):
                pass # only get the last
            answer = None
            if parsed_answer is not None:
                answer = parsed_answer.group(1).rstrip(" .")
            num_generated_tokens = torch.sum(
                model_output.sequences[gen_idx, input_ids.input_ids.shape[-1] :]
                != self.tokenizer.pad_token_id
            ).item()
            total_compute_tokens = torch.sum(
                model_output.sequences[gen_idx] != self.tokenizer.pad_token_id
            ).item()
            extracted_answers[gen_idx] = (
                answer,
                num_generated_tokens,
                total_compute_tokens,
                model_prediction[query_len:],
            )

        return extracted_answers

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

    def run_experiment(self, k, m, N):
        subfolder = os.path.join(foldername, f"k{k}_m{m}_N{N}")
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print(filename)
        results = []
        if os.path.exists(filename): results = json.load(open(filename))

        n_gens_remaining = self.n_samples - len(results)
        while n_gens_remaining > 0:
            prompts = []
            true_answers = []
            num_wraps = []
            while len(prompts) < self.max_batch_size:
                turns, true_final_location, wrap_count = random_walk(k, m, N)
                prompt = make_prompt(self.tokenizer, k, m, turns, include_starter=True)
                prompts.append(prompt)
                true_answers.append(true_final_location)
                num_wraps.append(wrap_count)

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
            extracted_answers = self.extract_answers(input_ids, model_output)

            for gen_idx, (pred_answer, num_generated_tokens, total_compute_tokens, model_generation) in enumerate(extracted_answers):
                if pred_answer is None: continue
                input_idx = gen_idx // generation_config["num_return_sequences"]
                true_final_location = true_answers[input_idx]
                is_correct = pred_answer is not None and eval(pred_answer) == true_final_location
                results.append(
                    {
                        "query": prompts[input_idx],
                        "model_generation": model_generation,
                        "total_compute_tokens": total_compute_tokens,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": true_final_location,
                        "correct": is_correct,
                        "num_wraps": num_wraps[input_idx]
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

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_gens_per", type=int, default=1)
    parser.add_argument("--k_vals", nargs='+', default=[5, 9, 17])
    parser.add_argument("--m_vals", nargs='+', default=[1, 5, 9, 17])
    parser.add_argument("--N_vals", nargs='+', default=[1, 10, 16, 24])
    parser.add_argument("--models", nargs='+', default=["google/gemma-2-9b-it"])
    args = parser.parse_args()
    return args


def run():
    args = get_args()
    n_samples = 100

    for model_name in args.models:
        model = load_model(model_name)
        experiment = ExperimentNoCOT(model, model_name, args.num_gens_per, n_samples=n_samples, temperature=args.temperature, num_beams=args.num_beams)
        for k in args.k_vals:
            for m in args.m_vals:
                for N in args.N_vals:
                    results = experiment.run_experiment(k, m, N)
        del model

if __name__ == "__main__":
    run()
