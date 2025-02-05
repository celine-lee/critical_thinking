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

from llm_string_consts import *

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

foldername = "outputs"


class Experiment:
    def __init__(
        self,
        model,
        model_name,
        num_gens_per,
        n_samples,
        temperature,
        num_beams=1,
        max_new_tokens=2400,
        max_batch_size=6,
        disable_cot=False,
    ):
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
            self.filename = (
                f"{modelname}_T{self.temperature}_B{num_beams}_S{num_gens_per}"
            )
        else:
            self.filename = f"{modelname}_T{self.temperature}"
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.max_batch_size = max(1, max_batch_size // num_gens_per)
        self.disable_cot = disable_cot

    def extract_answers(self, input_ids, model_output):
        model_predictions = self.tokenizer.batch_decode(
            model_output.sequences, skip_special_tokens=True,
        )
        new_queries = []
        gens_need_augmenting = []
        extracted_answers = [None for _ in model_predictions]
        for gen_idx, model_prediction in enumerate(model_predictions):
            input_idx = gen_idx // self.num_gens_per
            query_len = len(
                self.tokenizer.decode(
                    input_ids.input_ids[input_idx], skip_special_tokens=True
                )
            )
            parsed_answer = None
            for parsed_answer in re.finditer(answer_regex, model_prediction):
                pass  # only get the last
            if parsed_answer is None or (
                (not self.disable_cot) and parsed_answer.start() < query_len
            ):
                gens_need_augmenting.append(gen_idx)
                new_queries.append(model_prediction + reprompt_string)
                num_generated_tokens = torch.sum(
                    model_output.sequences[gen_idx, input_ids.input_ids.shape[-1] :]
                    != self.tokenizer.pad_token_id
                ).item()
                answer = None
            else:
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

        if len(new_queries) == 0:
            return extracted_answers
        new_input_ids = self.tokenizer(
            new_queries,
            padding=True,
            truncation=True,
            max_length=self.max_new_tokens,
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
            model_prediction = self.tokenizer.decode(
                model_output.sequences[new_idx], skip_special_tokens=True
            )
            new_query_len = len(new_queries[new_idx])
            parsed_answer = None
            for parsed_answer in re.finditer(answer_regex, model_prediction):
                pass  # only get the last
            if parsed_answer is None:
                answer = None
            else:
                answer = parsed_answer.group(1).rstrip(" .")
            (_, prev_num_generated_tokens, _, prev_generated) = extracted_answers[
                orig_idx
            ]
            num_generated_tokens = torch.sum(
                model_output.sequences[new_idx, new_input_ids.input_ids.shape[-1] :]
                != self.tokenizer.pad_token_id
            ).item()
            total_compute_tokens = torch.sum(
                model_output.sequences[new_idx] != self.tokenizer.pad_token_id
            ).item()
            extracted_answers[orig_idx] = (
                answer,
                prev_num_generated_tokens + num_generated_tokens,
                total_compute_tokens,
                prev_generated + reprompt_string + model_prediction[new_query_len:],
            )
        return extracted_answers

    def get_generation_config(self):
        if self.temperature == 0.0:
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

    def run_experiment(self):
        subfolder = os.path.join(f"{foldername}{'_nocot' if self.disable_cot else ''}")
        os.makedirs(subfolder, exist_ok=True)
        filename = f"{subfolder}/{self.filename}.json"
        print(filename)
        examples = json.load(open("cruxeval_profiled.json"))
        results = []
        if os.path.exists(filename):
            results = json.load(open(filename))

        used_uids = set(ex["id"] for ex in results)
        idx = len(results)
        while idx < min(len(examples), self.n_samples):
            prompts = []
            batch = []
            while len(prompts) < self.max_batch_size:
                if idx >= len(examples): break
                example = examples[idx]
                idx += 1
                if example["id"] in used_uids:
                    continue
                prompt = make_prompt(
                    self.tokenizer,
                    example["code"],
                    example["input"],
                    include_starter=self.disable_cot,
                )
                prompts.append(prompt)
                batch.append(example)
            if len(prompts) == 0: continue
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

            for (
                gen_idx,
                (
                    pred_answer,
                    num_generated_tokens,
                    total_compute_tokens,
                    model_generation,
                ),
            ) in enumerate(extracted_answers):
                if pred_answer is None:
                    continue
                input_idx = gen_idx // generation_config["num_return_sequences"]
                example = batch[input_idx]
                try:
                    evaluated_pred = eval(pred_answer)
                    is_correct = evaluated_pred == eval(example["output"])
                    # If it was an assert with text for the printout, ignore the text.
                    if (
                        (not is_correct)
                        and type(evaluated_pred) == tuple
                        and len(evaluated_pred) == 2
                        and type(evaluated_pred[1]) == str
                    ):
                        print("TUPLE MAYBE?", example, pred_answer)
                        is_correct = evaluated_pred[0] == eval(example["output"])
                except: 
                    continue
                results.append(
                    {
                        "query": prompts[input_idx],
                        "model_generation": model_generation,
                        "total_compute_tokens": total_compute_tokens,
                        "generated_tokens": num_generated_tokens,
                        "pred_answer": pred_answer,
                        "true_answer": example["output"],
                        "correct": is_correct,
                        "id": example["id"],
                        "N": sum(example["line_execution_counts"].values()),
                        "k": example["ast_size"],
                        "l": len(example["code"].splitlines())
                    }
                )

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
    parser.add_argument("--models", nargs="+", default=["google/gemma-2-9b-it"])
    parser.add_argument("--disable_cot", action="store_true")
    args = parser.parse_args()
    return args


def run():
    args = get_args()

    for model_name in args.models:
        model = load_model(model_name)
        experiment = Experiment(
            model,
            model_name,
            args.num_gens_per,
            n_samples=800,
            temperature=args.temperature,
            num_beams=args.num_beams,
            disable_cot=args.disable_cot,
            max_batch_size=4 if "32B" in model_name else 8,
        )
        results = experiment.run_experiment()
        del model


if __name__ == "__main__":
    run()
