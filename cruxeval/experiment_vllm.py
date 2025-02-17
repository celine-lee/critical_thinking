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

from vllm import LLM, SamplingParams
from transformers import  BitsAndBytesConfig

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
        disable_cot=False,
    ):
        self.model = model
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
        self.disable_cot = disable_cot

    def extract_answers(self, orig_prompts, outputs):
        new_queries = []
        gens_need_augmenting = []
        extracted_answers = [[None for _ in outputs[0]] for _ in outputs]
        for input_idx, model_predictions in enumerate(outputs):
            for gen_idx, model_prediction in enumerate(model_predictions):
                parsed_answer = None
                for parsed_answer in re.finditer(answer_regex, model_prediction):
                    pass  # only get the last
                if parsed_answer is None or (
                    (not self.disable_cot) and parsed_answer.start() < query_len
                ):
                    gens_need_augmenting.append((input_idx, gen_idx))
                    new_queries.append(orig_prompts[input_idx] + model_prediction + reprompt_string)
                    answer = None
                else:
                    answer = parsed_answer.group(1).rstrip(" .")
                num_generated_tokens = outputs[input_idx][gen_idx].num_generated_tokens
                extracted_answers[input_idx][gen_idx] = (
                    answer,
                    num_generated_tokens,
                    model_prediction,
                )

        if len(new_queries) == 0:
            return extracted_answers

        force_answer_gen_params = self.get_generation_config_final_answer()
        model_output = self.model.generate(new_queries, force_answer_gen_params)
        for new_idx, orig_idx in enumerate(gens_need_augmenting):
            (input_idx, gen_idx) = orig_idx
            model_prediction = model_output[new_idx][0].text
            parsed_answer = None
            for parsed_answer in re.finditer(answer_regex, model_prediction):
                pass  # only get the last
            if parsed_answer is None:
                answer = None
            else:
                answer = parsed_answer.group(1).rstrip(" .")
            num_generated_tokens = model_output[new_idx][0].num_generated_tokens
            (_, prev_num_generated_tokens, prev_generated) = extracted_answers[
                input_idx][gen_idx
            ]
            extracted_answers[input_idx][gen_idx] = (
                answer,
                prev_num_generated_tokens + num_generated_tokens,
                prev_generated + reprompt_string + model_prediction
            )
        return extracted_answers

    def get_generation_config(self):
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=stop_strings,
            n=self.num_gens_per,
        )

    def get_generation_config_final_answer(self):
        return SamplingParams(
            temperature=0.0,
            max_tokens=50,
            stop=stop_strings,
            n=1,
        )

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
        just_move_on_counter = 0
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
            generation_config = self.get_generation_config()
            model_output = self.model.generate(
                prompts,
                generation_config
            )
            extracted_answers = self.extract_answers(prompts, model_output)

            for (
                gen_idx,
                (
                    pred_answer,
                    num_generated_tokens,
                    model_generation,
                ),
            ) in enumerate(extracted_answers):
                if pred_answer is None:
                    just_move_on_counter += 1
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

            if just_move_on_counter > 30: break
        with open(filename, "w") as wf:
            json.dump(results, wf, indent=4)
        return results


def load_model(model_name, quantize=True):
    # bnb_config = None
    # if quantize and torch.cuda.is_available():
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #     )
    try:
        model = LLM(model_name, quantization="fp8")
        return model
    except:
        pass
    try:
        model = LLM(model_name, quantization="awq")
        return model
    except pass
    return None


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
    max_new_tokens = 6000
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
            max_batch_size=2,
            max_new_tokens=max_new_tokens
        )
        results = experiment.run_experiment()
        del model


if __name__ == "__main__":
    run()
