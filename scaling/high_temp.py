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


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook


def get_code_str_from_tree_node(ast_node, og_code):
    if "lineno" not in dir(ast_node):
        print(f"Cannot get codestr for node {ast_node}")
        return None
    code_lines = og_code.splitlines(keepends=True)
    start_index = (
        sum(len(line) for line in code_lines[: ast_node.lineno - 1])
        + ast_node.col_offset
    )
    end_index = (
        sum(len(line) for line in code_lines[: ast_node.end_lineno - 1])
        + ast_node.end_col_offset
    )
    return (start_index, end_index, og_code[start_index:end_index])


eot_id = "<|eot_id|>"
start_header_id = "<|start_header_id|>"

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant.
{task_insn}

If it is helpful, first think step-by-step.<|eot_id|><|start_header_id|>user<|end_header_id|>

{task_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

code_solving_insn = """Based on the given Python code, complete the assert statement with the expected output when executing the code."""

qa_insn = """Answer the given question."""

foldername = "zeroshot_outputs"

from transformers import StoppingCriteria, StoppingCriteriaList


class StopStringCriteria(StoppingCriteria):
    # Necessary for beam search...
    def __init__(self, input_len, stop_strings, tokenizer):
        self.input_len = input_len
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode generated tokens and check for the stop string
        decoded_outputs = self.tokenizer.batch_decode(
            input_ids[:, self.input_len :], skip_special_tokens=True
        )
        return all(
            any(ss in decoded_output for ss in self.stop_strings)
            for decoded_output in decoded_outputs
        )


def get_generation_config(generation_config, input_ids_shape):
    generation_config["stopping_criteria"] = None
    if "num_beams" in generation_config:
        stop_criteria = StopStringCriteria(
            input_len=input_ids_shape[-1],
            stop_strings=generation_config["stop_strings"],
            tokenizer=generation_config["tokenizer"],
        )
        generation_config["stopping_criteria"] = StoppingCriteriaList([stop_criteria])
    return generation_config


def rank_generations(
    input_ids,
    num_gens_per_input,
    model_output,
    new_input_ids,
    answered_model_output,
    extracted_answers_and_indices,
    augmented_generation_idxes,
    tokenizer,
):
    rankings = [{} for _ in range(input_ids.input_ids.shape[0])]
    # model_output.scores: gen-length tuple of [batch size x vocab size]1
    # rank length-normalized logprobs of extracted answer
    for input_idx in range(input_ids.input_ids.shape[0]):
        sorted_scores = []
        for gen_idx in range(
            input_idx * num_gens_per_input, (input_idx + 1) * num_gens_per_input
        ):
            in_batch_gen_idx = gen_idx - input_idx * num_gens_per_input
            (answer, answer_indices, _) = extracted_answers_and_indices[gen_idx]
            if gen_idx in augmented_generation_idxes:
                scores_to_use = answered_model_output.scores
                output_ids_to_use = answered_model_output.sequences[
                    augmented_generation_idxes.index(gen_idx)
                ]
                input_ids_to_use = new_input_ids.input_ids[
                    augmented_generation_idxes.index(gen_idx)
                ]
            else:
                scores_to_use = model_output.scores
                output_ids_to_use = model_output.sequences[gen_idx]
                input_ids_to_use = input_ids.input_ids[input_idx]
            input_len = input_ids_to_use.shape[-1]
            answer_scores = [
                scores_to_use[idx - input_len][gen_idx].log_softmax(dim=-1)[tok_idx]
                for (idx, tok_idx) in zip(
                    range(*answer_indices),
                    output_ids_to_use[answer_indices[0] : answer_indices[1]],
                )
            ]
            answer_score = sum(answer_scores) / len(answer_scores)
            sorted_scores.append([in_batch_gen_idx, answer_score.item()])
        sorted_scores = sorted(sorted_scores, key=lambda x: x[1])
        rankings[input_idx]["len_norm_logprobs"] = sorted_scores
    return rankings


def get_token_indices(
    output_ids, answer_string, string_index_start, string_index_end, tokenizer, init_tok_offset=0, init_char_offset=0
):
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    start_tok_idx = init_tok_offset
    curr_offset = init_char_offset
    while start_tok_idx < len(output_tokens):
        output_tok = output_tokens[start_tok_idx]
        if output_tok in tokenizer.all_special_tokens:
            start_tok_idx += 1
            continue
        curr_offset += len(output_tok)
        if curr_offset >= string_index_start:
            break
        start_tok_idx += 1
    end_tok_idx = start_tok_idx + 1
    while answer_string not in tokenizer.decode(output_ids[start_tok_idx:end_tok_idx]):
        end_tok_idx += 1
        if end_tok_idx > len(output_ids): breakpoint()
    # print(f"extracted: {answer_string} -> {tokenizer.decode(output_ids[start_tok_idx:end_tok_idx])}")
    return (start_tok_idx, end_tok_idx)


def answer_generations(
    input_ids, batch_size, model_output, model, tokenizer, method="continue_so_answer_is"
):
    input_tok_len = input_ids.input_ids.shape[-1]
    input_char_lens = [len(tokenizer.decode(input_ids.input_ids[idx], skip_special_tokens=True)) for idx in range(input_ids.input_ids.shape[0])]
    model_predictions = tokenizer.batch_decode(
        model_output.sequences, skip_special_tokens=True,
    )
    model_generations = tokenizer.batch_decode(
        model_output.sequences[:, input_tok_len:],
        skip_special_tokens=True,
    )
    new_queries = []
    gens_need_augmenting = []
    extracted_answers_and_indices = [None for _ in model_predictions]
    for gen_idx, model_prediction in enumerate(model_predictions):
        input_idx = gen_idx // batch_size
        if "So the answer is " in model_prediction:
            answer = (
                re.search(r"So the answer is (.+)", model_prediction)
                .group(1)
                .rstrip(" .")
            )
            string_index_start = model_prediction.index(
                f"So the answer is {answer}"
            ) + len("So the answer is ")
            string_index_end = string_index_start + len(answer)
            output_ids = model_output.sequences[gen_idx]
            answer_indices = get_token_indices(
                model_output.sequences[gen_idx],
                answer,
                string_index_start,
                string_index_end,
                tokenizer,
                init_tok_offset=input_tok_len,
                init_char_offset=input_char_lens[input_idx]
            )
            extracted_answers_and_indices[gen_idx] = (
                answer,
                answer_indices,
                model_generations[gen_idx],
            )
            continue
        else:
            if model_prediction.rstrip().endswith(eot_id):
                model_prediction = model_prediction.rstrip()[: -len(eot_id)]
            continuation = model_prediction + " So the answer is "
            new_queries.append(continuation)
            gens_need_augmenting.append(gen_idx)
    if len(new_queries) == 0:
        return None, None, None, extracted_answers
    new_input_ids = tokenizer(
        new_queries,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
        return_offsets_mapping=True,
    ).to(device)
    answered_model_output = model.generate(
        return_dict_in_generate=True,
        input_ids=new_input_ids.input_ids,
        attention_mask=new_input_ids.attention_mask,
        temperature=None,
        top_p=None,
        output_scores=True,
        max_new_tokens=512,
        tokenizer=tokenizer,
        stop_strings=["\n", eot_id, start_header_id],
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_model_predictions = tokenizer.batch_decode(
        answered_model_output.sequences, skip_special_tokens=True,
    )
    new_model_generations = tokenizer.batch_decode(
        answered_model_output.sequences[
            :, new_input_ids.input_ids.shape[-1] :
        ],  # offset it by the original
        skip_special_tokens=True,
    )
    for new_gen_idx, orig_gen_idx in enumerate(gens_need_augmenting):
        answer = (
            re.search(r"So the answer is (.+)", new_model_predictions[new_gen_idx])
            .group(1)
            .rstrip(" .")
        )
        string_index_start = new_model_predictions[new_gen_idx].index(
            f"So the answer is {answer}"
        ) + len("So the answer is ")
        string_index_end = string_index_start + len(answer)
        answer_indices = get_token_indices(
            answered_model_output.sequences[new_gen_idx],
            answer,
            string_index_start,
            string_index_end,
            tokenizer,
            init_tok_offset=new_input_ids.input_ids.shape[-1],
            init_char_offset=len(tokenizer.decode(new_input_ids.input_ids[new_gen_idx], skip_special_tokens=True))
        )
        full_generation = (
            model_generations[orig_gen_idx]
            + " So the answer is "
            + new_model_generations[new_gen_idx]
        )
        extracted_answers_and_indices[orig_gen_idx] = (
            answer,
            answer_indices,
            full_generation,
        )

    return (
        new_input_ids,
        answered_model_output,
        gens_need_augmenting,
        extracted_answers_and_indices,
    )


def generate(
    model,
    tokenizer,
    get_batch,
    make_queries,
    examples,
    get_prediction_and_correctness,
    max_batch_size,
    generation_config,
    output_filename,
):
    num_samples = (
        generation_config["num_return_sequences"]
        if "num_return_sequences" in generation_config
        else 1
    )
    num_beams = (
        generation_config["num_beams"] if "num_beams" in generation_config else 1
    )
    max_batch_size = max_batch_size // (num_samples * num_beams)
    if max_batch_size == 0:
        max_batch_size = 1

    outputs = []
    if os.path.exists(output_filename):
        outputs = json.load(open(output_filename))
    ex_idx = len(outputs)
    pbar = tqdm(total=len(examples))
    pbar.update(ex_idx)
    while ex_idx < len(examples):
        exs = get_batch(ex_idx, max_batch_size)
        queries = make_queries(exs)
        input_ids = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)
        generation_config = get_generation_config(
            generation_config, input_ids.input_ids.shape
        )
        model_output = model.generate(
            input_ids=input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            **generation_config,
        )
        input_char_lens = [len(query) for query in queries]
        (
            new_input_ids,
            answered_model_output,
            augmented_generation_idxes,
            extracted_answers_and_indices,
        ) = answer_generations(input_ids, max_batch_size, model_output, model, tokenizer)
        rankings = rank_generations(
            input_ids,
            num_samples,
            model_output,
            new_input_ids,
            answered_model_output,
            extracted_answers_and_indices,
            augmented_generation_idxes,
            tokenizer,
        )
        for batch_idx, input_ex in enumerate(exs):
            example_output = {
                "input_example": input_ex,
                "query": queries[batch_idx],
                "generations": [],
                "ranking": rankings[batch_idx],
            }
            for ex_generation in range(num_samples):
                output_idx = batch_idx * num_samples + ex_generation
                (pred_answer, _, model_generation) = extracted_answers_and_indices[
                    output_idx
                ]
                is_correct, predicted_answer = get_prediction_and_correctness(
                    pred_answer, exs, batch_idx
                )
                num_generated_tokens = torch.sum(
                    model_output.sequences[output_idx, input_ids.input_ids.shape[-1] :]
                    != tokenizer.pad_token_id
                ).item()
                example_output["generations"].append(
                    {
                        "model_generation": model_generation,
                        "total_compute_tokens": torch.sum(
                            model_output.sequences[output_idx] != tokenizer.pad_token_id
                        ).item(),
                        "generated_tokens": num_generated_tokens,
                        "answer": predicted_answer,
                        "correct": is_correct,
                    }
                )
            outputs.append(example_output)
            if len(outputs) % 5 == 0:
                with open(output_filename, "w") as wf:
                    json.dump(outputs, wf, indent=4)

        ex_idx += max_batch_size
        pbar.update(max_batch_size)
    pbar.close()
    return outputs

# https://arxiv.org/pdf/2402.10200
def cot_decode(
    model,
    tokenizer,
    get_batch,
    make_queries,
    examples,
    get_prediction_and_correctness,
    max_batch_size,
    first_tok_generate_config,
    remainder_greedy_generate_config,
    output_filename,
):
    num_samples = first_tok_generate_config["num_return_sequences"]
    max_batch_size = max_batch_size // num_samples
    if max_batch_size == 0:
        max_batch_size = 1

    outputs = []
    if os.path.exists(output_filename):
        outputs = json.load(open(output_filename))
    ex_idx = len(outputs)
    pbar = tqdm(total=len(examples))
    pbar.update(ex_idx)
    while ex_idx < len(examples):
        exs = get_batch(ex_idx, max_batch_size)
        queries = make_queries(exs)
        input_ids = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)
        
        first_tok_model_output = model.generate(
            input_ids=input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            **first_tok_generate_config,
        )

        model_output = model.generate(
            input_ids=first_tok_model_output,
            **remainder_greedy_generate_config
        )
        (
            new_input_ids,
            answered_model_output,
            augmented_generation_idxes,
            extracted_answers_and_indices,
        ) = answer_generations(input_ids, max_batch_size, model_output, model, tokenizer)
        rankings = rank_generations(
            input_ids,
            num_samples,
            model_output,
            new_input_ids,
            answered_model_output,
            extracted_answers_and_indices,
            augmented_generation_idxes,
            tokenizer,
        )
        for batch_idx, input_ex in enumerate(exs):
            example_output = {
                "input_example": input_ex,
                "query": queries[batch_idx],
                "generations": [],
                "ranking": rankings[batch_idx],
            }
            for ex_generation in range(num_samples):
                output_idx = batch_idx * num_samples + ex_generation
                (pred_answer, _, model_generation) = extracted_answers_and_indices[
                    output_idx
                ]
                is_correct, predicted_answer = get_prediction_and_correctness(
                    pred_answer, exs, batch_idx
                )
                num_generated_tokens = torch.sum(
                    model_output.sequences[output_idx, input_ids.input_ids.shape[-1] :]
                    != tokenizer.pad_token_id
                ).item()
                example_output["generations"].append(
                    {
                        "model_generation": model_generation,
                        "total_compute_tokens": torch.sum(
                            model_output.sequences[output_idx] != tokenizer.pad_token_id
                        ).item(),
                        "generated_tokens": num_generated_tokens,
                        "answer": predicted_answer,
                        "correct": is_correct,
                    }
                )
            outputs.append(example_output)
            if len(outputs) % 5 == 0:
                with open(output_filename, "w") as wf:
                    json.dump(outputs, wf, indent=4)

        ex_idx += max_batch_size
        pbar.update(max_batch_size)
    pbar.close()
    return outputs


def load_model(model_name):

    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", truncation_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def run_experiment(
    model,
    tokenizer,
    modelname,
    domain,
    max_batch_size,
    num_samples,
    temperature,
    num_ex,
):
    if domain == "trivia_qa":
        examples = load_dataset(
            "mandarjoshi/trivia_qa", "rc", split="validation"
        ).select(
            range(5, 5 + num_ex)
        )  # used the first few to construct the prompts

        def get_batch(ex_idx, max_batch_size):
            exs = [{} for _ in range(min(max_batch_size, len(examples) - ex_idx))]
            batch = examples[ex_idx : min(len(examples), ex_idx + max_batch_size)]
            for key in ["question", "answer"]:
                for batch_idx, value in enumerate(batch[key]):
                    exs[batch_idx][key] = value
            return exs

        make_queries = lambda exs: [
            prompt_template.format(task_insn=qa_insn, task_input=ex["question"])
            for ex in exs
        ]

        def get_prediction_and_correctness(prediction, exs, batch_idx):
            normalized_answer = re.sub(r"[^\w\s]", "", prediction.lower())
            is_correct = (
                normalized_answer in exs[batch_idx]["answer"]["normalized_aliases"]
            )
            return is_correct, normalized_answer

    if domain in {"indexing", "idx_management", "arrayworld"}:

        if domain == "indexing":
            assert_regex = r"(^[\s\S]*)assert answer == "
            examples = json.load(open("data/indexing_array_N20.json"))
        if domain == "idx_management":
            assert_regex = r"(^[\s\S]*)assert idx == "
            examples = json.load(open("data/idx_management_N20.json"))
        if domain == "arrayworld":
            assert_regex = r"(^[\s\S]*)assert answer == "
            examples = json.load(open("data/uniformed_arrayworld_N20.json"))

        examples = examples[: min(len(examples), num_ex)]
        get_batch = lambda ex_idx, max_batch_size: examples[
            ex_idx : min(len(examples), ex_idx + max_batch_size)
        ]

        def make_queries(exs):
            queries = []
            for ex in exs:
                code_assert_prefix = (
                    re.search(assert_regex, ex["code"], re.MULTILINE).group(0).strip()
                    + " ??"
                )
                query = prompt_template.format(
                    task_insn=code_solving_insn,
                    task_input=f"\n```\n{code_assert_prefix}\n```",
                )
                queries.append(query)
            return queries

        def get_prediction_and_correctness(prediction, exs, batch_idx):
            is_correct = False
            predicted_answer = None
            try:
                eval(prediction)
                assert_line = f"assert answer == {prediction}"
            except:
                assert_line = f'assert answer == "{prediction.strip()}"'
            try:
                tree = ast.parse(assert_line.strip())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assert):
                        if isinstance(node.test.ops[0], ast.Eq):
                            answer_node = node.test.comparators[0]
                            (_, _, predicted_answer) = get_code_str_from_tree_node(
                                answer_node, assert_line
                            )
                            predicted_answer = predicted_answer.strip()
                            break
            except:
                return False, None
            is_correct = (predicted_answer is not None) and (
                eval(exs[batch_idx]["true_answer"]) == eval(predicted_answer)
            )
            return is_correct, predicted_answer

    if domain == "compgap":
        examples = json.load(open("data/bamboogle_prerelease.json"))
        examples = examples[: min(len(examples), num_ex)]

        get_batch = lambda ex_idx, max_batch_size: examples[
            ex_idx : min(len(examples), ex_idx + max_batch_size)
        ]
        make_queries = lambda exs: [
            prompt_template.format(task_insn=qa_insn, task_input=ex["Question"])
            for ex in exs
        ]

        def get_prediction_and_correctness(prediction, exs, batch_idx):
            normalized_answer = re.sub(r"[^\w\s]", "", prediction.lower())
            is_correct = normalized_answer == exs[batch_idx]["Answer"].lower()
            return is_correct, normalized_answer

    stop_strings = [eot_id, start_header_id]
    total_ex = len(examples)

    experiments = []

    # print(" ====== GREEDY ===== ")
    experiments.append(
        (
            f"{foldername}/{domain}_greedy_{modelname}.json",
            max_batch_size,
            {
                "return_dict_in_generate": True,
                "temperature": None,
                "top_p": None,
                "output_scores": True,
                "max_new_tokens": 1200,
                "tokenizer": tokenizer,
                "stop_strings": stop_strings,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
            },
        )
    )

    # print(f" ====== MULTINOMIAL SAMPLING N={n_samples} TEMP {temperature}===== ")
    for n_samples in num_samples:
        experiments.append(
            (
                f"{foldername}/{domain}_sampling_temp{int(temperature*100)}_N{n_samples}_{modelname}.json",
                max_batch_size,
                {
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "max_new_tokens": 1200,
                    "tokenizer": tokenizer,
                    "stop_strings": stop_strings,
                    "num_return_sequences": n_samples,
                    "do_sample": True,
                    "temperature": temperature,
                    "pad_token_id": tokenizer.eos_token_id,
                },
            )
        )
        # print(f" ====== BEAM N={n_beams}===== ")
        n_beams = int(np.sqrt(n_samples))
        experiments.append(
            (
                f"{foldername}/{domain}_beams{n_beams}_{modelname}.json",
                1,
                {
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "max_new_tokens": 1200,
                    "tokenizer": tokenizer,
                    "stop_strings": stop_strings,
                    "num_return_sequences": n_beams,
                    "do_sample": False,
                    "num_beams": n_beams,
                    "top_p": None,
                    "temperature": None,
                    "pad_token_id": tokenizer.eos_token_id,
                },
            )
        )

    for output_filename, batch_size, generate_config in experiments:
        outputs = generate(
            model,
            tokenizer,
            get_batch,
            make_queries,
            examples,
            get_prediction_and_correctness,
            batch_size,
            generate_config,
            output_filename,
        )
        with open(output_filename, "w") as wf:
            json.dump(outputs, wf, indent=4)

    # COT decoding
    output_filename = f"{foldername}/{domain}_cotdecoding{n_samples}_temp{int(temperature*100)}_{modelname}.json"
    first_tok_generate_config = {
        "max_new_tokens": 1,
        "tokenizer": tokenizer,
        "do_sample": True,
        "num_return_sequences": n_samples,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
    }
    remainder_greedy_generate_config = {
        "return_dict_in_generate": True,
        "temperature": None,
        "top_p": None,
        "output_scores": True,
        "max_new_tokens": 1200,
        "tokenizer": tokenizer,
        "stop_strings": stop_strings,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    outputs = cot_decode(
        model,
        tokenizer,
        get_batch,
        make_queries,
        examples,
        get_prediction_and_correctness,
        max_batch_size,
        first_tok_generate_config,
        remainder_greedy_generate_config,
        output_filename,
    )
    with open(output_filename, "w") as wf:
        json.dump(outputs, wf, indent=4)


def main():

    num_ex = 100
    num_samples = [4, 9, 16]
    temperature = 0.7

    models = [
        ("meta-llama/Llama-3.2-1B-Instruct", 24),
        ("meta-llama/Llama-3.2-3B-Instruct", 16),
        ("meta-llama/Llama-3.1-8B-Instruct", 8),
    ]
    domains = [
        "indexing",
        "idx_management",
        "trivia_qa",
        "compgap",
        "arrayworld",
    ]

    for (model, batch_size) in models:
        modelname = re.search(r"/(.+)", model).group(1)
        print(f"===== {modelname} ====")
        model, tokenizer = load_model(model)
        for domain in domains:
            print(f"---- {domain} ----")
            run_experiment(
                model,
                tokenizer,
                modelname,
                domain,
                batch_size,
                num_samples,
                temperature,
                num_ex,
            )


if __name__ == "__main__":
    os.makedirs(foldername, exist_ok=True)
    main()
