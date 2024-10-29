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


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

def get_code_str_from_tree_node(ast_node, og_code):
    if 'lineno' not in dir(ast_node):
        print(f"Cannot get codestr for node {ast_node}")
        return None
    code_lines = og_code.splitlines(keepends=True)
    start_index = sum(len(line) for line in code_lines[:ast_node.lineno-1]) + ast_node.col_offset
    end_index = sum(len(line) for line in code_lines[:ast_node.end_lineno-1]) + ast_node.end_col_offset
    return (start_index, end_index, og_code[start_index:end_index])



## NOTES
# could potentially use https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.ExponentialDecayLengthPenalty for max limit
# dont know yet how to encourage longer cot though... elegantly.
# options:
# temperature sampling
# random seed choice
# fs example selection

def fs_basic(
    model,
    tokenizer,
    get_batch,
    make_queries,
    examples,
    get_prediction_and_correctness,
    max_batch_size,
    stop_strings,
):
    outputs = []
    ex_idx = 0
    while ex_idx < len(examples):
        exs = get_batch(examples, ex_idx, max_batch_size)
        queries = make_queries(exs)
        input_ids = tokenizer(
            queries, padding=True, truncation=True, max_length=2048, return_tensors="pt"
        ).to(device)
        model_output = model.generate(
            **input_ids,
            return_dict_in_generate=True,
            max_new_tokens=1024,
            tokenizer=tokenizer,
            stop_strings=stop_strings,
            pad_token_id=tokenizer.eos_token_id,
        )
        model_predictions = tokenizer.batch_decode(
            model_output.sequences[:, input_ids.input_ids.shape[-1] :],
            skip_special_tokens=True,
        )
        for batch_idx, model_prediction in enumerate(model_predictions):
            predicted_answer = model_prediction.strip()
            for ss in stop_strings:
                if predicted_answer.endswith(ss):
                    predicted_answer = predicted_answer[: -len(ss)].strip()
                    break
            is_correct, predicted_answer = get_prediction_and_correctness(predicted_answer, exs, batch_idx)
            
            outputs.append(
                {
                    "input_example": exs[batch_idx],
                    "query": queries[batch_idx],
                    "model_generation": model_prediction,
                    "total_compute_tokens": torch.sum(
                        model_output.sequences[batch_idx] != tokenizer.pad_token_id
                    ).item(),
                    "generated_tokens": torch.sum(
                        model_output.sequences[
                            batch_idx, input_ids.input_ids.shape[-1] :
                        ]
                        != tokenizer.pad_token_id
                    ).item(),
                    "answer": predicted_answer,
                    "correct": is_correct,
                }
            )

        ex_idx += max_batch_size
    return outputs


def fs_cot(
    model,
    tokenizer,
    get_batch,
    make_queries,
    examples,
    get_prediction_and_correctness,
    max_batch_size,
    stop_strings,
    num_samples,
    temperature,
):
    outputs = []
    ex_idx = 0
    while ex_idx < len(examples):
        exs = get_batch(examples, ex_idx, max_batch_size)
        queries = make_queries(exs)
        input_ids = tokenizer(
            queries, padding=True, truncation=True, max_length=2048, return_tensors="pt"
        ).to(device)
        model_output = model.generate(
            **input_ids,
            return_dict_in_generate=True,
            max_new_tokens=2048,
            tokenizer=tokenizer,
            stop_strings=stop_strings,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        model_predictions = tokenizer.batch_decode(
            model_output.sequences[:, input_ids.input_ids.shape[-1] :],
            skip_special_tokens=True,
        )
        for batch_idx, input_ex in enumerate(exs):
            example_output = {
                "input_example": exs[batch_idx],
                "query": queries[batch_idx],
                "generations": []
            }
            for ex_generation in range(num_samples):
                output_idx = batch_idx * num_samples + ex_generation
                model_prediction = model_predictions[output_idx]
                is_correct, predicted_answer = get_prediction_and_correctness(model_prediction, exs, batch_idx)
                num_generated_tokens = torch.sum(
                    model_output.sequences[output_idx, input_ids.input_ids.shape[-1] :]
                    != tokenizer.pad_token_id
                ).item()
                example_output["generations"].append(
                    {
                        "model_generation": model_prediction,
                        "total_compute_tokens": torch.sum(
                            model_output.sequences[output_idx] != tokenizer.pad_token_id
                        ).item(),
                        "generated_tokens": num_generated_tokens,
                        "answer": predicted_answer,
                        "correct": is_correct,
                    }
                )
            outputs.append(example_output)

        ex_idx += max_batch_size
    return outputs

def load_model(model_name):

    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", truncation_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer
    
def run_experiment(model, modelname, domain, max_batch_size, num_samples, temperature, num_ex=200):

    if domain == "trivia_qa":
        examples = load_dataset(
            "mandarjoshi/trivia_qa", "rc", split="validation"
        ).select(
            range(5, 5 + num_ex)
        )  # used the first few to construct the prompts
        from prompts import trivia_basic_prompt, trivia_cot_prompt

        def get_batch(examples, ex_idx, max_batch_size):
            exs = []
            for key, value in examples.items():
                for batch_idx in range(ex_idx, min(len(examples), ex_idx + max_batch_size)):
                    while batch_idx-ex_idx >= len(exs): exs.append({})
                    exs[batch_idx-ex_idx][key] = value[ex_idx]
            return exs
        make_basic_queries = lambda exs: [
            trivia_basic_prompt.format(question=ex['question']) for ex in exs
        ]
        make_cot_queries = lambda exs: [
            trivia_cot_prompt.format(question=ex['question']) for ex in exs
        ]
        basic_stop_strings = ["\nQuestion:"]
        cot_stop_strings = ["\nQuestion:"]

        def get_prediction_and_correctness_basic(prediction, exs, batch_idx):
            is_correct = re.sub(
                    r"[^\w\s]", "", prediction.lower()
                ) in exs[batch_idx]["answer"]["normalized_aliases"]
            predicted_answer = prediction.lower()
            return is_correct, predicted_answer

        def get_prediction_and_correctness_cot(prediction, exs, batch_idx):
            is_correct = False
            predicted_answer = re.search(r'So the final answer is: (.+)', prediction.strip())
            if predicted_answer:
                predicted_answer = predicted_answer.group(1).strip().rstrip(".")
                is_correct = re.sub(
                    r"[^\w\s]", "", prediction.lower()
                ) in exs[batch_idx]["answer"]["normalized_aliases"]
            return is_correct, predicted_answer
    if domain in {"indexing", "idx_management", "arrayworld"}:
        from prompts import code_solving_insn

        if domain == "indexing":
            from prompts import indexing_examples as fs_examples
            from prompts import cot_query_template

            examples = json.load(open("data/indexing_array_N20.json"))
            assert_regex = r"(^[\s\S]*)assert answer == "
        if domain == "idx_management":
            from prompts import idx_management_examples as fs_examples
            from prompts import idx_management_cot_query_template as cot_query_template

            examples = json.load(open("data/idx_management_N20.json"))
            assert_regex = r"(^[\s\S]*)assert idx == "
        if domain == "arrayworld":
            from prompts import array_world_examples as fs_examples
            from prompts import cot_query_template

            examples = json.load(open("data/uniformed_arrayworld_N20.json"))
            assert_regex = r"(^[\s\S]*)assert answer == "

        code_solving_prompt = code_solving_insn + "\n\n".join(
            f"```\n{code.strip()}\n```" for code in fs_examples
        )
        get_batch = lambda examples, ex_idx, max_batch_size: examples[ex_idx : min(len(examples), ex_idx + max_batch_size)]

        def make_basic_queries(exs):
            queries = []
            for ex in exs:
                code_assert_prefix = (
                    re.search(assert_regex, ex["code"], re.MULTILINE).group(0).strip()
                )
                query = code_solving_prompt + f"\n\n```\n{code_assert_prefix}"
                queries.append(query)
            return queries

        make_cot_queries = lambda exs: [
            cot_query_template.format(
                code=re.search(assert_regex, ex["code"], re.MULTILINE).group(0) + "??"
            )
            for ex in exs
        ]
        basic_stop_strings = ["```"]
        cot_stop_strings = ["[/ANSWER]"]

        def get_prediction_and_correctness_basic(prediction, exs, batch_idx):
            assert_line = f"assert answer == {prediction.strip().rstrip('`')}"
            predicted_answer = None
            try:
                tree = ast.parse(assert_line.strip())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assert):
                        if isinstance(node.test.ops[0], ast.Eq):
                            answer_node = node.test.comparators[0]
                            (_, _, predicted_answer) = get_code_str_from_tree_node(answer_node, assert_line)
                            predicted_answer = predicted_answer.strip()
                            break
            except:
                pass
            is_correct = False
            try:
                is_correct = eval(exs[batch_idx]['true_answer']) == eval(predicted_answer)
            except: 
                pass

            return is_correct, predicted_answer

        def get_prediction_and_correctness_cot(prediction, exs, batch_idx):
            assert_line = re.search(r'\[ANSWER\](.+?)\[/ANSWER\]', prediction)
            predicted_answer = None
            if assert_line:
                assert_line = assert_line.group(1)
                try:
                    tree = ast.parse(assert_line.strip())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assert):
                            if isinstance(node.test.ops[0], ast.Eq):
                                answer_node = node.test.comparators[0]
                                (_, _, predicted_answer) = get_code_str_from_tree_node(answer_node, assert_line)
                                predicted_answer = predicted_answer.strip()
                                break
                except:
                    pass
            is_correct = False
            try:
                is_correct = (predicted_answer is not None) and (eval(exs[batch_idx]['true_answer']) == eval(predicted_answer))
            except: 
                pass

            return is_correct, predicted_answer       
    if domain == "compgap":
        from prompts import qa_basic_prompt, compgap_cot_prompt

        examples = json.load(open("data/bamboogle_prerelease.json"))
        get_batch = lambda examples, ex_idx, max_batch_size: examples[ex_idx : min(len(examples), ex_idx + max_batch_size)]
        make_basic_queries = lambda exs: [
            qa_basic_prompt.format(question=ex["Question"]) for ex in exs
        ]
        make_cot_queries = lambda exs: [
            compgap_cot_prompt.format(question=ex["Question"]) for ex in exs
        ]
        basic_stop_strings = ["\nQuestion:"]
        cot_stop_strings = ["\nQuestion:"]

        def get_prediction_and_correctness_basic(prediction, exs, batch_idx):
            is_correct = prediction.lower() == exs[batch_idx]["Answer"].lower()
            predicted_answer = prediction.lower()
            return is_correct, predicted_answer

        def get_prediction_and_correctness_cot(prediction, exs, batch_idx):
            is_correct = False
            predicted_answer = re.search(r'So the final answer is: (.+)', prediction.strip())
            if predicted_answer:
                predicted_answer = predicted_answer.group(1).strip().rstrip(".")
                is_correct = predicted_answer.lower() == exs[batch_idx]['Answer'].lower()
            return is_correct, predicted_answer

    examples = examples[:num_ex]
    total_ex = len(examples)

    print(" ====== FS BASIC GREEDY ===== ")
    output_filename = f"outputs/{domain}_fs_basic_{modelname}.json"
    if os.path.exists(output_filename):
        fs_basic_outputs = json.load(open(output_filename))
    else:
        fs_basic_outputs = fs_basic(
            model,
            tokenizer,
            get_batch,
            make_basic_queries,
            examples,
            get_prediction_and_correctness_basic,
            max_batch_size,
            basic_stop_strings,
        )
        with open(output_filename, "w") as wf:
            json.dump(fs_basic_outputs, wf, indent=4)
    fs_basic_correct = len([ex for ex in fs_basic_outputs if ex["correct"]])
    print(f"Correct: {fs_basic_correct} / {total_ex}")
    print(
        f"Avg tot tokens: {sum(op['total_compute_tokens'] for op in fs_basic_outputs) / total_ex:.2f}"
    )
    print(
        f"Avg gen tokens: {sum(op['generated_tokens'] for op in fs_basic_outputs) / total_ex:.2f}"
    )

    print(f" ====== FS COT TEMP {temperature}===== ")
    output_filename = f"outputs/{domain}_fs_cot_temp{temperature*100}_{modelname}.json"
    if os.path.exists(output_filename):
        fs_cot_outputs = json.load(open(output_filename))
    else:
        fs_cot_outputs = fs_cot(
            model,
            tokenizer,
            get_batch,
            make_cot_queries,
            examples,
            get_prediction_and_correctness_cot,
            max_batch_size,
            cot_stop_strings,
            num_samples,
            temperature
        )
        with open(output_filename, "w") as wf:
            json.dump(fs_cot_outputs, wf, indent=4)
    
    fs_cot_correct = len([ex for ex in fs_cot_outputs if any(gen["correct"] for gen in ex['generations'])])
    print(f"(pass@{num_samples}) Correct: {fs_cot_correct} / {total_ex}")
    got_correct_lengths = [generation['total_compute_tokens'] for op in fs_cot_outputs for generation in op['generations'] if generation['correct'] ]
    print("lengths of correct:", got_correct_lengths)
    got_incorrect_lengths = [generation['total_compute_tokens'] for op in fs_cot_outputs for generation in op['generations'] if not generation['correct'] ]
    print("lengths of incorrect:", got_incorrect_lengths)


batch_size = 12
num_ex = 200
num_samples = 5
temperatures = [0.6]
# min_new_token_lengths = [0, 100, 200, 300]
models = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
domains = ["trivia_qa", "compgap", "arrayworld", "indexing", "idx_management"]

for model in models:
    modelname = re.search(r"/(.+)", model).group(1)
    model, tokenizer = load_model(model)
    for domain in domains:
        for temperature in temperatures:
            run_experiment(model, modelname, domain, batch_size, num_samples, temperature, num_ex=num_ex)
