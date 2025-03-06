from src.task import *
from src.experimentor import *
from src.generator import *

import itertools
import sys
import ipdb
import traceback

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_gens_per", type=int, default=1)
    parser.add_argument("--max_num_tokens", type=int, default=10000)
    parser.add_argument("--n_samples_per", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_vals", type=int, nargs='+') 
    parser.add_argument("--m_vals", type=int, nargs='+') 
    parser.add_argument("--k_vals", type=int, nargs='+') 
    parser.add_argument("--N_vals", type=int, nargs='+')
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--disable_cot", action="store_true")
    parser.add_argument("--generator", choices=['hf', 'openai', 'deepseek', 'vllm', 'together'])
    parser.add_argument("--task", choices=['dyck', 'array_idx', 'cruxeval', 'even_odd', 'navigate', 'bool', 'arith'])
    args = parser.parse_args()
    return args

def run(args):
    print(args.generator)
    gen_kwargs = {
        "max_new_tokens": args.max_num_tokens,
        "num_beams": args.num_beams, 
        "stop_strings": ["[/ANSWER]"],
        "num_return_sequences": args.num_gens_per,
        "temperature": args.temperature
    }
    n_samples_per = args.n_samples_per

    experimentor = Experimenter
    match args.task:
        case 'dyck':
            task = DyckNTask()
        case 'arith':
            task = ArithmeticTask()
        case 'array_idx':
            task = ArrayIdxTask()
        case 'bool':
            task = NestedBoolTask()
        case 'navigate':
            task = NavigateTask()
        case 'cruxeval':
            task = CRUXEvalTask()
            experimentor = CRUXEvalExperimenter

    match args.generator:
        case 'openai':
            generator = OpenAIGenerator
        case 'hf':
            generator = HFGenerator
        case 'deepseek':
            generator = DeepseekGenerator
        case 'vllm':
            generator = VLLMGenerator
        case 'together':
            generator = TogetherGenerator

    for model_name in args.models:
        # TODO add a check saying that if all experiments wanted are done, dont load the model
        model_generator = generator(model_name, gen_kwargs, args.batch_size)
        experimentor = experimentor(task, model_generator, n_samples_per, force_no_cot=args.disable_cot)
        if args.task == 'cruxeval':
            experimentor.run()
            model_generator.free_and_delete()
            continue

        keys = ["k", "N"]
        values = [args.k_vals, args.N_vals]
        if args.m_vals: 
            keys.append("m")
            values.append(args.m_vals)
        if args.d_vals: 
            keys.append("d")
            values.append(args.d_vals)

        all_dfa_configs = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        
        for dfa_config in all_dfa_configs:
            experimentor.run(dfa_config)
        
        model_generator.free_and_delete()

if __name__ == "__main__":
    run(get_args())
