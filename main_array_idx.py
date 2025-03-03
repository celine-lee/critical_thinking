from src.task import *
from src.experimentor import *
from src.generator import *


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
    parser.add_argument("--max_num_tokens", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k_vals", type=int, nargs='+', default=[5, 9, 17])
    parser.add_argument("--m_vals", type=int, nargs='+', default=[1, 5, 9, 17])
    parser.add_argument("--N_vals", type=int, nargs='+', default=[1, 10, 16, 24])
    parser.add_argument("--models", nargs='+', default=["google/gemma-2-9b-it"])
    parser.add_argument("--disable_cot", action="store_true")
    parser.add_argument("--generator", type=str, default='hf')
    args = parser.parse_args()
    return args

def run(args):
    gen_kwargs = {
        "max_new_tokens": args.max_num_tokens,
        "num_beams": args.num_beams, 
        "stop_strings": ["[/ANSWER]"],
        "num_return_sequences": args.num_gens_per,
        "temperature": args.temperature
    }
    n_samples_per = 100

    task = ArrayIdxTask()
    for model_name in args.models:
        if 'openai' == args.generator:
            generator = OpenAIGenerator(model_name, gen_kwargs, args.batch_size)
        elif 'hf' == args.generator:
            generator = HFGenerator(model_name, gen_kwargs, args.batch_size)
        else: assert False, f"{args.generator} not supported yet"
        experimentor = Experimenter(task, generator, n_samples_per, force_no_cot=args.disable_cot)

        for k in args.k_vals:
            for m in args.m_vals:
                for N in args.N_vals:
                    experimentor.run({"k": k, "m": m, "N": N})



if __name__ == "__main__":
    run(get_args())