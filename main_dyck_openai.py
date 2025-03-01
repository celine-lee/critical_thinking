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
    parser.add_argument("--d_vals", nargs='+', default=[1, 2, 4]) # num diff operators
    parser.add_argument("--k_vals", nargs='+', default=[2, 5, 8]) # depth
    parser.add_argument("--N_vals", nargs='+', default=[8, 16, 24, 30])
    parser.add_argument("--models", nargs='+', default=["google/gemma-2-9b-it"])
    parser.add_argument("--disable_cot", action="store_true")
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

    dyck_task = DyckNTask()
    for model_name in args.models:
        generator = OpenAIGenerator(model_name, gen_kwargs, args.batch_size)
        experimentor = Experimenter(dyck_task, generator, n_samples_per, force_no_cot=args.disable_cot)

        for k in args.k_vals:
            for d in args.d_vals:
                for N in args.N_vals:
                    experimentor.run({"k": k, "d": d, "N": N})


if __name__ == "__main__":
    run(get_args())