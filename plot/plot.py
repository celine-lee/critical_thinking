import json
import re
import os
import shutil
import random
import itertools
import pandas as pd

import matplotlib.pyplot as plt

from plot_utils import *

global compute_random
global foldername_parser
global dfa_factors_order

def get_args():
    global compute_random
    global foldername_parser
    global dfa_factors_order
    parser = global_parser()
    parser.add_argument("--d_vals", type=int, nargs='+') 
    parser.add_argument("--m_vals", type=int, nargs='+') 
    parser.add_argument("--k_vals", type=int, nargs='+') 
    parser.add_argument("--N_vals", type=int, nargs='+')
    parser.add_argument("--task", choices=['dyck', 'array_idx', 'even_odd', 'navigate', 'bool', 'arith'])

    args = parser.parse_args()
    match args.task:
        case 'dyck':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kdN
            dfa_factors_order = {"k": 0, "d": 1, "N": 2}
            output_folder = "dyck/outputs"
        case 'arith':
            compute_random = lambda factor_vals: 0.
            foldername_parser = parse_kmN
            dfa_factors_order = {"k": 0, "m": 1, "N": 2}
            output_folder = "arithmetic/outputs"
        case 'array_idx':
            compute_random = lambda factor_vals: 0.
            foldername_parser = parse_kmN
            dfa_factors_order = {"k": 0, "m": 1, "N": 2}
            output_folder = "array_idx_mult/outputs"
        case 'even_odd':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kmN
            dfa_factors_order = {"k": 0, "m": 1, "N": 2}
            output_folder = "even_odd_mult/outputs"
        case 'bool':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kN
            dfa_factors_order = {"k": 0, "N": 1}
            output_folder = "nested_boolean_expression/outputs"
        case 'navigate':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kdN
            dfa_factors_order = {"k": 0, "d": 1, "N": 2}
            output_folder = "navigate/outputs"

    args.foldername = os.path.join(
        f"{output_folder}_graphs_{args.n_buckets}buckets"
    )
    args.output_folder = output_folder

    return args

def parse_kdN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_d(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = int(parsed_experimentname.group(1))
    d = int(parsed_experimentname.group(2))
    N = int(parsed_experimentname.group(3))
    return {"k": k, "d": d, "N": N, "Model": modelname}

def parse_kmN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_m(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = int(parsed_experimentname.group(1))
    m = int(parsed_experimentname.group(2))
    N = int(parsed_experimentname.group(3))
    return {"k": k, "m": m, "N": N, "Model": modelname}

def parse_kN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = int(parsed_experimentname.group(1))
    N = int(parsed_experimentname.group(2))
    return {"k": k, "N": N, "Model": modelname}

if __name__ == "__main__":
    args = get_args()

    if args.delete_old and os.path.exists(args.foldername):
        shutil.rmtree(args.foldername)
    os.makedirs(args.foldername, exist_ok=True)

    plot_kwargs = {
        "n_buckets": args.n_buckets,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "num_gens": args.num_gens,
        "foldername": args.foldername,
        "only_meta": args.only_meta,
        "compute_random": compute_random,
        "plot_incorrect": False,
    }

    factor_names = ["k", "N"]
    values = [args.k_vals, args.N_vals]
    if args.m_vals: 
        factor_names.append("m")
        values.append(args.m_vals)
    if args.d_vals: 
        factor_names.append("d")
        values.append(args.d_vals)
    dfa_config_info = {factor_name: factor_name_values for factor_name, factor_name_values in zip(factor_names, values)}
    dfa_config_info["Model"] = args.models

    df = load_data(
        args.output_folder,
        dfa_config_info,
        foldername_parser,
        plot_kwargs,
    )
    df_nocot = load_data(
        args.output_folder + "_nocot",
        dfa_config_info,
        foldername_parser,
        plot_kwargs,
    )
    df = pd.concat([df, df_nocot])

    factor_to_peak_ttoks = {}
    for dfa_factor_name in factor_names:
        if dfa_factor_name == "Model": continue
        plot_length_generated(df, plot_kwargs, dfa_factor_name)
        factor_to_peak_ttoks[dfa_factor_name] = {}

    for factor_name in factor_names:
        if dfa_factor_name == "Model": continue
        factor_to_peak_ttoks[factor_name] = {}
    
    for free_factor_name in factor_names:
        if free_factor_name == "Model": continue
        other_factor_names = [factor_name for factor_name in factor_names if factor_name != free_factor_name]
        other_factor_values = [dfa_config_info[ofn] for ofn in other_factor_names]
        set_factor_combos = [dict(zip(other_factor_names, combination)) for combination in itertools.product(*other_factor_values)]
        for set_factors in set_factor_combos:
            for modelname in args.models:
                ptt_data = plot_correctness_by_ttoks_isolate_factor(
                    df, set_factors | {"Model": modelname}, free_factor_name, plot_kwargs
                )
                if ptt_data is None: continue
                ffn_to_ptts, _, _= ptt_data
                if ffn_to_ptts:
                    dfa_config = [None for _ in dfa_factors_order.keys()]
                    for set_factor_name, set_factor_val in set_factors.items():
                        dfa_config[dfa_factors_order[set_factor_name]] = set_factor_val
                    if modelname not in factor_to_peak_ttoks[free_factor_name]:
                        factor_to_peak_ttoks[free_factor_name][modelname] = {}
                    factor_to_peak_ttoks[free_factor_name][modelname][tuple(dfa_config)] = ffn_to_ptts
    
    for free_factor_name in factor_names:
        if free_factor_name == "Model": continue
        for modelname in args.models:
            if modelname not in factor_to_peak_ttoks[free_factor_name]: continue
            if len(factor_to_peak_ttoks[free_factor_name][modelname]) == 0:
                del factor_to_peak_ttoks[free_factor_name][modelname]

    plot_normalized_correctness_by_ttoks(df, plot_kwargs)
    plt.clf()

    all_factor_corrs = []
    for factor_name in factor_names:
        factor_corrs = plot_ptt_by_factor(factor_to_peak_ttoks[factor_name], factor_name, plot_kwargs)
        all_factor_corrs.append(factor_corrs)
    ptt_table(all_factor_corrs, plot_kwargs["foldername"])
    acc_table(df, plot_kwargs["foldername"])