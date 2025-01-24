import json
import re
import os
import shutil
import numpy as np
import pandas as pd

from plot.plot_utils import *

def get_args():
    parser = global_parser()
    parser.add_argument("--k_vals", nargs='+', default=None)
    parser.add_argument("--N_vals", nargs='+', default=None)

    args = parser.parse_args()
    args.foldername = os.path.join(f"{args.output_folder.rstrip('/')}_graphs_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}")
    return args

def parse_kN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = parsed_experimentname.group(1)
    N = parsed_experimentname.group(2)
    return {"k": k, "N": N, "Model": modelname}


if __name__ == "__main__":
    args = get_args()

    if args.delete_old and os.path.exists(args.foldername):
        shutil.rmtree(args.foldername)
    os.makedirs(args.foldername, exist_ok=True)

    plot_kwargs = {"n_buckets": args.n_buckets, "temperature": args.temperature, "num_beams": args.num_beams, "num_gens": args.num_gens, "foldername": args.foldername, "only_meta": args.only_meta, 'compute_random': lambda factor_vals: 0.5}

    df = load_data(args.output_folder, {"k": args.k_vals, "N": args.N_vals, "Model": args.models}, parse_kN, plot_kwargs)
    df_nocot = load_data(args.output_folder+"_nocot", {"k": args.k_vals, "N": args.N_vals, "Model": args.models}, parse_kN, plot_kwargs)
    df = pd.concat([df, df_nocot])

    plot_length_generated(df, plot_kwargs)

    k_vals = df["k"].unique()
    N_vals = df["N"].unique()
    models = df["Model"].unique()

    N_to_peak_ttoks = {}
    k_to_peak_ttoks = {}

    for modelname in models:
        N_to_peak_ttoks[modelname] = {}
        k_to_peak_ttoks[modelname] = {}
        for k in k_vals:
            set_factors = {"k": k, "Model": modelname}
            N_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, set_factors, "N", plot_kwargs)
            if N_to_ptts:
                N_to_peak_ttoks[modelname][(k, None)] = N_to_ptts
        for N in N_vals:
            set_factors = {"N": N,"Model": modelname}
            k_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, set_factors, "k", plot_kwargs)
            if k_to_ptts:
                k_to_peak_ttoks[modelname][(None, N)] = k_to_ptts

        plot_correctness_by_isolate_factor(df, "k", ["N"], plot_kwargs)
        plot_correctness_by_isolate_factor(df, "N", ["k"], plot_kwargs)
            
        plt.clf()
        plot_ptt_by_factor(N_to_peak_ttoks, "N", False, plot_kwargs)
        plot_ptt_by_factor(N_to_peak_ttoks, "N", True, plot_kwargs)
        plot_ptt_by_factor(k_to_peak_ttoks, "k", False, plot_kwargs)
        plot_ptt_by_factor(k_to_peak_ttoks, "k", True, plot_kwargs)

    # for k in k_vals:
    #     for N in N_vals:
    #         plot_correctness_by_ttoks_isolate_factor(df, k, N, None, models)
