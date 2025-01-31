import json
import re
import os
import shutil
import numpy as np
import scipy.stats as stats
import random
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from plot.plot_utils import *

global compute_random


def get_args():
    global compute_random
    parser = global_parser()
    parser.add_argument("--k_vals", nargs="+", default=None)
    parser.add_argument("--d_vals", nargs="+", default=None)
    parser.add_argument("--N_vals", nargs="+", default=None)

    args = parser.parse_args()
    compute_random = lambda factor_vals: 0.5
    args.foldername = os.path.join(
        f"{args.output_folder.rstrip('/')}_graphs_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}"
    )

    return args


def parse_kdN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_d(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = parsed_experimentname.group(1)
    d = parsed_experimentname.group(2)
    N = parsed_experimentname.group(3)
    return {"k": k, "d": d, "N": N, "Model": modelname}


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

    df = load_data(
        args.output_folder,
        {"k": args.k_vals, "d": args.d_vals, "N": args.N_vals, "Model": args.models},
        parse_kdN,
        plot_kwargs,
    )
    df_nocot = load_data(
        args.output_folder + "_nocot",
        {"k": args.k_vals, "d": args.d_vals, "N": args.N_vals, "Model": args.models},
        parse_kdN,
        plot_kwargs,
    )
    df = pd.concat([df, df_nocot])

    plot_length_generated(df, plot_kwargs, "k")
    plot_length_generated(df, plot_kwargs, "N")
    plot_length_generated(df, plot_kwargs, "d")
    plot_length_generated(df, plot_kwargs)

    k_vals = df["k"].unique()
    d_vals = df["d"].unique()
    N_vals = df["N"].unique()

    N_to_peak_ttoks = {}
    k_to_peak_ttoks = {}
    d_to_peak_ttoks = {}

    kdN_to_peak_ttoks_incorrect = {}
    kdN_to_peak_acc = {}


    for modelname in args.models:
        N_to_peak_ttoks[modelname] = {}
        k_to_peak_ttoks[modelname] = {}
        d_to_peak_ttoks[modelname] = {}
        kdN_to_peak_ttoks_incorrect[modelname] = {}
        kdN_to_peak_acc[modelname] = {}
        for d in d_vals:
            for k in k_vals:
                set_factors = {"k": k, "d": d, "Model": modelname}
                (
                    N_to_ptts,
                    N_to_ptts_incorrect,
                    N_to_peak_acc,
                ) = plot_correctness_by_ttoks_isolate_factor(
                    df, set_factors, "N", plot_kwargs
                )
                if N_to_ptts:
                    N_to_peak_ttoks[modelname][(k, d, None)] = N_to_ptts
                for N_val, peak_ttoks_incorrect in N_to_ptts_incorrect:
                    kdN_to_peak_ttoks_incorrect[modelname][
                        (k, d, N_val)
                    ] = peak_ttoks_incorrect
                for N_val, peak_acc in N_to_peak_acc:
                    kdN_to_peak_acc[modelname][(k, d, N_val)] = peak_acc

                # plot_correctness_by_ttoks_isolate_factor(
                #     df, set_factors, "N", plot_kwargs | {"plot_incorrect": True}
                # )

            for N in N_vals:
                set_factors = {"N": N, "d": d, "Model": modelname}
                (
                    k_to_ptts,
                    k_to_ptts_incorrect,
                    k_to_peak_acc,
                ) = plot_correctness_by_ttoks_isolate_factor(
                    df, set_factors, "k", plot_kwargs
                )
                if k_to_ptts:
                    k_to_peak_ttoks[modelname][(None, d, N)] = k_to_ptts
                for k_val, peak_ttoks_incorrect in k_to_ptts_incorrect:
                    kdN_to_peak_ttoks_incorrect[modelname][
                        (k_val, d, N)
                    ] = peak_ttoks_incorrect
                for k_val, peak_acc in k_to_peak_acc:
                    kdN_to_peak_acc[modelname][(k_val, d, N)] = peak_acc
                # plot_correctness_by_ttoks_isolate_factor(
                #     df, set_factors, "k", plot_kwargs | {"plot_incorrect": True}
                # )
        for k in k_vals:

            for N in N_vals:
                set_factors = {"N": N, "k": k, "Model": modelname}
                plot_cdfs(df, set_factors, plot_kwargs)
                d_to_ptts, _, _ = plot_correctness_by_ttoks_isolate_factor(
                    df, set_factors, "d", plot_kwargs
                )
                if d_to_ptts:
                    d_to_peak_ttoks[modelname][(k, None, N)] = d_to_ptts

        # for N in N_vals:
        # plot_correctness_by_k_isolate_factor(df, None, N, modelname)
        # plot_correctness_by_d_isolate_factor(df, None, N, modelname)

        if len(N_to_peak_ttoks[modelname]) == 0:
            del N_to_peak_ttoks[modelname]
        if len(k_to_peak_ttoks[modelname]) == 0:
            del k_to_peak_ttoks[modelname]
        if len(d_to_peak_ttoks[modelname]) == 0:
            del d_to_peak_ttoks[modelname]

    plt.clf()
    plot_ptt_by_factor(N_to_peak_ttoks, "N", False, plot_kwargs)
    plot_ptt_by_factor(N_to_peak_ttoks, "N", True, plot_kwargs)
    plot_ptt_by_factor(k_to_peak_ttoks, "k", False, plot_kwargs)
    plot_ptt_by_factor(k_to_peak_ttoks, "k", True, plot_kwargs)
    plot_ptt_by_factor(d_to_peak_ttoks, "d", False, plot_kwargs)
    plot_ptt_by_factor(d_to_peak_ttoks, "d", True, plot_kwargs)

    plot_ptt_by_factor(N_to_peak_ttoks, "N", False, plot_kwargs, True)
    plot_ptt_by_factor(N_to_peak_ttoks, "N", True, plot_kwargs, True)
    plot_ptt_by_factor(k_to_peak_ttoks, "k", False, plot_kwargs, True)
    plot_ptt_by_factor(k_to_peak_ttoks, "k", True, plot_kwargs, True)
    plot_ptt_by_factor(d_to_peak_ttoks, "d", False, plot_kwargs, True)
    plot_ptt_by_factor(d_to_peak_ttoks, "d", True, plot_kwargs, True)

    plot_peak_accuracy_heatmap(kdN_to_peak_acc, plot_kwargs)
    plot_peak_token_difference_heatmap(kdN_to_peak_ttoks_incorrect, plot_kwargs)
