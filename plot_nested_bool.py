import json
import re
import os
import shutil
import numpy as np
import pandas as pd

from plot.plot_utils import *


def get_args():
    parser = global_parser()
    parser.add_argument("--k_vals", nargs="+", default=None)
    parser.add_argument("--N_vals", nargs="+", default=None)

    args = parser.parse_args()
    args.foldername = os.path.join(
        f"{args.output_folder.rstrip('/')}_graphs_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}"
    )
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

    plot_kwargs = {
        "n_buckets": args.n_buckets,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "num_gens": args.num_gens,
        "foldername": args.foldername,
        "only_meta": args.only_meta,
        "compute_random": lambda factor_vals: 0.5,
        "plot_incorrect": False,
    }

    df = load_data(
        args.output_folder,
        {"k": args.k_vals, "N": args.N_vals, "Model": args.models},
        parse_kN,
        plot_kwargs,
    )
    df_nocot = load_data(
        args.output_folder + "_nocot",
        {"k": args.k_vals, "N": args.N_vals, "Model": args.models},
        parse_kN,
        plot_kwargs,
    )
    df = pd.concat([df, df_nocot])

    plot_length_generated(df, plot_kwargs)
    plot_length_generated(df, plot_kwargs, "k")
    plot_length_generated(df, plot_kwargs, "N")

    k_vals = df["k"].unique()
    N_vals = df["N"].unique()
    models = df["Model"].unique()

    N_to_peak_ttoks = {}
    k_to_peak_ttoks = {}

    kN_to_peak_ttoks_incorrect = {}
    kN_to_peak_acc = {}
    kN_to_peak_precision = {}
    kN_to_peak_recall = {}

    for modelname in models:
        N_to_peak_ttoks[modelname] = {}
        k_to_peak_ttoks[modelname] = {}
        kN_to_peak_ttoks_incorrect[modelname] = {}
        kN_to_peak_acc[modelname] = {}
        kN_to_peak_precision[modelname] = {}
        kN_to_peak_recall[modelname] = {}
        for k in k_vals:
            set_factors = {"k": k, "Model": modelname}
            ptt_data = plot_correctness_by_ttoks_isolate_factor(
                df, set_factors, "N", plot_kwargs
            )
            if ptt_data:
                (
                    N_to_ptts,
                    N_to_ptts_incorrect,
                    N_to_peak_acc,
                ) = ptt_data
                if N_to_ptts:
                    N_to_peak_ttoks[modelname][(k, None)] = N_to_ptts
                for N_val, peak_ttoks_incorrect in N_to_ptts_incorrect:
                    kN_to_peak_ttoks_incorrect[modelname][
                        (k, N_val)
                    ] = peak_ttoks_incorrect
                for N_val, (peak_acc, peak_precision, peak_recall) in N_to_peak_acc:
                    kN_to_peak_acc[modelname][(k, N_val)] = peak_acc
                    kN_to_peak_precision[modelname][(k, N_val)] = peak_precision
                    kN_to_peak_recall[modelname][(k, N_val)] = peak_recall

            # for N in N_vals:
            #     set_factors = {"N": N, "k": k, "Model": modelname}
            #     plot_cdfs(df, set_factors, plot_kwargs)
        for N in N_vals:
            set_factors = {"N": N, "Model": modelname}
            ktt_data = plot_correctness_by_ttoks_isolate_factor(
                df, set_factors, "k", plot_kwargs
            )
            if ktt_data:
                (
                    k_to_ptts,
                    k_to_ptts_incorrect,
                    k_to_peak_acc,
                ) = ktt_data
                if k_to_ptts:
                    k_to_peak_ttoks[modelname][(None, N)] = k_to_ptts
                for k_val, peak_ttoks_incorrect in k_to_ptts_incorrect:
                    kN_to_peak_ttoks_incorrect[modelname][
                        (k_val, N)
                    ] = peak_ttoks_incorrect
                for k_val, (peak_acc, peak_precision, peak_recall) in k_to_peak_acc:
                    kN_to_peak_acc[modelname][(k_val, N)] = peak_acc
                    kN_to_peak_precision[modelname][(k_val, N)] = peak_precision
                    kN_to_peak_recall[modelname][(k_val, N)] = peak_recall

        plot_correctness_by_isolate_factor(df, "k", ["N"], plot_kwargs)
        plot_correctness_by_isolate_factor(df, "N", ["k"], plot_kwargs)

    plot_normalized_correctness_by_ttoks(df, plot_kwargs)
    plt.clf()

    N_corrs = plot_ptt_by_factor(N_to_peak_ttoks, "N", plot_kwargs)
    k_corrs = plot_ptt_by_factor(k_to_peak_ttoks, "k", plot_kwargs)
    ptt_table((N_corrs, k_corrs), plot_kwargs["foldername"])
    acc_table(df, plot_kwargs["foldername"])

    plot_peak_accuracy_heatmap(kN_to_peak_acc, "Peak Accuracy", plot_kwargs)
    # plot_peak_accuracy_heatmap(kN_to_peak_precision, "Precision", plot_kwargs)
    # plot_peak_accuracy_heatmap(kN_to_peak_recall, "Recall", plot_kwargs)
    # plot_peak_token_difference_heatmap(kN_to_peak_ttoks_incorrect, plot_kwargs)
