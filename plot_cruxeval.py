import json
import re
import os
import shutil
import numpy as np
import pandas as pd

from plot.plot_utils import *


def get_args():
    parser = global_parser()

    parser.add_argument("--k_min", type=int)
    parser.add_argument("--k_max", type=int)
    parser.add_argument("--N_min", type=int)
    parser.add_argument("--N_max", type=int)
    parser.add_argument("--l_min", type=int)
    parser.add_argument("--l_max", type=int)
    parser.add_argument("--num_k_buckets", type=int)
    parser.add_argument("--num_N_buckets", type=int)
    parser.add_argument("--num_l_buckets", type=int)

    args = parser.parse_args()
    args.foldername = os.path.join(
        f"{args.output_folder.rstrip('/')}_graphs_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}"
    )
    return args


def load_data(data_folder, varnames_and_wanted_vals, kwargs):
    loaded_data = {
        "No gen toks": [],
        "Correct?": [],
        "Model": [],
        "k": [],
        "N": [],
        "l": [],
    }

    # Load data from experiment files
    for experiment_file in glob.glob(os.path.join(data_folder, '*')):
        if f"T{kwargs['temperature']}" not in experiment_file:
            continue
        if re.search(r'_B\d+_S\d+', experiment_file):
            if f"_B{kwargs['num_beams']}_S{kwargs['num_gens']}.json" not in experiment_file: 
                continue
        elif kwargs['temperature'] == 0.0: 
            assert kwargs['num_beams'] == 1 and kwargs['num_gens'] == 1
        modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
        if modelname not in varnames_and_wanted_vals["Model"]: continue
        
        results = json.load(open(experiment_file))
        max_l = 0
        for ex in results:
            code = re.search(r'def f\(.*\):\s*\n\s+([\s\S]+?)\nanswer = f', ex["query"]).group(1).strip()
            ex["l"] = len(code.splitlines())
            if ex["l"] > max_l: max_l = ex["l"]
            if ex["k"] not in varnames_and_wanted_vals["k"]: continue
            if ex["N"] not in varnames_and_wanted_vals["N"]: continue
            if ex["l"] not in varnames_and_wanted_vals["l"]: continue
            loaded_data["Model"].append(modelname)
            loaded_data["k"].append(ex["k"])
            loaded_data["N"].append(ex["N"])
            loaded_data["l"].append(ex["l"])
            loaded_data["No gen toks"].append(ex["generated_tokens"])
            loaded_data["Correct?"].append(ex["correct"])
        print(max_l)
    # Create a DataFrame for the new data
    df = pd.DataFrame(loaded_data)

    return df

def bucketize_k_n(df, num_k_buckets, num_N_buckets, num_l_buckets):
    # There's too many distinct k and N values, so instead replace each entry's k and N with the closest bucket
    # where the buckets are selected based on average of equal splitting
    df = df.copy()

    # Assign each 'k' and 'N' value to the nearest equal-split center
    df["k"] = pd.qcut(df["k"], q=num_k_buckets, duplicates="drop").apply(
        lambda x: (x.left + x.right) / 2
    ).astype(int)
    df["N"] = pd.qcut(df["N"], q=num_N_buckets, duplicates="drop").apply(
        lambda x: (x.left + x.right) / 2
    ).astype(int)
    df["l"] = pd.qcut(df["l"], q=num_l_buckets, duplicates="drop").apply(
        lambda x: (x.left + x.right) / 2
    ).astype(int)
    return df

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
        "compute_random": lambda factor_vals: 0,
        "plot_incorrect": False,
    }

    df = load_data(
        args.output_folder,
        {"Model": args.models, "k": list(range(args.k_min, args.k_max)), "N": list(range(args.N_min, args.N_max)), "l": list(range(args.l_min, args.l_max))},
        plot_kwargs,
    )
    df_nocot = load_data(
        args.output_folder + "_nocot",
        {"Model": args.models, "k": list(range(args.k_min, args.k_max)), "N": list(range(args.N_min, args.N_max)), "l": list(range(args.l_min, args.l_max))},
        plot_kwargs,
    )
    df = pd.concat([df, df_nocot])

    df = bucketize_k_n(df, args.num_k_buckets, args.num_N_buckets, args.num_l_buckets)

    plot_length_generated(df, plot_kwargs)
    plot_length_generated(df, plot_kwargs, "k")
    plot_length_generated(df, plot_kwargs, "N")
    plot_length_generated(df, plot_kwargs, "l")

    k_vals = df["k"].unique()
    N_vals = df["N"].unique()
    l_vals = df["l"].unique()
    models = df["Model"].unique()

    N_to_peak_ttoks = {}
    k_to_peak_ttoks = {}
    l_to_peak_ttoks = {}

    kN_to_peak_ttoks_incorrect = {}
    kN_to_peak_acc = {}
    kN_to_peak_precision = {}
    kN_to_peak_recall = {}

    for modelname in models:
        N_to_peak_ttoks[modelname] = {}
        k_to_peak_ttoks[modelname] = {}
        l_to_peak_ttoks[modelname] = {}
        kN_to_peak_ttoks_incorrect[modelname] = {}
        kN_to_peak_acc[modelname] = {}
        kN_to_peak_precision[modelname] = {}
        kN_to_peak_recall[modelname] = {}
        for k in k_vals:
            for l in l_vals:
                set_factors = {"k": k, "l": l, "Model": modelname}
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
                        N_to_peak_ttoks[modelname][(k, l, None)] = N_to_ptts
                    for N_val, peak_ttoks_incorrect in N_to_ptts_incorrect:
                        kN_to_peak_ttoks_incorrect[modelname][
                            (k, l, N_val)
                        ] = peak_ttoks_incorrect
                    for N_val, (peak_acc, peak_precision, peak_recall) in N_to_peak_acc:
                        kN_to_peak_acc[modelname][(k, l, N_val)] = peak_acc
                        kN_to_peak_precision[modelname][(k, l, N_val)] = peak_precision
                        kN_to_peak_recall[modelname][(k, l, N_val)] = peak_recall

            for N in N_vals:
                set_factors = {"N": N, "k": k, "Model": modelname}
                ptt_data = plot_correctness_by_ttoks_isolate_factor(
                    df, set_factors, "l", plot_kwargs
                )
                if ptt_data:
                    (
                        l_to_ptts,
                        l_to_ptts_incorrect,
                        l_to_peak_acc,
                    ) = ptt_data
                    if l_to_ptts:
                        l_to_peak_ttoks[modelname][(k, None, N)] = l_to_ptts
                    for l_val, peak_ttoks_incorrect in l_to_ptts_incorrect:
                        kN_to_peak_ttoks_incorrect[modelname][
                            (k, l_val, N)
                        ] = peak_ttoks_incorrect
                    for l_val, (peak_acc, peak_precision, peak_recall) in l_to_peak_acc:
                        kN_to_peak_acc[modelname][(k, l_val, N)] = peak_acc
                        kN_to_peak_precision[modelname][(k, l_val, N)] = peak_precision
                        kN_to_peak_recall[modelname][(k, l_val, N)] = peak_recall

                # for N in N_vals:
                #     set_factors = {"N": N, "k": k, "l": l, "Model": modelname}
                #     plot_cdfs(df, set_factors, plot_kwargs)
        for N in N_vals:
            for l in l_vals:
                set_factors = {"l": l, "N": N, "Model": modelname}
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
                        k_to_peak_ttoks[modelname][(None, l, N)] = k_to_ptts
                    for k_val, peak_ttoks_incorrect in k_to_ptts_incorrect:
                        kN_to_peak_ttoks_incorrect[modelname][
                            (k_val, l, N)
                        ] = peak_ttoks_incorrect
                    for k_val, (peak_acc, peak_precision, peak_recall) in k_to_peak_acc:
                        kN_to_peak_acc[modelname][(k_val, l, N)] = peak_acc
                        kN_to_peak_precision[modelname][(k_val, l, N)] = peak_precision
                        kN_to_peak_recall[modelname][(k_val, l, N)] = peak_recall

        plot_correctness_by_isolate_factor(df, "k", ["N", "l"], plot_kwargs)
        plot_correctness_by_isolate_factor(df, "N", ["k", "l"], plot_kwargs)
        plot_correctness_by_isolate_factor(df, "l", ["k", "N"], plot_kwargs)

    plot_normalized_correctness_by_ttoks(df, plot_kwargs)
    plt.clf()
    # plot_ptt_by_factor(N_to_peak_ttoks, "N", False, plot_kwargs)
    plot_ptt_by_factor(N_to_peak_ttoks, "N", True, plot_kwargs)
    # plot_ptt_by_factor(k_to_peak_ttoks, "k", False, plot_kwargs)
    plot_ptt_by_factor(k_to_peak_ttoks, "k", True, plot_kwargs)
    # plot_ptt_by_factor(l_to_peak_ttoks, "l", False, plot_kwargs)
    plot_ptt_by_factor(l_to_peak_ttoks, "l", True, plot_kwargs)

    # plot_ptt_by_factor(N_to_peak_ttoks, "N", False, plot_kwargs, True)
    plot_ptt_by_factor(N_to_peak_ttoks, "N", True, plot_kwargs, True)
    # plot_ptt_by_factor(k_to_peak_ttoks, "k", False, plot_kwargs, True)
    plot_ptt_by_factor(k_to_peak_ttoks, "k", True, plot_kwargs, True)
    # plot_ptt_by_factor(l_to_peak_ttoks, "l", False, plot_kwargs, True)
    plot_ptt_by_factor(l_to_peak_ttoks, "l", True, plot_kwargs, True)

    plot_peak_accuracy_heatmap(kN_to_peak_acc, "Peak Accuracy", plot_kwargs)
    plot_peak_accuracy_heatmap(kN_to_peak_precision, "Precision", plot_kwargs)
    plot_peak_accuracy_heatmap(kN_to_peak_recall, "Recall", plot_kwargs)
    plot_peak_token_difference_heatmap(kN_to_peak_ttoks_incorrect, plot_kwargs)
