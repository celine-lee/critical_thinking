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
    parser.add_argument("--k_vals", nargs='+', default=None)
    parser.add_argument("--m_vals", nargs='+', default=None)
    parser.add_argument("--N_vals", nargs='+', default=None)

    args = parser.parse_args()

    args.foldername = os.path.join(f"{args.output_folder.rstrip('/')}_graphs_{args.n_buckets}buckets_T{args.temperature}_B{args.num_beams}_S{args.num_gens}")

    if "even_odd" in args.output_folder: compute_random = lambda factor_vals: 0.5
    elif "array_i" in args.output_folder: compute_random = lambda factor_vals: 1/int(factor_vals['k'])

    return args


def plot_gen_by_factor(df, k, m, N, gen_factor="No gen toks"):
    if ONLY_PRODUCE_META_PLOTS: return
    assert sum((factor is None for factor in (k, m, N))) >= 1, f"{(k, m, N)} at least one must be None"

    if k is None:
        isolated_factor = "k"
        plot_multiple_of = "m"
    elif m is None:
        isolated_factor = "m"
        plot_multiple_of = "k"
    elif N is None:
        isolated_factor = "N"
        plot_multiple_of = "m"
        

    for modelname in df["Model"].unique():
        base_color = model_colors.get(modelname, "blue")

        # Filter the data for the specific model, k, m, N
        filtered_data = df[
            df["Model"].str.contains(modelname) 
            & ((df["k"] == k) if k is not None and plot_multiple_of != "k" else True)
            & ((df["m"] == m) if m is not None and plot_multiple_of != "m" else True)
            & ((df["N"] == N) if N is not None and plot_multiple_of != "N" else True)
        ]


        # Ensure there is data to plot
        if filtered_data.empty:
            print(f"No examples found for: {(k, m, N, modelname)}.")
            continue

        plt.figure(figsize=(12, 6))
        lines_plotted = set()

        line_labe_values = sorted(filtered_data[plot_multiple_of].unique(), key=int)
        max_line_label_value = int(line_labe_values[-1])
        for line_label in line_labe_values:
            line_data = filtered_data[filtered_data[plot_multiple_of] == line_label]
            factor_values = sorted(line_data[isolated_factor].unique(), key=int)
            if len(factor_values) == 0: continue

            means = []
            lower_bounds = []
            upper_bounds = []

            for val in factor_values:
                sub_df = line_data[line_data[isolated_factor] == val]
                gen_toks = sub_df[gen_factor]
                mean = gen_toks.mean()
                std_err = stats.sem(gen_toks)
                conf_int = std_err * stats.t.ppf((1 + 0.95) / 2., len(gen_toks) - 1)

                means.append(mean)
                lower_bounds.append(mean - conf_int)
                upper_bounds.append(mean + conf_int)

            lines_plotted.add(line_label)

            color_intensity = int(line_label) / (max_line_label_value + 1)
            rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
            plt.plot(factor_values, means, label=f"{plot_multiple_of}={line_label}", color=rgba_color)
            plt.fill_between(factor_values, lower_bounds, upper_bounds, color=rgba_color, alpha=color_intensity)

        # Customize plot labels and legend
        plt.ylabel(f"Average {gen_factor}")
        plt.xlabel(isolated_factor)
        plt.title(
            f"Average {gen_factor} vs. {isolated_factor} ({k, m, N, modelname})"
        )
        plt.legend(loc="best", fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save and clear the figure
        filename = f"{gen_factor}_{isolated_factor}{''.join(sorted(list(lines_plotted)))}_"
        filename += f"k{k}_" if k is not None and plot_multiple_of != 'k' else ""
        filename += f"m{m}_" if m is not None and plot_multiple_of != 'm' else ""
        filename += f"N{N}_" if N is not None and plot_multiple_of != 'N' else ""
        filename += f"{modelname}.png"
        plt.savefig(
            os.path.join(foldername, filename)
        )
        plt.clf()


def plot_correctness_by_N_isolate_factor(df, k, m, modelname):
    if ONLY_PRODUCE_META_PLOTS: return
    assert sum((factor is None for factor in (k, m))) == 1, f"{(k, m)} one must be None"
    # Filter the data for the specific model, k, m, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["k"] == k) if k is not None else True)
        & ((df["m"] == m) if m is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if m is None:
        isolated_factor = "m"
    elif k is None:
        isolated_factor = "k"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: {(k, m, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    def exponential_decay(x, a, b): 
       return a * np.exp(-b * x)

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("N")
            ["Correct?"].mean()
        )
        if len(performance.values) <= 1: continue
        used_vals.append(factor_value)
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            label=f"{isolated_factor}={factor_value}",
            marker="."
        )
        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for N in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["N"] == str(N)]["Correct?"]
            if sample.empty:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                ci = np.percentile(np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), [2.5, 97.5])
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

        # Plot confidence intervals as a shaded region
        plt.fill_between(
            performance.index.astype(int),
            ci_lower,
            ci_upper,
            color=rgba_color,
            alpha=color_intensity,
        )

        # curve fit
        initial_a_guess = 1.0
        initial_b_guess = performance.values[0]
        popt, pcov = curve_fit(exponential_decay, performance.index.astype(int).tolist(), performance.values, p0=[initial_a_guess, initial_b_guess])

        # overlay fitted exponential decay on plot.
        fitted_values = exponential_decay(performance.index.astype(int), *popt)
        plt.plot(
            performance.index.astype(int),
            fitted_values,
            linestyle="--",
            color="black",
            label=f"Fitted Curve ({isolated_factor}={factor_value}): d={popt[1]:.2f}",
            alpha=color_intensity,
            marker="."
        )

    if not len(used_vals): return
    # Add random guessing baseline (1/k) TODO
        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("N")
    plt.title(
        f"Correctness vs. N ({k, m, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_N_"
    if k is None:
        filename += f"m{m}_{modelname}.png"
    elif m is None:
        filename += f"k{k}_{modelname}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_k_isolate_factor(df, m, N, modelname):
    if ONLY_PRODUCE_META_PLOTS: return
    assert sum((factor is None for factor in (m, N))) == 1, f"{(m, N)} one must be None"
    # Filter the data for the specific model, k, m, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N"] == N) if N is not None else True)
        & ((df["m"] == m) if m is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if m is None:
        isolated_factor = "m"
    elif N is None:
        isolated_factor = "N"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: (m,N) {(m, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]
        filtered_data_factor["k"] = filtered_data_factor["k"].astype(int)

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("k")
            ["Correct?"].mean()
        )
        if len(performance.values) == 0: continue
        performance = performance.sort_index()
        used_vals.append(factor_value)
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            label=f"{isolated_factor}={factor_value}",
            marker="."
        )

        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for k in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["k"] == k]["Correct?"]
            if sample.empty:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                ci = np.percentile(np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), [2.5, 97.5])
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

        # Plot confidence intervals as a shaded region
        plt.fill_between(
            performance.index.astype(int),
            ci_lower,
            ci_upper,
            color=rgba_color,
            alpha=color_intensity,
        )


    if not len(used_vals): return

    # Add random guessing baseline (1/k)
    if not filtered_data["k"].empty:
        max_k = filtered_data["k"].astype(int).max()
        k_values = np.arange(1, max_k + 1)
        baseline = [compute_random(k) for k in k_values]
        plt.plot(
            k_values,
            baseline,
            color='gray',
            linestyle='--',
            label='Random Guessing (1/k)',
        )
        
    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("k")
    plt.title(
        f"Correctness vs. k ({m, N, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_k_"
    if N is None:
        filename += f"m{m}_{modelname}.png"
    elif m is None:
        filename += f"N{N}_{modelname}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def plot_correctness_by_m_isolate_factor(df, k, N, modelname):
    if ONLY_PRODUCE_META_PLOTS: return
    assert sum((factor is None for factor in (k, N))) == 1, f"{(k, N)} one must be None"
    # Filter the data for the specific model, k, N, modelname
    filtered_data = df[
        df["Model"].str.contains(modelname)
        & ((df["N"] == N) if N is not None else True)
        & ((df["k"] == k) if k is not None else True)
    ]
    base_color = model_colors.get(modelname, "blue")

    if k is None:
        isolated_factor = "k"
    elif N is None:
        isolated_factor = "N"

    # Ensure there is data to plot
    if filtered_data.empty:
        print(f"No examples found for: (k,N) {(k, N, modelname)}.")
        return

    plt.figure(figsize=(12, 6))

    max_val = filtered_data[isolated_factor].unique().astype(int).max().item() 
    used_vals = []
    for cmap_idx, factor_value in enumerate(sorted(filtered_data[isolated_factor].unique().astype(int))):
        filtered_data_factor = filtered_data[filtered_data[isolated_factor] == str(factor_value)]
        filtered_data_factor["m"] = filtered_data_factor["m"].astype(int)

        # Normalize the intensity of the color based on factor value
        color_intensity = int(factor_value) / (max_val+ 1)

        rgba_color = mcolors.to_rgba(base_color, alpha=color_intensity)
        
        # Calculate performance:
        performance = (
            filtered_data_factor.groupby("m")
            ["Correct?"].mean()
        )
        if len(performance.values) == 0: continue
        performance = performance.sort_index()
        used_vals.append(factor_value)
        # Plot the performance
        plt.plot(
            performance.index.astype(int),
            performance.values,
            color=rgba_color,
            label=f"{isolated_factor}={factor_value}",
            marker="."
        )

        # Calculate and display confidence intervals
        ci_lower = []
        ci_upper = []
        for m in performance.index.astype(int):
            sample = filtered_data_factor[filtered_data_factor["m"] == m]["Correct?"]
            if sample.empty:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
            else:
                ci = np.percentile(np.random.choice(sample, size=(1000, len(sample)), replace=True).mean(axis=1), [2.5, 97.5])
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

        # Plot confidence intervals as a shaded region
        plt.fill_between(
            performance.index.astype(int),
            ci_lower,
            ci_upper,
            color=rgba_color,
            alpha=color_intensity,
        )

    if not len(used_vals): return

    # Add random guessing baseline (1/k)
    if k:
        plt.axhline(
            y = compute_random(k),
            color='gray',
            linestyle='--',
            label=f'Random Guessing (1/k)',
        )

    # Customize plot labels and legend
    plt.xlim(xmin=0)
    plt.ylim(0, 1)
    plt.ylabel("Correctness")
    plt.xlabel("m")
    plt.title(
        f"Correctness vs. m ({k, N, modelname})"
    )
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and clear the figure
    filename = f"correctness_by_m_"
    if N is None:
        filename += f"k{k}"
    elif k is None:
        filename += f"N{N}"
    filename += f"_{modelname}.png"
    os.makedirs(os.path.join(foldername, "isolate_factor"), exist_ok=True)
    plt.savefig(
        os.path.join(foldername, "isolate_factor", filename)
    )
    plt.clf()

def parse_kmN(experiment_file):
    parsed_experimentname = re.search(r"k(\d+)_m(\d+)_N(\d+)", experiment_file)
    if parsed_experimentname is None:
        return None
    modelname = re.search(r"([^\/]+)_T", experiment_file).group(1)
    k = parsed_experimentname.group(1)
    m = parsed_experimentname.group(2)
    N = parsed_experimentname.group(3)
    return {"k": k, "m": m, "N": N, "Model": modelname}
    
if __name__ == "__main__":
    args = get_args()

    if args.delete_old and os.path.exists(args.foldername):
        shutil.rmtree(args.foldername)
    os.makedirs(args.foldername, exist_ok=True)

    plot_kwargs = {"n_buckets": args.n_buckets, "temperature": args.temperature, "num_beams": args.num_beams, "num_gens": args.num_gens, "foldername": args.foldername, "only_meta": args.only_meta, 'compute_random': compute_random}

        
    df = load_data(args.output_folder, {"k": args.k_vals, "m": args.m_vals, "N": args.N_vals, "Model": args.models}, parse_kmN, plot_kwargs)
    df_nocot = load_data(args.output_folder+"_nocot", {"k": args.k_vals, "m": args.m_vals, "N": args.N_vals, "Model": args.models}, parse_kmN, plot_kwargs)
    df = pd.concat([df, df_nocot])

    plot_length_generated(df, plot_kwargs, "k")
    plot_length_generated(df, plot_kwargs, "N")
    plot_length_generated(df, plot_kwargs, "m")
    plot_length_generated(df, plot_kwargs)

    k_vals = df["k"].unique()
    m_vals = df["m"].unique()
    N_vals = df["N"].unique()

    N_to_peak_ttoks = {}
    k_to_peak_ttoks = {}
    m_to_peak_ttoks = {}

    for modelname in args.models:
        N_to_peak_ttoks[modelname] = {}
        k_to_peak_ttoks[modelname] = {}
        m_to_peak_ttoks[modelname] = {}
        for m in m_vals:
            for k in k_vals:
                if "N" in args.get_isolated:
                    set_factors = {"k": k, "m": m, "Model": modelname}
                    N_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, set_factors, "N", plot_kwargs)
                    if N_to_ptts:
                        N_to_peak_ttoks[modelname][(k, m, None)] = N_to_ptts
                
            # plot_correctness_by_N_isolate_factor(df, None, m, modelname)
            # plot_correctness_by_k_isolate_factor(df, m, None, modelname)
            
            for N in N_vals:
                if "k" in args.get_isolated:
                    set_factors = {"N": N, "m": m, "Model": modelname}
                    k_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, set_factors, "k", plot_kwargs)
                    if k_to_ptts:
                        k_to_peak_ttoks[modelname][(None, m, N)] = k_to_ptts
            
        for k in k_vals:
            # plot_correctness_by_N_isolate_factor(df, k, None, modelname)
            # plot_correctness_by_m_isolate_factor(df, k, None, modelname)

            for N in N_vals:
                if "m" in args.get_isolated:
                    set_factors = {"N": N, "k": k, "Model": modelname}
                    m_to_ptts = plot_correctness_by_ttoks_isolate_factor(df, set_factors, "m", plot_kwargs)
                    if m_to_ptts:
                        m_to_peak_ttoks[modelname][(k, None, N)] = m_to_ptts

        # for N in N_vals:
            # plot_correctness_by_k_isolate_factor(df, None, N, modelname)
            # plot_correctness_by_m_isolate_factor(df, None, N, modelname)

        if len(N_to_peak_ttoks[modelname]) == 0:
            del N_to_peak_ttoks[modelname]
        if len(k_to_peak_ttoks[modelname]) == 0:
            del k_to_peak_ttoks[modelname]
        if len(m_to_peak_ttoks[modelname]) == 0:
            del m_to_peak_ttoks[modelname]

    plt.clf()
    plot_ptt_by_factor(N_to_peak_ttoks, "N", False, plot_kwargs)
    plot_ptt_by_factor(N_to_peak_ttoks, "N", True, plot_kwargs)
    plot_ptt_by_factor(k_to_peak_ttoks, "k", False, plot_kwargs)
    plot_ptt_by_factor(k_to_peak_ttoks, "k", True, plot_kwargs)
    plot_ptt_by_factor(m_to_peak_ttoks, "m", False, plot_kwargs)
    plot_ptt_by_factor(m_to_peak_ttoks, "m", True, plot_kwargs)

    # for k in k_vals:
    #     plot_gen_by_factor(df, k, None, None, "No gen toks")
    #     plot_gen_by_factor(df, k, None, None, "Len gen no digits")

    # for m in m_vals:
    #     plot_gen_by_factor(df, None, m, None, "No gen toks")
    #     plot_gen_by_factor(df, None, m, None, "Len gen no digits")

    # for N in N_vals:
    #     plot_gen_by_factor(df, None, None, N, "No gen toks")
    #     plot_gen_by_factor(df, None, None, N, "Len gen no digits")
        # for m in m_vals:
        #     if "Model" in args.get_isolated:
        #         for k in k_vals:
        #             model_to_peak_ttoks = plot_correctness_by_ttoks_isolate_factor(df, k, m, N, None)
