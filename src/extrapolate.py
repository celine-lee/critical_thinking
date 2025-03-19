import numpy as np
import scipy.stats as stats
from tabulate import tabulate
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from plot import load_data, model_nicknames
from experimentor import modelname_mappings
from plot_utils import *

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
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--n_buckets", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", default=['dyck', 'array_idx', 'cruxeval', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies', 'logical_deduction'])
    args = parser.parse_args()
    return args

def get_Lstar(filtered_df, set_factor_values, kwargs):
    bucket_avg, filtered_data = calculate_buckets(filtered_df, kwargs['n_buckets'])
    if filtered_data is None: return None
    if len(bucket_avg) == 0: return None
        
    # Find the index of the maximum value
    index_peak = np.argmax(bucket_avg["Correct?"])
    peak_ttoks = bucket_avg["Length Bucket Center"][index_peak]
    best_performance = bucket_avg["Correct?"][index_peak]

    sem_values = filtered_data.groupby("Length Bucket Center", observed=True)["Correct?"].apply(stats.sem)
    # Calculate confidence intervals
    ci = sem_values * 1.96  # For 95% confidence
    ci = sem_values.reindex(bucket_avg["Length Bucket Center"]).fillna(0)

    return (peak_ttoks.item(), (best_performance - ci.values[index_peak]).item() > kwargs['compute_random'](set_factor_values))
    
def collect_k_N_Lstar_for_model(model_df, kwargs):
    k_N_Lstar = []

    for k_val in model_df["k"].unique():
        for N_val in model_df["N"].unique():
            factor_set_values = {"k": k_val, "N": N_val}

            # Filter the data for the specified factor_set_values
            bool_filter = True
            for factor_name, factor_val in factor_set_values.items():
                bool_filter = bool_filter & (model_df[factor_name] == factor_val)
            filtered_df = model_df[bool_filter]

            if filtered_df.empty:
                print(f"No examples found for: {factor_set_values}.")
                continue

            peak_results = get_Lstar(filtered_df, factor_set_values, kwargs)
            if peak_results:
                (peak_ttoks, task_doable) = peak_results
                if task_doable:
                    k_N_Lstar.append((k_val, N_val, peak_ttoks))

    return k_N_Lstar

def linear_interpolate(df, new_k, new_N, kwargs):
    """
    For each model in the dataframe, bucket the data by (k, N) using calculate_buckets,
    then for each bucketed group, determine:
      - L* low: the first (lowest x-axis value) at which the peak "Correct?" performance is observed.
      - L* high: the last (highest x-axis value) at which the peak performance is observed.
    
    A linear least-squares fit is performed separately for L* low and L* high as:
        L* = a + b * k + c * N
    to obtain predictions for the new (k, N) values.
    
    Returns:
        A dictionary mapping each model name to a tuple 
        (predicted L* low, predicted L* high, tol)
        where tol is computed as the average of the standard deviations of the residuals from the two fits.
    """
    predictions = {}
    for modelname in df["Model"].unique():
        model_df = df[df["Model"] == modelname]
        models_k_N_Lstar = []  # holds tuples of (k, N, L*_low, L*_high)
        
        # Group by (k, N) for the current model.
        for (k_val, N_val), group in model_df.groupby(["k", "N"]):
            bucket_avg, _ = calculate_buckets(group, kwargs["n_buckets"])
            if bucket_avg is None or len(bucket_avg) == 0:
                continue

            # Ensure buckets are sorted by the x-axis value.
            bucket_avg = bucket_avg.sort_values("Length Bucket Center")
            # Determine the peak "Correct?" value.
            peak_val = bucket_avg["Correct?"].max()
            # Select only the buckets where the performance equals the peak.
            peak_buckets = bucket_avg[bucket_avg["Correct?"] == peak_val]
            if peak_buckets.empty:
                continue
            # L* low: first (lowest) Length Bucket where peak occurs.
            # L* high: last (highest) Length Bucket where peak occurs.
            Lstar_bucket = peak_buckets["Length Bucket"].iloc[0]

            # Handle cases where the bucket isn't an interval
            if isinstance(Lstar_bucket, str):
                # Attempt to parse interval from string
                Lstar_parsed = re.search(r'\((\d+),\s*(\d+)\)', Lstar_bucket)
                if Lstar_parsed:
                    Lstar_low = int(Lstar_parsed.group(1))
                    Lstar_high = int(Lstar_parsed.group(2))
                else:
                    print(f"Warning: Could not parse bucket interval from string '{Lstar_bucket}' for model {modelname}, (k={k_val}, N={N_val}). Skipping.")
                    continue
            elif isinstance(Lstar_bucket, tuple) and len(Lstar_bucket) == 2:
                # If stored as a tuple (low, high)
                Lstar_low, Lstar_high = Lstar_bucket
            elif hasattr(Lstar_bucket, "left") and hasattr(Lstar_bucket, "right"):
                # If it's a proper interval object (e.g., pd.Interval)
                Lstar_low = Lstar_bucket.left
                Lstar_high = Lstar_bucket.right
            else:
                print(f"Warning: Unexpected bucket format '{Lstar_bucket}' for model {modelname}, (k={k_val}, N={N_val}). Skipping.")
                continue
            models_k_N_Lstar.append((k_val, N_val, Lstar_low, Lstar_high))
            
        if len(models_k_N_Lstar) == 0:
            print(f"No (k, N, L*) data found for model {modelname}.")
            continue
        
        # Extract k, N, and L* values from the collected tuples.
        k_vals = np.array([item[0] for item in models_k_N_Lstar])
        N_vals = np.array([item[1] for item in models_k_N_Lstar])
        Lstar_low_vals = np.array([item[2] for item in models_k_N_Lstar])
        Lstar_high_vals = np.array([item[3] for item in models_k_N_Lstar])
        
        # Build the design matrix for linear regression.
        # Our model is: L* = a + b * k + c * N
        A = np.column_stack((np.ones(len(k_vals)), k_vals, N_vals))
        
        # Compute the least squares solutions for L* low and L* high.
        coeffs_low, _, _, _ = np.linalg.lstsq(A, Lstar_low_vals, rcond=None)
        coeffs_high, _, _, _ = np.linalg.lstsq(A, Lstar_high_vals, rcond=None)
        
        # Compute predictions on the training points and calculate residuals.
        predicted_train_low = coeffs_low[0] + coeffs_low[1] * k_vals + coeffs_low[2] * N_vals
        residuals_low = Lstar_low_vals - predicted_train_low
        std_low = np.std(residuals_low)

        predicted_train_high = coeffs_high[0] + coeffs_high[1] * k_vals + coeffs_high[2] * N_vals
        residuals_high = Lstar_high_vals - predicted_train_high
        std_high = np.std(residuals_high)
        
        # Average the standard deviations to compute tolerance.
        tol = (std_low + std_high) / 2
        
        # Predict L* low and high for the new (k, N)
        pred_low = coeffs_low[0] + coeffs_low[1] * new_k + coeffs_low[2] * new_N
        pred_high = coeffs_high[0] + coeffs_high[1] * new_k + coeffs_high[2] * new_N

        predictions[modelname] = (pred_low, pred_high, tol, coeffs_low, coeffs_high)

    return predictions


def process_kn_pair(new_k_val, new_N_val, df, kwargs):
    """
    Process a single (k, N) pair:
    - Interpolates predictions.
    - Computes old & new accuracy.
    - Computes delta.
    Returns a list of table rows for this (k, N).
    """
    df_without_new_k_and_N = df[~((df["k"] == new_k_val) & (df["N"] == new_N_val))]
    predictions = linear_interpolate(df_without_new_k_and_N, new_k_val, new_N_val, kwargs)
    
    table_data = []
    for modelname, (pred_Lstar_low, pred_Lstar_high, tol, coeffs_low, coeffs_high) in predictions.items():
        low_bound = int(pred_Lstar_low - tol)
        high_bound = int(pred_Lstar_high + tol)

        # Get the data for the new (k, N) for this model.
        df_model_new = df[(df["k"] == new_k_val) & (df["N"] == new_N_val) & (df["Model"] == modelname)]
        if df_model_new.empty:
            continue

        old_acc = df_model_new["Correct?"].mean()

        # Compute new accuracy
        new_acc_df = df_model_new[(df_model_new["No gen toks"] >= low_bound) & 
                                  (df_model_new["No gen toks"] <= high_bound)]
        if new_acc_df.empty:
            continue
        new_acc = new_acc_df["Correct?"].mean()

        if np.isnan(new_acc) or np.isnan(old_acc):
            continue

        delta = 100 * (new_acc - old_acc)
        row = [
            model_nicknames[modelname],
            f"{coeffs_low[0].item():.1f} + {coeffs_low[1].item():.1f}*k + {coeffs_low[2].item():.1f}*N",
            f"{coeffs_high[0].item():.1f} + {coeffs_high[1].item():.1f}*k + {coeffs_high[2].item():.1f}*N",
            f"({int(pred_Lstar_low)}, {int(pred_Lstar_high)}]",
            f"{int(tol)}",
            f"{old_acc:.3f}",
            f"{new_acc:.3f}",
            f"{'+' if delta >= 0 else ''}{delta:.1f}"
        ]
        table_data.append(row)

    return table_data

if __name__ == "__main__":
    args = get_args()

    all_models = [modelname_mappings[os.path.basename(modelname)] if os.path.basename(modelname) in modelname_mappings else os.path.basename(modelname) for modelname in args.models]

    modelname_to_deltas = {}
    for task in args.tasks:
        compute_random, foldername_parser, dfa_factors_order, output_folder = get_task_info(task)

        kwargs = {
            "n_buckets": args.n_buckets,
            "temperature": 0.0,
            "num_beams": 1,
            "num_gens": 1,
            "compute_random": compute_random
        }

        dfa_config_info = {
            "Model": all_models,
        }
        for dfa_key in dfa_factors_order.keys():
            dfa_config_info[dfa_key] = None

        df = load_data(
            output_folder,
            dfa_config_info,
            foldername_parser,
            kwargs,
            filter_stddev_count=0
        )

        # Get average delta per model across all held-out (k, N)
        kN_table_data = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_kn_pair, new_k, new_N, df, kwargs): (new_k, new_N)
                for new_k in df["k"].unique() for new_N in df["N"].unique()
            }
            for future in as_completed(futures):
                kN_table_data.extend(future.result())

        # Compute averages and standard deviations for each model
        averaged_table_data = {}
        for row in kN_table_data:
            modelname = row[0]
            old_acc = float(row[-3])
            new_acc = float(row[-2])
            delta_value = float(row[-1])
            if modelname not in averaged_table_data:
                averaged_table_data[modelname] = []
            averaged_table_data[modelname].append((old_acc, new_acc, delta_value))

        # Compute mean and stddev per model for this task
        averaged_table = []
        for modelname, acc_info in averaged_table_data.items():
            old_acc = np.mean([info[0] for info in acc_info])
            new_acc = np.mean([info[1] for info in acc_info])
            deltas = [info[2] for info in acc_info]
            average_delta_task_model = np.mean(deltas)
            stddev = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0  # Use sample stddev
            averaged_table.append([modelname, f"{old_acc:.2f}", f"{new_acc:2f}", f"{average_delta_task_model:.2f}", f"{stddev/np.sqrt(len(deltas)):.2f}"])
            
            if modelname not in modelname_to_deltas:
                modelname_to_deltas[modelname] = []
            modelname_to_deltas[modelname].extend(deltas)

        averaged_table = sorted(averaged_table, key=lambda row: float(row[3]))  # Sort by average delta
        headers = ["Model", "Old Acc", "New Acc", "Delta (%)", "SE"]
        print(f"\n----\nTask: {task}")
        print(tabulate(averaged_table, headers=headers, tablefmt="pretty"))
        print("----\n")

    # Compute averages and stddev across all tasks
    averaged_model_table = []
    for modelname, deltas in modelname_to_deltas.items():
        averaged_deltas = np.mean(deltas)
        stddev = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0  # Sample stddev
        averaged_model_table.append([modelname, f"{averaged_deltas:.2f}", f"{stddev/np.sqrt(len(deltas)):.2f}"])

    averaged_model_table = sorted(averaged_model_table, key=lambda row: float(row[1]))  # Sort by avg delta
    headers = ["Model", "Delta (%)", "SE"]
    print("Averaged across all (task, k, N)")
    print(tabulate(averaged_model_table, headers=headers, tablefmt="pretty"))
    print("----\n")
