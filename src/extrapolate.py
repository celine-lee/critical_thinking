import numpy as np
import scipy.stats as stats
from tabulate import tabulate
import pandas as pd

from plot import load_data, model_nicknames
from task import *
from generator import *
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
    parser.add_argument("--tasks", nargs="+", default=['dyck', 'array_idx', 'cruxeval', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies'])
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


if __name__ == "__main__":
    args = get_args()

    all_models = [modelname_mappings[os.path.basename(modelname)] if os.path.basename(modelname) in modelname_mappings else os.path.basename(modelname) for modelname in args.models]

    modelname_to_average_delta = {}
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
        for dfa_key in dfa_factors_order.keys(): dfa_config_info[dfa_key] = None

        df = load_data(
            output_folder,
            dfa_config_info,
            foldername_parser,
            kwargs,
            filter_stddev_count=0
        )

        # get average delta per model across all heldout (k, N)
        kN_table_data = []
        for new_k_val in df["k"].unique():
            for new_N_val in df["N"].unique():

                # Exclude rows corresponding to the new (k, N) pair from the data used for interpolation.
                df_without_new_k_and_L = df[~((df["k"] == new_k_val) & (df["N"] == new_N_val))]

                predictions = linear_interpolate(df_without_new_k_and_L, new_k_val, new_N_val, kwargs)
                table_data = []
                for modelname, (pred_Lstar_low, pred_Lstar_high, tol, coeffs_low, coeffs_high) in predictions.items():
                    # For new_acc, filter rows to those with token count in [pred_Lstar - tol, pred_Lstar + tol]
                    low_bound = int(pred_Lstar_low - tol)
                    high_bound = int(pred_Lstar_high + tol)

                    # Get the data for the new (k, N) for this model.
                    df_model_new = df[(df["k"] == new_k_val) & (df["N"] == new_N_val) & (df["Model"] == modelname)]
                    if df_model_new.empty:
                        # print(f"No data for model {modelname} at new_k and new_N.")
                        continue

                    # Get average of df_model_new["Correct?"]
                    old_acc = df_model_new["Correct?"].mean()
                    # For new_acc, filter rows to those with token count in [pred_Lstar - tol, pred_Lstar + tol]
                    new_acc_df = df_model_new[(df_model_new["No gen toks"] >= low_bound) & 
                                            (df_model_new["No gen toks"] <= high_bound)]
                    if new_acc_df.empty:
                        continue
                    else:
                        new_acc = new_acc_df["Correct?"].mean()

                    if np.isnan(new_acc) or np.isnan(old_acc):
                        continue
                    else:
                        delta = f"{'+' if 100*(new_acc-old_acc) >= 0 else ''}{100*(new_acc-old_acc):.1f}"
                    row = [
                        model_nicknames[modelname],
                        f"{coeffs_low[0].item():.1f} + {coeffs_low[1].item():.1f}*k + {coeffs_low[2].item():.1f}*N",
                        f"{coeffs_high[0].item():.1f} + {coeffs_high[1].item():.1f}*k + {coeffs_high[2].item():.1f}*N",
                        f"({int(pred_Lstar_low)}, {int(pred_Lstar_high)}]",
                        f"{int(tol)}",
                        f"{old_acc:.3f}",
                        f"{new_acc:.3f}",
                        delta
                    ]
                    table_data.append(row)
                table_data = sorted(table_data, key=lambda row:float(row[-1]))

                kN_table_data.extend(table_data)
                # headers = ["Model", "Pred low", "Pred high", "Pred L*", "Tol", "Old Acc", "New Acc", "Delta (%)"]
                # print(f"Task: {task}. Pred for k={new_k_val}, N={new_N_val}")
                # print(tabulate(table_data, headers=headers, tablefmt="pretty"))

        averaged_table_data = {} # average kN_table_data 
        for row in kN_table_data:
            if row[0] not in averaged_table_data: averaged_table_data[row[0]] = []
            averaged_table_data[row[0]].append(float(row[-1]))

        averaged_table = []
        for modelname, deltas in averaged_table_data.items():
            average_delta_task_model = sum(deltas)/len(deltas)
            averaged_table.append([modelname, f"{average_delta_task_model:.2f}"])
            if modelname not in modelname_to_average_delta: modelname_to_average_delta[modelname] = []
            modelname_to_average_delta[modelname].append(average_delta_task_model)

        averaged_table = sorted(averaged_table, key=lambda row:float(row[-1]))
        headers = ["Model", "Delta (%)"]
        print(f"\n----\nTask: {task}")
        print(tabulate(averaged_table, headers=headers, tablefmt="pretty"))
        print("----\n")
    
    averaged_model_table = []
    for modelname, averaged_deltas in modelname_to_average_delta.items():
        averaged_averaged_deltas = sum(averaged_deltas) / len(averaged_deltas)
        averaged_model_table.append([modelname, f"{averaged_averaged_deltas:.2f}"])

    averaged_model_table = sorted(averaged_model_table, key=lambda row:float(row[-1]))
    headers = ["Model", "Delta (%)"]
    print("Averaged across all (task, k, N)")
    print(tabulate(averaged_model_table, headers=headers, tablefmt="pretty"))
    print("----\n")
