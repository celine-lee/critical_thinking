import numpy as np
import scipy.stats as stats
from tabulate import tabulate

from plot.plot import parse_kdN, parse_kmN, parse_kN, calculate_buckets, load_data, model_nicknames

import argparse
global compute_random
global foldername_parser
def get_args():
    global compute_random
    global foldername_parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_buckets", type=int, default=5)
    parser.add_argument("--num_gens", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--d_vals", type=int, nargs='+') 
    parser.add_argument("--m_vals", type=int, nargs='+') 
    parser.add_argument("--k_vals", type=int, nargs='+') 
    parser.add_argument("--N_vals", type=int, nargs='+')
    # parser.add_argument("--new_d_val", type=int, default=None)
    # parser.add_argument("--new_m_val", type=int, default=None)
    parser.add_argument("--new_k_val", type=int, default=None)
    parser.add_argument("--new_N_val", type=int, default=None)
    parser.add_argument("--task", choices=['dyck', 'array_idx', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects'])
    args = parser.parse_args()
    match args.task:
        case 'dyck':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kdN
            output_folder = "../dyck/outputs"
        case 'arith':
            compute_random = lambda factor_vals: 0.
            foldername_parser = parse_kmN
            output_folder = "../arithmetic/outputs"
        case 'array_idx':
            compute_random = lambda factor_vals: 0.
            foldername_parser = parse_kmN
            output_folder = "../array_idx_mult/outputs"
        case 'even_odd':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kmN
            output_folder = "../even_odd_mult/outputs"
        case 'bool':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kN
            output_folder = "../nested_boolean_expression/outputs"
        case 'navigate':
            compute_random = lambda factor_vals: 0.5
            foldername_parser = parse_kdN
            output_folder = "../navigate/outputs"
        case 'shuffled_objects':
            compute_random = lambda factor_vals: 0.
            foldername_parser = parse_kN
            output_folder = "../shuffled_objects/outputs"
    args.output_folder = output_folder

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
    then collect its (k, N, L*) values, where L* is taken as the Length Bucket Center 
    corresponding to the peak "Correct?" value.
    
    Then perform a linear least-squares fit to obtain a prediction of L* for the new (k, N).
    
    Returns:
        A dictionary mapping each model name to a tuple (predicted L*, tol)
    """
    predictions = {}
    for modelname in df["Model"].unique():
        model_df = df[df["Model"] == modelname]
        models_k_N_Lstar = []
        
        # Group by (k, N) for the current model.
        for (k_val, N_val), group in model_df.groupby(["k", "N"]):
            bucket_avg, _ = calculate_buckets(group, kwargs["n_buckets"])
            if bucket_avg is None or len(bucket_avg) == 0:
                continue

            # Find the bucket with the highest "Correct?" value.
            index_peak = np.argmax(bucket_avg["Correct?"])
            # Extract L* as the center of that bucket.
            Lstar = bucket_avg["Length Bucket Center"].iloc[index_peak]
            models_k_N_Lstar.append((k_val, N_val, Lstar))
        
        if len(models_k_N_Lstar) == 0:
            print(f"No (k, N, L*) data found for model {modelname}.")
            continue
        
        # Extract k, N, and L* values from the collected tuples.
        k_vals = np.array([item[0] for item in models_k_N_Lstar])
        N_vals = np.array([item[1] for item in models_k_N_Lstar])
        Lstar_vals = np.array([item[2] for item in models_k_N_Lstar])
        
        # Build the design matrix for linear regression.
        # Our model is: L* = a + b * k + c * N
        A = np.column_stack((np.ones(len(k_vals)), k_vals, N_vals))
        
        # Compute the least squares solution.
        coeffs, _, _, _ = np.linalg.lstsq(A, Lstar_vals, rcond=None)
        # Compute predictions on our training points.
        predicted_train = coeffs[0] + coeffs[1]*k_vals + coeffs[2]*N_vals
        residuals = Lstar_vals - predicted_train
        # Use the standard deviation of the residuals as a tolerance.
        tol = np.std(residuals)
        
        # Predict L* for the new (k, N)
        pred_Lstar = coeffs[0] + coeffs[1]*new_k + coeffs[2]*new_N

        predictions[modelname] = (pred_Lstar, tol)

    return predictions


if __name__ == "__main__":
    args = get_args()

    kwargs = {
        "n_buckets": args.n_buckets,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "num_gens": args.num_gens,
        "compute_random": compute_random,
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
        kwargs,
    )

    # Exclude rows corresponding to the new (k, N) pair from the data used for interpolation.
    df_without_new_k_and_L = df[~((df["k"] == args.new_k_val) & (df["N"] == args.new_N_val))]

    # # Get unique (k, N) pairs used for training the interpolator.
    # unique_kn = df_without_new_k_and_L[['k', 'N']].drop_duplicates()
    # print(f"Used (k, N) pairs: {unique_kn.to_dict(orient='records')}")

    predictions = linear_interpolate(df_without_new_k_and_L, args.new_k_val, args.new_N_val, kwargs)
    
    table_data = []
    for modelname, (pred_Lstar, tol) in predictions.items():

        # Get the data for the new (k, N) for this model.
        df_model_new = df[(df["k"] == args.new_k_val) & (df["N"] == args.new_N_val) & (df["Model"] == modelname)]
        if df_model_new.empty:
            # print(f"No data for model {modelname} at new_k and new_N.")
            continue

        # Get average of df_model_new["Correct?"]
        old_acc = df_model_new["Correct?"].mean()

        # For new_acc, filter rows to those with token count in [pred_Lstar - tol, pred_Lstar + tol]
        low_bound = pred_Lstar - tol
        high_bound = pred_Lstar + tol
        df_filtered = df_model_new[(df_model_new["No gen toks"] >= low_bound) & (df_model_new["No gen toks"] <= high_bound)]
        new_acc = df_filtered["Correct?"].mean() if not df_filtered.empty else 0.

        delta=f"{'+' if new_acc-old_acc >= 0 else ''}{new_acc-old_acc:.3f}"
        row = [
            model_nicknames[modelname],
            f"{pred_Lstar:.2f}",
            f"{tol:.2f}",
            f"{old_acc:.3f}",
            f"{new_acc:.3f}",
            delta
        ]
        table_data.append(row)
    table_data = sorted(table_data, key=lambda row:float(row[-3]))
    headers = ["Model", "Pred L*", "Tol", "Old Acc", "New Acc", "Delta"]
    print(f"Task: {args.task}. Pred for k={args.new_k_val}, N={args.new_N_val}")
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))