import numpy as np
import scipy.stats as stats
from tabulate import tabulate
import pandas as pd

from plot import parse_kdN, parse_kmN, parse_kN, calculate_buckets, load_data, model_nicknames
from task import *
from generator import *
from experimentor import modelname_mappings
from plot_utils import *

import argparse
global compute_random
global foldername_parser
def get_args():
    global compute_random
    global foldername_parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--new_temperature", type=float, default=0.6)
    parser.add_argument("--n_buckets", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
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
    parser.add_argument("--task", choices=['dyck', 'array_idx', 'cruxeval', 'even_odd', 'navigate', 'bool', 'arith', 'shuffled_objects', 'web_of_lies'])
    parser.add_argument("--generator", choices=['hf', 'openai', 'deepseek', 'vllm', 'together'])
    args = parser.parse_args()
    compute_random, foldername_parser, dfa_factors_order, output_folder = get_task_info(args.task)
    # match args.task:
    #     case 'dyck':
    #         task = DyckNTask()
    #     case 'arith':
    #         task = ArithmeticTask()
    #     case 'array_idx':
    #         task = ArrayIdxTask()
    #     case 'even_odd':
    #         task = EvenOddTask()
    #     case 'bool':
    #         task = NestedBoolTask()
    #     case 'navigate':
    #         task = NavigateTask()
    #     case 'shuffled_objects':
    #         task = ShuffledObjectsTask()
    #     case 'web_of_lies':
    #         task = WebOfLiesTask()
    #     case 'logical_deduction':
    #         task = LogicalDeductionTask()
    #     case 'cruxeval':
    #         task = CRUXEvalTask()

    match args.generator:
        case 'openai':
            generator = OpenAIGenerator
        case 'hf':
            generator = HFGenerator
        case 'deepseek':
            generator = DeepseekGenerator
        case 'vllm':
            generator = VLLMGenerator
        case 'together':
            generator = TogetherGenerator

    args.generator = generator
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
            Lstar_low = peak_buckets["Length Bucket"].iloc[0].left
            # L* high: last (highest) Length Bucket where peak occurs.
            Lstar_high = peak_buckets["Length Bucket"].iloc[-1].right
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


def predict_with_constraint(df_model_task, low_bound, high_bound, generator, task, test_set_size, output_filename, max_tries=5):
    print(output_filename)
    # For new_acc, filter rows to those with token count in [pred_Lstar - tol, pred_Lstar + tol]
    df_filtered = df_model_task[(df_model_task["No gen toks"] >= low_bound) & 
                                (df_model_task["No gen toks"] <= high_bound)].copy()
    orig_len = len(df_filtered)
    df_need_regen = df_model_task[(df_model_task["No gen toks"] < low_bound) | 
                                  (df_model_task["No gen toks"] > high_bound)].copy()
    if os.path.exists(output_filename): 
        # TODO handle this... load in the df there. 
        # if we have any repeat examples we're about to face, skip them
        # (can tell repeat examples by checking "prompt" and "Model")

    just_move_on_counter = {}
    old_acc = df_need_regen["Correct?"].mean()

    # Update the generator's sampling bounds using the correct variables.
    updated_all = generator.update_sampling_args_bounds(low_bound, high_bound)
    
    # This flag is computed for potential later use
    attempt_sampling_multiple_times = (
        (isinstance(generator.sampling_args, dict) and generator.sampling_args.get("temperature", 0) > 0.0) or 
        (not hasattr(generator.sampling_args, "temperature") or getattr(generator.sampling_args, "temperature", 0) > 0.0)
    )
    
    while len(df_filtered) < test_set_size and not df_need_regen.empty:
        # Select indices where the move-on counter is less than max_tries.
        available_indices = [idx for idx in df_need_regen.index if just_move_on_counter.get(idx, 0) < max_tries]
        if not available_indices:
            break
        
        # Randomly sample up to generator.max_batch_size indices from the available indices.
        k = min(generator.max_batch_size, len(available_indices))
        batch_idx = random.sample(available_indices, k=k)
        batch_df = df_need_regen.loc[batch_idx]
        prompts = batch_df["prompt"].tolist()
        if len(prompts) == 0:
            break
        generations, generation_lengths, extracted_answers = generator.generate(prompts, task)

        for idx, prompt, model_generation, num_generated_tokens, pred_answer in zip(
            batch_idx, prompts, generations, generation_lengths, extracted_answers
        ):
            # If the generated token count is outside the acceptable range,
            # update the move-on counter and possibly remove the row if max_tries is reached.
            if pred_answer is None or num_generated_tokens < low_bound or num_generated_tokens > high_bound:
                just_move_on_counter[idx] = just_move_on_counter.get(idx, 0) + 1
                if just_move_on_counter[idx] >= max_tries:
                    df_need_regen = df_need_regen.drop(idx)
                print(just_move_on_counter)
                continue
            # Extract the row using its index label
            row = batch_df.loc[idx]
            query = row["prompt"]
            true_answer = row["True"]
            
            if task.name in {"shuffled_objects", "web_of_lies"}:
                is_correct = pred_answer.lower().strip() == true_answer.lower().strip()
            else:
                try:
                    is_correct = eval(pred_answer) == true_answer
                except Exception:
                    is_correct = False
            row["No gen toks"] = num_generated_tokens
            row["Predicted"] = pred_answer
            row["Correct?"] = is_correct
            # Append the new entry to df_filtered (reassigning the result)
            df_filtered = pd.concat([df_filtered, row.to_frame().T], ignore_index=True)
            # Remove the processed row from df_need_regen
            df_need_regen = df_need_regen.drop(idx)
            df_filtered.to_json(output_filename, orient="records", indent=4)
            # if not df_filtered.empty:
            #     print(f"\tProgress... old acc {old_acc:.3f} --> {df_filtered.iloc[orig_len:]["Correct?"].mean()}")
    new_acc = df_filtered["Correct?"].mean() if not df_filtered.empty else 0.0

    return new_acc

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
    models = []
    unnicknamed = {}
    for model_name in args.models:
        modelname = os.path.basename(model_name)
        if modelname in modelname_mappings:
            modelname = modelname_mappings[modelname]
        models.append(modelname)
        unnicknamed[modelname] = model_name
    dfa_config_info["Model"] = models

    df = load_data(
        args.output_folder,
        dfa_config_info,
        foldername_parser,
        kwargs,
        filter_stddev_count=0
    )

    # Exclude rows corresponding to the new (k, N) pair from the data used for interpolation.
    df_without_new_k_and_L = df[~((df["k"] == args.new_k_val) & (df["N"] == args.new_N_val))]

    # # Get unique (k, N) pairs used for training the interpolator.
    # unique_kn = df_without_new_k_and_L[['k', 'N']].drop_duplicates()
    # print(f"Used (k, N) pairs: {unique_kn.to_dict(orient='records')}")

    predictions = linear_interpolate(df_without_new_k_and_L, args.new_k_val, args.new_N_val, kwargs)
    table_data = []
    for modelname, (pred_Lstar_low, pred_Lstar_high, tol, coeffs_low, coeffs_high) in predictions.items():
        # For new_acc, filter rows to those with token count in [pred_Lstar - tol, pred_Lstar + tol]
        low_bound = int(pred_Lstar_low - tol)
        high_bound = int(pred_Lstar_high + tol)
        gen_kwargs = {
            "max_new_tokens": high_bound,
            "min_new_tokens": low_bound,
            "num_beams": args.num_beams, 
            "stop_strings": ["[/ANSWER]"],
            "num_return_sequences": args.num_gens,
            "temperature": args.new_temperature
        }
        generator = args.generator(unnicknamed.get(modelname, modelname), gen_kwargs, args.batch_size)
        # Get the data for the new (k, N) for this model.
        df_model_new = df[(df["k"] == args.new_k_val) & (df["N"] == args.new_N_val) & (df["Model"] == modelname)]
        if df_model_new.empty:
            # print(f"No data for model {modelname} at new_k and new_N.")
            continue

        # Get average of df_model_new["Correct?"]
        old_acc = df_model_new["Correct?"].mean()
        test_set_size = 100
        output_filename = os.path.join(args.task.foldername, f"extrapolated_k{args.new_k_val}_N{args.new_N_val}_{low_bound}_{high_bound}_T{args.new_temperature}.json")
        new_acc = predict_with_constraint(df_model_new, low_bound, high_bound, generator, args.task, test_set_size, output_filename)

        delta=f"{'+' if new_acc-old_acc >= 0 else ''}{new_acc-old_acc:.3f}"
        row = [
            model_nicknames[modelname],
            f"{coeffs_low[0]:2f} + {coeffs_low[1]:2f}*k + {coeffs_low[2]:2f}*N",
            f"{coeffs_high[0]:2f} + {coeffs_high[1]:2f}*k + {coeffs_high[2]:2f}*N",
            f"({pred_Lstar_low:.2f}, {pred_Lstar_high:.2f}]",
            f"{tol:.2f}",
            f"{old_acc:.3f}",
            f"{new_acc:.3f}",
            delta
        ]
        table_data.append(row)
    table_data = sorted(table_data, key=lambda row:float(row[-3]))
    headers = ["Model", "Pred low", "Pred high", "Pred L*", "Tol", "Old Acc", "New Acc", "Delta"]
    print(f"Task: {args.task}. Pred for k={args.new_k_val}, N={args.new_N_val}")
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))