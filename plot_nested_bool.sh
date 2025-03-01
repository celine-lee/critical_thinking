source .models_env
python plot_nested_bool.py --output_folder nested_boolean_expression/outputs  \
               --n_buckets 3 \
               --k_vals 2 3 4 \
               --N_vals 1 3 5 8 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --delete_old 

# python plot_nested_bool.py --output_folder nested_boolean_expression/outputs  \
#                --n_buckets 5 \
#                --k_vals 2 3 4 \
#                --N_vals 1 3 5 8 \
#                --temperature 0.0 --num_beams 1 --num_gens 1 \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old 
