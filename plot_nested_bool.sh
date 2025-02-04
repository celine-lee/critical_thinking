python plot_nested_bool.py --output_folder nested_boolean_expression/outputs  \
               --n_buckets 3 \
               --k_vals 2 4 \
               --N_vals 1 3 5 8 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models Qwen2.5-32B-Instruct Ministral-8B-Instruct-2410 gemma-2-9b-it Qwen2.5-7B-Instruct \
               --only_meta 

# python plot_nested_bool.py --output_folder nested_boolean_expression/outputs  \
#                --n_buckets 3 \
#                --temperature 0.9 --num_beams 1 --num_gens 6 \
#                --models Qwen2.5-32B-Instruct Ministral-8B-Instruct-2410 gemma-2-9b-it Qwen2.5-7B-Instruct \
#                --delete_old 
python plot_nested_bool.py --output_folder nested_boolean_expression/outputs  \
               --n_buckets 5 \
               --k_vals 2 4 \
               --N_vals 1 3 5 8 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models Qwen2.5-32B-Instruct Ministral-8B-Instruct-2410 gemma-2-9b-it Qwen2.5-7B-Instruct \
               --only_meta 

# python plot_nested_bool.py --output_folder nested_boolean_expression/outputs  \
#                --n_buckets 5 \
#                --temperature 0.9 --num_beams 1 --num_gens 6 \
#                --models Qwen2.5-32B-Instruct Ministral-8B-Instruct-2410 gemma-2-9b-it Qwen2.5-7B-Instruct \
#                --delete_old 
