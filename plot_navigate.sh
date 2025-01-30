python plot_navigate.py --output_folder navigate/outputs  \
               --k_vals 10 30 100 \
               --d_vals 1 2 3 \
               --N_vals 5 10 15 \
               --get_isolated Model k d N \
               --n_buckets 5 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models gemma-2-9b-it Qwen2.5-32B-Instruct Qwen2.5-7B-Instruct Ministral-8B-Instruct-2410  
               # --only_meta

python plot_navigate.py --output_folder navigate/outputs  \
               --k_vals 10 30 100 \
               --d_vals 1 2 3 \
               --N_vals 5 10 15 \
               --get_isolated Model k d N \
               --n_buckets 3 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models gemma-2-9b-it Qwen2.5-32B-Instruct Qwen2.5-7B-Instruct Ministral-8B-Instruct-2410  
               # --only_meta
            