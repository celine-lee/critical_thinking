python plot_dyck.py --output_folder dyck/outputs  \
               --n_buckets 5 \
               --k_vals 1 4 \
               --d_vals 2 5 8 \
               --N_vals 16 24 30 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models Qwen2.5-32B-Instruct Ministral-8B-Instruct-2410 gemma-2-9b-it Qwen2.5-7B-Instruct \
               --only_meta 
            #    OLMo-2-1124-7B OLMo-2-1124-13B 3.1-8B Qwen2.5-14B gemma-2-9b 

python plot_dyck.py --output_folder dyck/outputs  \
               --n_buckets 3 \
               --k_vals 1 4 \
               --d_vals 2 5 8 \
               --N_vals 16 24 30 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models Qwen2.5-32B-Instruct Ministral-8B-Instruct-2410 gemma-2-9b-it Qwen2.5-7B-Instruct \
               --only_meta 
            #    OLMo-2-1124-7B OLMo-2-1124-13B 3.1-8B Qwen2.5-14B gemma-2-9b 
