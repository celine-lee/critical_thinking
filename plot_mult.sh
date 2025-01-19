python plot_mult.py --output_folder array_idx_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 5 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models gemma-2-9b Qwen2.5-32B Qwen2.5-7B Ministral-8B  \
               --delete_old 
            #    OLMo-2-1124-7B OLMo-2-1124-13B 3.1-8B Qwen2.5-14B 

python plot_mult.py --output_folder array_idx_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 3 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models gemma-2-9b Qwen2.5-32B Qwen2.5-7B Ministral-8B  \
               --delete_old 
            #    OLMo-2-1124-7B OLMo-2-1124-13B 3.1-8B Qwen2.5-14B 

python plot_mult.py --output_folder array_idx_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 5 \
               --delete_old \
               --temperature 0.9 --num_beams 1 --num_gens 6 \
               --models 3.1-8B Ministral-8B gemma-2-9b Qwen2.5-32B Qwen2.5-14B Qwen2.5-7B OLMo-2-1124-13B 


python plot_mult.py --output_folder array_idx_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 3 \
               --delete_old \
               --temperature 0.9 --num_beams 1 --num_gens 6 \
               --models 3.1-8B Ministral-8B gemma-2-9b Qwen2.5-32B Qwen2.5-14B Qwen2.5-7B OLMo-2-1124-13B 


# python plot_mult.py --output_folder even_odd_mult/outputs  \
#                --k_vals 5 9 17 \
#                --m_vals 1 5 9 17 \
#                --N_vals 1 10 16 24 \
#                --get_isolated Model m t N \
#                --n_buckets 5 \
#                --temperature 0.0 --num_beams 1 --num_gens 1 \
#                --delete_old \
#                --models 3.1-8B Ministral-8B gemma-2-9b Qwen2.5-32B Qwen2.5-14B Qwen2.5-7B OLMo-2-1124-13B 


# python plot_mult.py --output_folder even_odd_mult/outputs  \
#                --k_vals 5 9 17 \
#                --m_vals 1 5 9 17 \
#                --N_vals 1 10 16 24 \
#                --get_isolated Model m t N \
#                --n_buckets 5 \
#                --temperature 0.9 --num_beams 1 --num_gens 6 \
#                --delete_old \
#                --models 3.1-8B Ministral-8B gemma-2-9b Qwen2.5-32B Qwen2.5-14B Qwen2.5-7B OLMo-2-1124-13B 

