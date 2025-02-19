source .models_env
python plot_mult.py --output_folder array_idx_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 5 \
               --delete_old --only_meta \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING}
            
python plot_mult.py --output_folder array_idx_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 3 \
               --delete_old --only_meta \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING}

python plot_mult.py --output_folder even_odd_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 5 \
               --delete_old --only_meta \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING}
            
python plot_mult.py --output_folder even_odd_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 3 \
               --delete_old --only_meta \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING}
               

# python plot_mult.py --output_folder array_idx_mult/outputs  \
#                --k_vals 5 9 17 \
#                --m_vals 1 5 9 17 \
#                --N_vals 1 10 16 24 \
#                --get_isolated Model k m N \
#                --n_buckets 5 \
#                --temperature 0.9 --num_beams 1 --num_gens 6 \
#                --models ${ALL_MODELS_PLOTTING}\
#                --only_meta --delete_old

# python plot_mult.py --output_folder array_idx_mult/outputs  \
#                --k_vals 5 9 17 \
#                --m_vals 1 5 9 17 \
#                --N_vals 1 10 16 24 \
#                --get_isolated Model k m N \
#                --n_buckets 3 \
#                --temperature 0.0 --num_beams 1 --num_gens 1 \
               # --models ${ALL_MODELS_PLOTTING}\
#                --delete_old 
#             #    OLMo-2-1124-7B OLMo-2-1124-13B 3.1-8B Qwen2.5-14B 

# python plot_mult.py --output_folder array_idx_mult/outputs  \
#                --k_vals 5 9 17 \
#                --m_vals 1 5 9 17 \
#                --N_vals 1 10 16 24 \
#                --get_isolated Model k m N \
#                --n_buckets 5 \
               # --models ${ALL_MODELS_PLOTTING}\
#                --delete_old \
#                --temperature 0.9 --num_beams 1 --num_gens 6 \


# python plot_mult.py --output_folder array_idx_mult/outputs  \
#                --k_vals 5 9 17 \
#                --m_vals 1 5 9 17 \
#                --N_vals 1 10 16 24 \
#                --get_isolated Model k m N \
#                --n_buckets 3 \
               # --models ${ALL_MODELS_PLOTTING}\
#                --delete_old \
#                --temperature 0.9 --num_beams 1 --num_gens 6 \


# # python plot_mult.py --output_folder even_odd_mult/outputs  \
# #                --k_vals 5 9 17 \
# #                --m_vals 1 5 9 17 \
# #                --N_vals 1 10 16 24 \
# #                --get_isolated Model m t N \
# #                --n_buckets 5 \
               # --models ${ALL_MODELS_PLOTTING}\
# #                --temperature 0.0 --num_beams 1 --num_gens 1 \
# #                --delete_old \


# # python plot_mult.py --output_folder even_odd_mult/outputs  \
# #                --k_vals 5 9 17 \
# #                --m_vals 1 5 9 17 \
# #                --N_vals 1 10 16 24 \
# #                --get_isolated Model m t N \
# #                --n_buckets 5 \
               # --models ${ALL_MODELS_PLOTTING}\
# #                --temperature 0.9 --num_beams 1 --num_gens 6 \
# #                --delete_old \

