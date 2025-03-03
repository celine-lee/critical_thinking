source .models_env
# python plot_mult.py --output_folder array_idx_mult/outputs  \
#                --k_vals 5 9 17 \
#                --m_vals 1 5 9 17 \
#                --N_vals 1 10 16 24 \
#                --get_isolated Model k m N \
#                --n_buckets 5 \
#                --delete_old --only_meta \
#                --temperature 0.0 --num_beams 1 --num_gens 1 \
#                --models ${ALL_MODELS_PLOTTING}
            
python plot_mult.py --output_folder even_odd_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 3 \
               --delete_old --only_meta \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${FEW_MODELS_PLOTTING} 
               
            
python plot_mult.py --output_folder array_idx_mult/outputs  \
               --k_vals 5 9 17 \
               --m_vals 1 5 9 17 \
               --N_vals 1 10 16 24 \
               --get_isolated Model k m N \
               --n_buckets 3 \
               --delete_old --only_meta \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} ${OPENAI_MODELS}

# python plot_mult.py --output_folder even_odd_mult/outputs  \
#                --k_vals 5 9 17 \
#                --m_vals 1 5 9 17 \
#                --N_vals 1 10 16 24 \
#                --get_isolated Model k m N \
#                --n_buckets 5 \
#                --delete_old --only_meta \
#                --temperature 0.0 --num_beams 1 --num_gens 1 \
#                --models ${ALL_MODELS_PLOTTING}