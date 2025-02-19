source .models_env
python plot_cruxeval.py --output_folder cruxeval/outputs  \
               --n_buckets 3 \
               --k_min 0 --k_max 120 --num_k_buckets 3 \
               --N_min 0 --N_max 40 --num_N_buckets 3 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --delete_old

            
# python plot_cruxeval.py --output_folder cruxeval/outputs  \
#                --n_buckets 3 \
#                --k_min 0 --k_max 120 --num_k_buckets 5 \
#                --N_min 0 --N_max 140 --num_N_buckets 5  \
#                --l_min 0 --l_max 15 --num_l_buckets 5  \
#                --temperature 0.0 --num_beams 1 --num_gens 1 \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old

            
python plot_cruxeval.py --output_folder cruxeval/outputs  \
               --n_buckets 5 \
               --k_min 0 --k_max 120 --num_k_buckets 3 \
               --N_min 0 --N_max 140 --num_N_buckets 3  \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --delete_old
