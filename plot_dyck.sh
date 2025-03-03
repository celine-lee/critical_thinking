source .models_env
# python plot_kdN.py --output_folder dyck/outputs  \
#                --n_buckets 5 \
#                --d_vals 1 4 \
#                --k_vals 2 5 8 \
#                --N_vals 16 24 30 \
#                --temperature 0.0 --num_beams 1 --num_gens 1 \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old --only_meta

python plot_kdN.py --output_folder  dyck/outputs  \
               --n_buckets 3 \
               --d_vals 1 4 \
               --k_vals 2 5 8 \
               --N_vals 16 24 30 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} gpt-4o-mini \
               --delete_old