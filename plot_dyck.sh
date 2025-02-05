source .env
python plot_dyck.py --output_folder dyck/outputs  \
               --n_buckets 5 \
               --k_vals 1 4 \
               --d_vals 2 5 8 \
               --N_vals 16 24 30 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --only_meta 

python plot_dyck.py --output_folder dyck/outputs  \
               --n_buckets 3 \
               --k_vals 1 4 \
               --d_vals 2 5 8 \
               --N_vals 16 24 30 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --only_meta 
