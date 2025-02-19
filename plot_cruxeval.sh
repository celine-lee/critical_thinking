source .models_env
python plot_cruxeval.py --output_folder cruxeval/outputs  \
               --n_buckets 3 \
               --k_min 10 --k_max 125 --num_k_buckets 3 \
               --N_min 3 --N_max 45 --num_N_buckets 3  \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --delete_old

            
python plot_cruxeval.py --output_folder cruxeval/outputs  \
               --n_buckets 5 \
               --k_min 10 --k_max 125 --num_k_buckets 3 \
               --N_min 3 --N_max 45 --num_N_buckets 3  \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --delete_old

python plot_cruxeval.py --output_folder cruxeval/outputs_straightlined  \
               --n_buckets 3 \
               --k_min 10 --k_max 125 --num_k_buckets 3 \
               --N_min 3 --N_max 45 --num_N_buckets 3  \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --delete_old

            
python plot_cruxeval.py --output_folder cruxeval/outputs_straightlined  \
               --n_buckets 5 \
               --k_min 10 --k_max 125 --num_k_buckets 3 \
               --N_min 3 --N_max 45 --num_N_buckets 3  \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models ${ALL_MODELS_PLOTTING} \
               --delete_old
