source .models_env
source .env

python main.py --task bool --k_vals 4 --N_vals 8 \
               --models ${VLLM_MODELS_FULL} --generator vllm --n_samples_per 30

python main.py --task array_idx --m_vals 17 --k_vals 17 --N_val 24 \
               --models ${VLLM_MODELS_FULL} --generator vllm --n_samples_per 30
