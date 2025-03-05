source .models_env
source .env
export TASK_FLAGS="--task array_idx --m_vals 1 5 9 17 --k_vals 5 9 17 --N_vals 1 10 16 24"

python main.py $TASK_FLAGS \
               --models DeepSeek-R1 --generator deepseek --n_samples_per 30