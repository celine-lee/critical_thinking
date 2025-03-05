source .models_env
source .env

export DYCK_TASK_FLAGS="--task dyck --d_vals 1 2 4 --k_vals 2 5 8 --N_vals 8 16 24 30"
python main.py $DYCK_TASK_FLAGS \
               --models DeepSeek-R1 --generator deepseek --batch_size 2 --n_samples_per 30

# export BOOL_TASK_FLAGS="--task bool --k_vals 2 3 4 --N_vals 1 3 5 8"
# echo ${OPENAI_MODELS}
# python main.py $TASK_FLAGS \
#                --models ${OPENAI_MODELS} --generator openai --n_samples_per 30


# export ARRAY_IDX_TASK_FLAGS="--task array_idx --m_vals 1 5 9 17 --k_vals 5 9 17 --N_vals 1 10 16 24"
# python main.py $ARRAY_IDX_TASK_FLAGS \
#                --models DeepSeek-R1 --generator deepseek --n_samples_per 30