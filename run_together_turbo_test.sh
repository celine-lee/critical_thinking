source .models_env
source .env
source .tasks_env
               
# python main.py --task cruxeval \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

python main.py --task bool --k_vals 4 --N_vals 8 \
               --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

# python main.py --task dyck  --d_vals 4 --k_vals 8 --N_vals 30 \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

# python main.py $NAVIGATE_TASK_FLAGS \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

# python main.py --task arith --m_vals 15 --k_vals 3 --N_vals 4 \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

python main.py --task array_idx --m_vals 17 --k_vals 17 --N_val 24 \
               --models ${TOGETHER_MODELS_TURBO} --generator together --n_samples_per 30

# python main.py $EVEN_ODD_TASK_FLAGS \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30