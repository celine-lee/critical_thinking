source .models_env
source .env

# export BOOL_TASK_FLAGS="--task bool --k_vals 2 3 4 --N_vals 1 3 5 8"
# python main.py $BOOL_TASK_FLAGS \
#                --models ${TOGETHER_MODELS} --generator together --n_samples_per 50

export DYCK_TASK_FLAGS="--task dyck --d_vals 1 2 4 --k_vals 2 5 8 --N_vals 8 16 24 30"
python main.py $DYCK_TASK_FLAGS \
               --models ${TOGETHER_MODELS} --generator together --n_samples_per 50

# export NAVIGATE_TASK_FLAGS="--task navigate --d_vals 1 2 3 --k_vals 10 30 100 --N_vals 5 10 15"
# python main.py $NAVIGATE_TASK_FLAGS \
#                --models ${TOGETHER_MODELS} --generator together --n_samples_per 50

# export ARITH_TASK_FLAGS="--task arith --m_vals 5 10 15 --k_vals 1 2 3 --N_vals 2 3 4" 
# export ARRAY_IDX_TASK_FLAGS="--task array_idx --m_vals 1 5 9 17 --k_vals 5 9 17 --N_vals 1 10 16 24"
# export CRUXEVAL_TASK_FLAGS="--task cruxeval"
# export EVEN_ODD_TASK_FLAGS="--task even_odd --"
