source .models_env
source .env
source .tasks_env
               
python main.py $CRUXEVAL_TASK_FLAGS \
               --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

python main.py $BOOL_TASK_FLAGS \
               --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

python main.py $DYCK_TASK_FLAGS \
               --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

# python main.py $NAVIGATE_TASK_FLAGS \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

# python main.py $ARITH_TASK_FLAGS \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

# python main.py $ARRAY_IDX_TASK_FLAGS \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30

# python main.py $EVEN_ODD_TASK_FLAGS \
#                --models ${TOGETHER_MODELS_TURBO_AND_FULL} --generator together --n_samples_per 30