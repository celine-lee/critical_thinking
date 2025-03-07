source .models_env
source .env
source .tasks_env
export TOGETHER_API_KEY=3786f74f9e503b1353402bcb49ab639790b7bd0182f688a09462f41308a3211f

python main.py $NAVIGATE_TASK_FLAGS \
               --models ${TOGETHER_MODELS_2} --generator together --n_samples_per 30

python main.py $DYCK_TASK_FLAGS \
               --models ${TOGETHER_MODELS_2} --generator together --n_samples_per 30

python main.py $BOOL_TASK_FLAGS \
               --models ${TOGETHER_MODELS_2} --generator together --n_samples_per 30
               
python main.py $CRUXEVAL_TASK_FLAGS \
               --models ${TOGETHER_MODELS_2} --generator together --n_samples_per 30

python main.py $ARRAY_IDX_TASK_FLAGS \
               --models ${TOGETHER_MODELS_2} --generator together --n_samples_per 30

python main.py $SHUFFLE_TASK_FLAGS \
               --models ${TOGETHER_MODELS_2} --generator together --n_samples_per 30
