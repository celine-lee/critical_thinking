source .models_env
source .env
source .tasks_env

python main.py $LOGICAL_DEDUCTION_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai --n_samples_per 50 --max_num_tokens 16384
python main.py $ARITH_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai --n_samples_per 50
python main.py $ARRAY_IDX_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai  --n_samples_per 50
python main.py $BOOL_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai  --n_samples_per 50
python main.py $CRUXEVAL_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai  --n_samples_per 50
python main.py $DYCK_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai  --n_samples_per 50
python main.py $EVEN_ODD_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai  --n_samples_per 50
python main.py $NAVIGATE_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai  --n_samples_per 50
python main.py $SHUFFLE_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai  --n_samples_per 50
python main.py $WEB_OF_LIES_TASK_FLAGS --models ${OPENAI_MODELS} --generator openai  --n_samples_per 100
