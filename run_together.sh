source .models_env
source .env
source .tasks_env
export TOGETHER_API_KEY=cb08ddf9f76cbac1a44c3fb27167bdc19dec56f67850f493ac2e102f85f12824

python main.py --task gsm8k --models ${TOGETHER_MODELS} --generator together --n_samples_per 500
# python main.py $LOGICAL_DEDUCTION_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100
# python main.py $BOOL_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100  --disable_cot --max_new_tokens 20
# python main.py $NAVIGATE_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100  --disable_cot --max_new_tokens 20
# python main.py $ARITH_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100  --disable_cot --max_new_tokens 20
# python main.py $ARRAY_IDX_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100
# python main.py $CRUXEVAL_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100
# python main.py $DYCK_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100  --disable_cot --max_new_tokens 20
# python main.py $SHUFFLE_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100
# python main.py $EVEN_ODD_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100
# python main.py $WEB_OF_LIES_TASK_FLAGS --models ${TOGETHER_MODELS_NOCOT} --generator together --n_samples_per 100

