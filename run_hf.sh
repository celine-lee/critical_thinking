source .models_env
source .env
source .tasks_env

ALL_MODELS="${VLLM_MODELS} ${VLLM_MODELS_DISTR1_MODELS} ${VLLM_MODELS_maybedrop}"

python main.py $ARITH_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $EVEN_ODD_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $ARRAY_IDX_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $BOOL_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $DYCK_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $NAVIGATE_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $CRUXEVAL_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $SHUFFLE_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $WEB_OF_LIES_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100
python main.py $LOGICAL_DEDUCTION_TASK_FLAGS --models $ALL_MODELS --generator hf  --n_samples_per 100