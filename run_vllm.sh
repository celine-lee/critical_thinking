source .models_env
source .env
source .tasks_env

# Loop over each model
for MODEL in $ALL_MODELS; do
    python main.py $ARITH_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $EVEN_ODD_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $ARRAY_IDX_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $BOOL_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $DYCK_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $NAVIGATE_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $CRUXEVAL_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $SHUFFLE_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $WEB_OF_LIES_TASK_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
    python main.py $LOGICAL_DEDUCTION_FLAGS --models $MODEL --generator vllm  --n_samples_per 80
done
