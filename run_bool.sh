source .models_env
source .env
export BOOL_TASK_FLAGS="--task bool --k_vals 2 3 4 --N_vals 1 3 5 8"

python main.py $BOOL_TASK_FLAGS \
               --models deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ${SMALLER_MODELS} ${BIGGER_MODELS} --generator vllm 


export DYCK_TASK_FLAGS="--task dyck --d_vals 1 2 4 --k_vals 2 5 8 --N_vals 8 16 24 30"
python main.py $DYCK_TASK_FLAGS \
               deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ${SMALLER_MODELS} ${BIGGER_MODELS} --generator vllm 
