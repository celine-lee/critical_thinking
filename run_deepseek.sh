source .models_env
source .env
source .tasks_env

python main.py $DYCK_TASK_FLAGS --models ${DS_MODELS} --generator deepseek --batch_size 2 --n_samples_per 30