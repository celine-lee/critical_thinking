source .models_env
source .env
echo ${OPENAI_MODELS}
python main_dyck.py --models o3-mini --generator openai
echo ${ALL_MODELS}
python main_dyck.py --models {ALL_MODELS} 
