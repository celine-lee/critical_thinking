source .models_env
echo ${OPENAI_MODELS}
python main_dyck_openai.py --models ${OPENAI_MODELS} 
