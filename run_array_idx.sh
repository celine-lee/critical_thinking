source .models_env
source .env
echo ${OPENAI_MODELS}
python main_array_idx.py --models o3-mini --generator openai --m_vals 5 17
