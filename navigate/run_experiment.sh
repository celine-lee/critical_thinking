source ../.models_env
echo ${DSR1_MODELS}
python experiment.py --disable_cot --models deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
python experiment.py --models ${DSR1_MODELS}
python experiment.py --disable_cot --models ${DSR1_MODELS}

