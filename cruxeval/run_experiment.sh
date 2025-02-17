source ../.models_env
echo ${DSR1_MODELS}
python experiment_vllm.py --models ${DSR1_MODELS}
python experiment_vllm.py --disable_cot --models ${DSR1_MODELS}

