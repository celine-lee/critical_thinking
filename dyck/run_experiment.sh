source ../.models_env
echo ${DSR1_MODELS}
python experiment.py --models ${DSR1_MODELS}
python experiment.py --disable_cot --models ${DSR1_MODELS}

