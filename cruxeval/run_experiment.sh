source ../.models_env
echo ${ALL_MODELS}
python experiment.py --models ${ALL_MODELS}
python experiment.py --disable_cot --models ${ALL_MODELS}
