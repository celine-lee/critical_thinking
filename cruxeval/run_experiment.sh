source ../.models_env
# echo ${BIGGER_MODELS}
# python experiment.py --models ${BIGGER_MODELS}
# python experiment.py --disable_cot --models ${BIGGER_MODELS}

echo ${SMALLER_MODELS}
python experiment.py --models ${SMALLER_MODELS}
python experiment.py --disable_cot --models ${SMALLER_MODELS}