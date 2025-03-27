source .models_env
source .tasks_env
source .env

python src/extrapolate.py --models ${ALL_MODELS} > extrapolation.txt
# python src/extrapolate.py --models ${ALL_MODELS} --only_N > extrapolation_onlyN.txt