source .models_env
source .tasks_env
source .env

python src/extrapolate.py --models ${ALL_MODELS} > extrapolation_3_24.txt
python src/extrapolate.py --models ${ALL_MODELS} --only_N > extrapolation_3_24_onlyN.txt