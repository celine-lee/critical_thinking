source ../.models_env
source ../.tasks_env

python extrapolate.py ${ARITH_TASK_FLAGS} --models ${ALL_MODELS_PLOTTING} --new_k_val 3 --new_N_val 4
python extrapolate.py ${ARRAY_IDX_TASK_FLAGS} --models ${ALL_MODELS_PLOTTING} --new_k_val 17 --new_N_val 24
python extrapolate.py ${BOOL_TASK_FLAGS} --models ${ALL_MODELS_PLOTTING} --new_k_val 4 --new_N_val 8
python extrapolate.py ${DYCK_TASK_FLAGS} --models ${ALL_MODELS_PLOTTING} --new_k_val 8 --new_N_val 30
python extrapolate.py ${EVEN_ODD_TASK_FLAGS} --models ${ALL_MODELS_PLOTTING} --new_k_val 17 --new_N_val 24
python extrapolate.py ${NAVIGATE_TASK_FLAGS} --models ${ALL_MODELS_PLOTTING} --new_k_val 100 --new_N_val 15
python extrapolate.py ${SHUFFLE_TASK_FLAGS} --models ${ALL_MODELS_PLOTTING} --new_k_val 7 --new_N_val 7