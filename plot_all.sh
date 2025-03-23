source .models_env
source .tasks_env 

python src/plot_all_tasks.py --models ${ALL_MODELS_PLOTTING} --delete_old

# python src/plot.py ${SHUFFLE_TASK_FLAGS} \
#                --delete_old --only_meta \
#                --models ${ALL_MODELS_PLOTTING} 

# python src/plot.py ${ARRAY_IDX_TASK_FLAGS} \
#                --delete_old --only_meta \
#                --models ${ALL_MODELS_PLOTTING} 

# python src/plot.py ${EVEN_ODD_TASK_FLAGS} \
#                --delete_old --only_meta \
#                --models ${ALL_MODELS_PLOTTING} 

# python src/plot.py ${DYCK_TASK_FLAGS} \
#                --models ${ALL_MODELS_PLOTTING}  \
#                --delete_old --only_meta

# python src/plot.py ${WEB_OF_LIES_TASK_FLAGS} \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old --only_meta

# python src/plot.py ${NAVIGATE_TASK_FLAGS} \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old --only_meta
            
# python src/plot.py ${ARITH_TASK_FLAGS} \
#                --delete_old --only_meta \
#                --models ${ALL_MODELS_PLOTTING}

# python src/plot.py ${BOOL_TASK_FLAGS} \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old --only_meta
               

# python src/plot.py ${LOGICAL_DEDUCTION_TASK_FLAGS} \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old --only_meta
               
# python src/plot.py ${CRUXEVAL_TASK_FLAGS} \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old --only_meta
               