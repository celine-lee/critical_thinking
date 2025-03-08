source .models_env
source .tasks_env 


python src/plot/plot.py ${SHUFFLE_TASK_FLAGS} \
               --n_buckets 4 \
               --delete_old --only_meta \
               --models ${ALL_MODELS_PLOTTING} 

python src/plot/plot.py ${ARRAY_IDX_TASK_FLAGS} \
               --n_buckets 5 \
               --delete_old --only_meta \
               --models ${ALL_MODELS_PLOTTING} 

# python src/plot/plot.py ${EVEN_ODD_TASK_FLAGS} \
#                --n_buckets 3 \
#                --delete_old --only_meta \
#                --models ${ALL_MODELS_PLOTTING} 

# python src/plot/plot.py ${DYCK_TASK_FLAGS} \
#                --n_buckets 3 \
#                --models ${ALL_MODELS_PLOTTING}  \
#                --delete_old --only_meta
               

# python src/plot/plot.py ${NAVIGATE_TASK_FLAGS} \
#                --n_buckets 3 \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old --only_meta
            
# python src/plot/plot.py ${ARITH_TASK_FLAGS} \
#                --n_buckets 3 \
#                --delete_old --only_meta \
#                --models ${ALL_MODELS_PLOTTING}

# python src/plot/plot.py ${BOOL_TASK_FLAGS} \
#                --n_buckets 3 \
#                --models ${ALL_MODELS_PLOTTING} \
#                --delete_old --only_meta
               
# . plot_cruxeval.sh