source .models_env
echo ${ALL_MODELS}
python main_dyck_hf.py --models ${ALL_MODELS} --batch_size 1

. plot_dyck.sh