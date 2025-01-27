python experiment.py --models  "Qwen/Qwen2.5-7B-Instruct"  "mistralai/Ministral-8B-Instruct-2410" #  "Qwen/Qwen2.5-32B-Instruct" "google/gemma-2-9b-it" # 
python experiment.py --disable_cot --models "mistralai/Ministral-8B-Instruct-2410" "google/gemma-2-9b-it" # "Qwen/Qwen2.5-32B-Instruct" "Qwen/Qwen2.5-7B-Instruct" # 

# python experiment.py --temperature 0.9 --num_gens_per 6 --models  "Qwen/Qwen2.5-32B-Instruct" "Qwen/Qwen2.5-7B-Instruct"  "mistralai/Ministral-8B-Instruct-2410" "google/gemma-2-9b-it" 
# python experiment.py --temperature 0.9 --num_gens_per 6 --disable_cot --models "Qwen/Qwen2.5-32B-Instruct" "Qwen/Qwen2.5-7B-Instruct"  "mistralai/Ministral-8B-Instruct-2410" "google/gemma-2-9b-it" 

