python plot.py --output_folder outputs  \
               --n_buckets 5 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models Qwen2.5-32B-Instruct Ministral-8B-Instruct-2410 gemma-2-9b-it Qwen2.5-7B-Instruct \
               --delete_old 
            #    OLMo-2-1124-7B OLMo-2-1124-13B 3.1-8B Qwen2.5-14B gemma-2-9b 

python plot.py --output_folder outputs  \
               --n_buckets 3 \
               --temperature 0.0 --num_beams 1 --num_gens 1 \
               --models Qwen2.5-32B-Instruct Ministral-8B-Instruct-2410 gemma-2-9b-it Qwen2.5-7B-Instruct \
               --delete_old 
            #    OLMo-2-1124-7B OLMo-2-1124-13B 3.1-8B Qwen2.5-14B gemma-2-9b 
