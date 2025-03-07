#!/bin/bash

source .models_env
source .env

# Combine all model lists into a single space-separated list
ALL_MODELS="${DIST_R1_MODELS} ${SMALLER_MODELS} ${BIGGER_MODELS}"

# Loop over each model
for MODEL in $ALL_MODELS; do
    echo "Running evaluation for model: $MODEL"
    python main.py $CRUXEVAL_TASK_FLAGS --models $MODEL --generator vllm
done
