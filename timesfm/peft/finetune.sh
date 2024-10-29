#!/bin/bash

# Script to finetune a model with specific configurations
# Adjust the parameters below as needed. For a full list of options and descriptions, run the script with the --help flag.

export TF_CPP_MIN_LOG_LEVEL=2 XLA_PYTHON_CLIENT_PREALLOCATE=false

for c_len in 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512
do
    echo "Context length: $c_len"
    python3 finetune.py \
        --model-name="google/timesfm-1.0-200m" \
        --backend="gpu" \
        --horizon-len=28 \
        --context-len=$c_len \
        --freq="D" \
        --data-path="/nethome/sli999/data/Tycho/timesfm_US_covid_pivot.csv" \
        --num-epochs=20 \
        --learning-rate=1e-3 \
        --adam-epsilon=1e-7 \
        --adam-clip-threshold=1e2 \
        --early-stop-patience=10 \
        --datetime-col="ds" \
        --use-lora \
        --lora-rank=1 \
        --lora-target-modules="all" \
        --use-dora \
        --cos-initial-decay-value=1e-4 \
        --cos-decay-steps=40000 \
        --cos-final-decay-value=1e-5 \
        --ema-decay=0.9999 \

done

# python3 finetune.py \
#     --model-name="google/timesfm-1.0-200m" \
#     --backend="gpu" \
#     --horizon-len=28 \
#     --context-len=32 \
#     --freq="D" \
#     --data-path="/nethome/sli999/data/Tycho/timesfm_US_covid_pivot.csv" \
#     --num-epochs=50 \
#     --learning-rate=1e-3 \
#     --adam-epsilon=1e-7 \
#     --adam-clip-threshold=1e2 \
#     --early-stop-patience=10 \
#     --datetime-col="ds" \
#     --use-lora \
#     --lora-rank=1 \
#     --lora-target-modules="all" \
#     --use-dora \
#     --cos-initial-decay-value=1e-4 \
#     --cos-decay-steps=40000 \
#     --cos-final-decay-value=1e-5 \
#     --ema-decay=0.9999 \

# To see all available options and their descriptions, use the --help flag
# python3 finetune.py --help