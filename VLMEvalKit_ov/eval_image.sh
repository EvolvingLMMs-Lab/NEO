#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

LOG_DIR="./logs"
MODEL="NEOov-9B-image"
DATASETS=(
    # Knowledge & Math
    "MMMU_DEV_VAL"
    # General VQA
    "MMBench_DEV_EN"
    "RealWorldQA"
    "MMStar"
    "SEEDBench_IMG"
    # OCR VQA
    "AI2D_TEST"
    "DocVQA_VAL"
    "ChartQA_TEST"
    "TextVQA_VAL"
    "OCRBench"
    # Hallucination
    "POPE"
    "HallusionBench"
)

mkdir -p "$LOG_DIR"

for DATA in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating ${MODEL} on ${DATA}"
    echo "=========================================="
    python run.py --data "$DATA" --model "$MODEL" --verbose > "$LOG_DIR/${MODEL}_${DATA}.log" 2>&1
done
