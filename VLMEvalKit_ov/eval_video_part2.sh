#!/bin/bash

LOG_DIR="./logs"
NUM_GPUS=2 # Set the number of GPUs to use for data-parallel sharding

MODELS=(
    "NEOov-9B-video"
)

DATASETS=(
    "Video-MME_1fps_max256"
    "MVBench_32frame"
    "MLVU_256frame"
    "lvbench_1fps_max256"
    "LongVideoBench_256frame"
    "VideoMMMU_256frame"
)

mkdir -p "$LOG_DIR"

# Run each (model, dataset) task with multi-GPU data sharding via torchrun
for MODEL in "${MODELS[@]}"; do
    for DATA in "${DATASETS[@]}"; do
        echo "Evaluating ${MODEL} on ${DATA} with ${NUM_GPUS} GPUs"
        torchrun --nproc-per-node="$NUM_GPUS" run.py \
            --data "$DATA" --model "$MODEL" --verbose \
            > "$LOG_DIR/${MODEL}_${DATA}.log" 2>&1
        echo "${MODEL} on ${DATA} done."
    done
done

echo "All evaluations completed."
