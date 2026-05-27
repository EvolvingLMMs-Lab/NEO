#!/bin/bash

LOG_DIR="./logs"
NUM_GPUS=2 # Set the number of GPUs to use for data-parallel sharding

MODELS=(
    "NEOov-2B-video"
)

DATASETS=(
    "Video-MME_1fps_max256"
    "MVBench_64frame"
    "lvbench_1fps_max256"
    "MLVU_128frame"
    "LongVideoBench_256frame"
    "VideoMMMU_128frame"
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
