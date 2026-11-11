#!/bin/bash

GPUS=(0 1)

LOG_DIR="./logs"
MODELS=(
    "NEOov-2B-si"
    "NEOov-9B-si"
)
DATASETS=(
    "VSI-Bench_32frame"
    "MMSIBench_wo_circular"
    "MindCubeBench_tiny_raw_qa"
    "ViewSpatialBench"
    "SiteBenchImage"
    "SiteBenchVideo_32frame"
    "3DSRBench"
    "EmbSpatialBench"
    "SparBench"
    "MMSIVideoBench_50frame"
    "OmniSpatialBench_manual_cot"
    "BLINK"
    "MUIRBench_EASI"
)

mkdir -p "$LOG_DIR"

# Run each model on a separate GPU in parallel, datasets run serially within each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU="${GPUS[$i]}"
    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        for DATA in "${DATASETS[@]}"; do
            echo "[GPU $GPU] Evaluating ${MODEL} on ${DATA}"
            python run.py --data "$DATA" --model "$MODEL" --verbose > "$LOG_DIR/${MODEL}_${DATA}.log" 2>&1
        done
        echo "[GPU $GPU] ${MODEL} all done."
    ) &
done

# Wait for all background jobs to finish
wait
echo "All evaluations completed."
