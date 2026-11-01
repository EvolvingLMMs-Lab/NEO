#!/usr/bin/env bash
# Pretrain the action expert and prefix tokenizers with the backbone frozen.
# Usage: bash scripts/stage1.sh [--print-argv] [-- HYDRA_OVERRIDE ...]
# Inputs and distributed settings: README.md.
set -euo pipefail
NEO_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
. "$NEO_REPO/scripts/lib.sh"

neo_stage_args "$@"
neo_need NEO_VLA_DATA_ROOT NEO_VLA_BACKBONE NEO_VLA_OUTPUT_ROOT
neo_shape

GLOBAL_BATCH=8192
PER_DEVICE_BATCH=8
ACCUMULATION=$(neo_accumulation "$GLOBAL_BATCH" "$PER_DEVICE_BATCH")

NEO_ENV=(HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}")

train=(--root "$NEO_VLA_DATA_ROOT" --corpus "${STAGE1_CORPUS:-gd7_tokens_nobw}"
       --run-name "${RUN_NAME:-stage1}" --backbone "$NEO_VLA_BACKBONE"
       --arch skip28 --mode frozen_vlm --data gd7 --schedule pretrain
       --output-root "$NEO_VLA_OUTPUT_ROOT" --nproc "$GPUS_PER_NODE"
       --workers "${WORKERS:-8}" --prefetch "${PREFETCH:-8}")
if [ "$NNODES" != 1 ]; then
    train+=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR"
            --master-port "$MASTER_PORT")
fi
train+=(--global-batch "$GLOBAL_BATCH" --per-device-batch "$PER_DEVICE_BATCH"
        --accumulation "$ACCUMULATION" --
        schedule.max_steps=113400 seed=0
        model.vision.pixel_convention=neo_native model.prefix.intrinsics_mode=tokens
        model.action.dct.enabled=false
        trainer.gradient_checkpointing=false trainer.ema_decay=0.99
        trainer.save_every=1000 trainer.keep_checkpoints=2 trainer.keep_skipped=3
        trainer.compile_layers=true trainer.trim_prefix=64
        ${NEO_EXTRA[@]+"${NEO_EXTRA[@]}"})

neo_launch "${train[@]}"
