#!/usr/bin/env bash
# Co-train the full policy with unpacked COCO ShareGPT at weight 0.1 from the stage-1 EMA.
# Usage: bash scripts/stage2.sh [--print-argv] [-- HYDRA_OVERRIDE ...]
# Inputs and distributed settings: README.md.
set -euo pipefail
NEO_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
. "$NEO_REPO/scripts/lib.sh"

neo_stage_args "$@"
neo_need NEO_VLA_DATA_ROOT NEO_VLA_BACKBONE NEO_VLA_OUTPUT_ROOT STAGE1_CKPT \
    NEO_VLA_COCOGPTSHARE_RECORDS NEO_VLA_COCOGPTSHARE_IMAGES
neo_shape

MAX_STEPS=75000
GLOBAL_BATCH=8192
PER_DEVICE_BATCH=8
ACCUMULATION=$(neo_accumulation "$GLOBAL_BATCH" "$PER_DEVICE_BATCH")

NEO_ENV=(HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
         PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
         NEO_VLA_COCOGPTSHARE_RECORDS="$NEO_VLA_COCOGPTSHARE_RECORDS"
         NEO_VLA_COCOGPTSHARE_IMAGES="$NEO_VLA_COCOGPTSHARE_IMAGES"
         NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}")

train=(--root "$NEO_VLA_DATA_ROOT" --corpus "${PRETRAIN_CORPUS:-action_pretrain}"
       --run-name "${RUN_NAME:-stage2}" --backbone "$NEO_VLA_BACKBONE"
       --arch skip28 --mode cotrain --data action_pretrain_cotrain --schedule pretrain
       --output-root "$NEO_VLA_OUTPUT_ROOT" --nproc "$GPUS_PER_NODE"
       --workers "${WORKERS:-12}" --prefetch "${PREFETCH:-16}")
if [ "$NNODES" != 1 ]; then
    train+=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR"
            --master-port "$MASTER_PORT")
fi
train+=(--global-batch "$GLOBAL_BATCH" --per-device-batch "$PER_DEVICE_BATCH"
        --accumulation "$ACCUMULATION" --
        schedule.max_steps=$MAX_STEPS schedule.learning_rate=0.0002
        schedule.vlm_learning_rate=5e-05 schedule.lr_final=2e-05 schedule.warmup_steps=2000
        seed=1
        model.vision.pixel_convention=neo_native model.prefix.intrinsics_mode=tokens
        model.action.dct.enabled=false
        trainer.init_from="$STAGE1_CKPT" trainer.init_use_ema=true
        trainer.ema_decay=0.999 +trainer.ema_exclude_prefix=encoder.backbone.
        +trainer.ema_device=cpu
        trainer.gradient_checkpointing=true +trainer.strategy_options.gradient_as_bucket_view=true
        trainer.save_every=5000 trainer.keep_checkpoints=2 trainer.keep_skipped=0
        data.train.epoch_samples=$(( MAX_STEPS * GLOBAL_BATCH ))
        +trainer.compile_robot=true
        +trainer.robot_eager_numerics=true)
train+=(${NEO_EXTRA[@]+"${NEO_EXTRA[@]}"})

neo_launch "${train[@]}"
