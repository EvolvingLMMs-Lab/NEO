#!/usr/bin/env bash
# Fine-tune on real-robot data with unpacked COCO ShareGPT at weight 0.1.
# Usage: bash scripts/posttrain_real.sh [--print-argv] [-- HYDRA_OVERRIDE ...]
# Inputs and distributed settings: README.md.
set -euo pipefail
NEO_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
. "$NEO_REPO/scripts/lib.sh"

neo_stage_args "$@"
neo_need NEO_VLA_DATA_ROOT REAL_CORPUS STAGE2_CKPT NEO_VLA_BACKBONE NEO_VLA_OUTPUT_ROOT \
    NEO_VLA_COCOGPTSHARE_RECORDS NEO_VLA_COCOGPTSHARE_IMAGES
neo_shape

MAX_STEPS=${MAX_STEPS:-30000}
GLOBAL_BATCH=${GLOBAL_BATCH:-128}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-4}
ACCUMULATION=$(neo_accumulation "$GLOBAL_BATCH" "$PER_DEVICE_BATCH")

frame=()
case "${REAL_FRAME_MODE:-wrist}" in
    none) ;;
    wrist|random)
        neo_need REAL_FRAME_SET
        frame+=(data.frame_mode="${REAL_FRAME_MODE:-wrist}")
        [ "${REAL_FRAME_MODE:-wrist}" != wrist ] || frame+=(data.pin_keeps_other_poses=true)
        frame+=(data.frame_draw="${REAL_FRAME_DRAW:-episode}")
        item='(robot|camera:[0-9]+)'
        tuple="\\[$item(,$item)*\\]"
        for pair in $REAL_FRAME_SET; do
            if [ "$NEO_PRINT_ARGV" = 0 ] && ! [[ "$pair" =~ ^[a-z0-9_]+=\[$tuple(,$tuple)*\]$ ]]; then
                neo_die "REAL_FRAME_SET entry '$pair'" "KEY=[[camera:N]] or KEY=[[robot],[camera:N]]" \
                    "one entry per embodiment key of the corpus"
            fi
            frame+=("+data.frame_set.$pair")
        done ;;
    *) neo_die "REAL_FRAME_MODE=${REAL_FRAME_MODE}" "wrist, random or none" "set REAL_FRAME_MODE" ;;
esac

NEO_ENV=(HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
         NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
         NEO_VLA_COCOGPTSHARE_RECORDS="$NEO_VLA_COCOGPTSHARE_RECORDS"
         NEO_VLA_COCOGPTSHARE_IMAGES="$NEO_VLA_COCOGPTSHARE_IMAGES")

train=(--root "$NEO_VLA_DATA_ROOT" --corpus "$REAL_CORPUS"
       --run-name "${RUN_NAME:-posttrain_real}" --backbone "$NEO_VLA_BACKBONE"
       --arch skip28 --mode cotrain --data robot_adaptation --schedule finetune
       --output-root "$NEO_VLA_OUTPUT_ROOT" --nproc "$GPUS_PER_NODE" --workers "${WORKERS:-8}")
if [ "$NNODES" != 1 ]; then
    train+=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR"
            --master-port "$MASTER_PORT")
fi
train+=(--global-batch "$GLOBAL_BATCH" --per-device-batch "$PER_DEVICE_BATCH"
        --accumulation "$ACCUMULATION" --
        data.corpus="$REAL_CORPUS" schedule.max_steps="$MAX_STEPS" schedule.learning_rate=5e-05
        schedule.vlm_learning_rate=2.5e-05 schedule.lr_final=1e-05 schedule.warmup_steps=1000
        schedule.weight_decay=1e-08 seed=42
        model.vision.pixel_convention=neo_native model.prefix.intrinsics_mode=tokens
        model.action.dct.enabled=false
        trainer.init_from="$STAGE2_CKPT" trainer.init_use_ema=true trainer.ema_decay=null
        trainer.gradient_checkpointing=true
        trainer.save_every=1000 trainer.keep_checkpoints=1 trainer.keep_skipped=0
        data.train.epoch_samples=$(( MAX_STEPS * GLOBAL_BATCH ))
        ${frame[@]+"${frame[@]}"})
if [ -n "${REAL_NORMALISER:-}" ]; then train+=(data.train.normaliser="$REAL_NORMALISER"); fi
train+=(${NEO_EXTRA[@]+"${NEO_EXTRA[@]}"})

neo_launch "${train[@]}"
