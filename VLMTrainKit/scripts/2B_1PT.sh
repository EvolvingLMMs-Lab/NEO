#!/bin/bash

# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
# For detailed logic, refer to: neo/model/build.py build_model function
mllm=""  # Path to pre-trained NEO model for SFT (Supervised Fine-Tuning) on top of an existing checkpoint
llm=""  # Path to the base LLM model for training NEO from scratch
tokenizer=""  # Path to the tokenizer

# Training hyperparameters
lr=8e-4
# Global batch size = batch_size * grad_accum_steps * num_gpus
batch_size=64  # Per-device batch size for controlling global batch size
grad_accum_steps=5  # Gradient accumulation steps for controlling global batch size

# Training entry point
entry_file=neo/train/train.py

# Dataset configuration (replace with public dataset names)
datasets=""

# Output configuration
run_name=neo-baseline-PT_2B
output_dir=./output

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --llm_model_name_or_path ${llm} \
    --tokenizer_name_or_path ${tokenizer} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --bf16 True \
    --output_dir ${output_dir} \
    --extra_num_layers 12 \
    --num_hidden_layers 40 \
    --train_buffer True \
    --max_steps 200000 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 4194304 \
    --min_pixels 65536 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --min_lr_ratio 0.1 \
    --max_grad_norm 1 \
    --logging_steps 1 \
    --max_seq_length 16384 \
    --model_max_length 8192 \
    --patch_size 16 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Launch training
torchrun --nproc_per_node=8 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}