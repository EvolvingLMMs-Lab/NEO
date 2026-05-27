# 🚀 NEO Series: Native Vision-Language Models

Welcome to the **NEO Training Framework** - a comprehensive toolkit for training state-of-the-art vision-language models.

## 📋 Quick Start Guide

To use this framework, follow these two essential steps:

1. **📁 Customize Your Dataset**: Prepare your data and implement the configuration
2. **⚙️ Modify Training Scripts**: Adjust parameters to match your training requirements

## 📑 Contents
- [🔧 Installation](#-installation)
- [📂 Repository Structure](#-repository-structure)
- [📊 Custom Dataset Configuration](#-custom-dataset-configuration)
- [🎯 Usage](#-usage)

---

## 🔧 Installation

### Environment Setup

```bash
git clone https://github.com/EvolvingLMMs-Lab/NEO.git
cd NEO/VLMTrainKit
conda create -n neo python=3.12 -y
conda activate neo

pip install --upgrade pip
pip install .
```

### Model Preparation

> **📌 Note**: Download the required Qwen3 base models:

| Model | Link |
|-------|------|
| Qwen3-1.7B-Base | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B-Base) |
| Qwen3-8B-Base | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen3-8B-Base) |

---

## 📂 Repository Structure

### 🏋️ `neo/train/`
Training-related modules:
- **`trainer.py`**: Main trainer (extended from HuggingFace Trainer)
- **`argument.py`**: Dataclasses for model, data, and training arguments

### 📊 `neo/data/`
Data processing modules:
- **`__init__.py`**: Dataset configuration registry
- **`data_processor.py`**: Data processing pipeline for NEO models

---

## 📊 Custom Dataset Configuration

### 📝 JSON Data Structure

Your dataset should follow this structure:

#### **Required Fields**:

| Field | Description | Required |
|-------|-------------|----------|
| `image` | Path to the media file | ✅ Yes |
| `conversations` | List of question-answer pairs | ✅ Yes |

#### **Media Tags**:
- Use `<image>` tag in prompts for image understanding tasks
- Each tag must correspond to exactly one media file

### 💡 Example Instances:

**Single Image Example**:
```json
{
  "image": "demo.jpg",
  "width": 335,
  "height": 500,
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nSummarize the content of this picture."
    },
    {
      "from": "gpt",
      "value": "A wooden chair in the living room"
    }
  ]
}
```

**Packed Data Example**:
```json
[
    {
        "image": "images/001.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat's the main object in this picture?"
            },
            {
                "from": "gpt",
                "value": "A red apple on a wooden table"
            }
        ]
    },
    {
        "image": "images/002.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat's the main object in this picture?"
            },
            {
                "from": "gpt",
                "value": "A green orange on a plastic table"
            }
        ]
    }
]
```

---

### ⚙️ Dataset Configuration for Training

#### Step 1: Define Your Dataset

Create a dataset dictionary in [`neo/data/__init__.py`](neo/data/__init__.py):
```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/image/data",  # Can be empty if paths are in annotations
}
```

#### Step 2: Register Your Dataset

Add your dataset to the `data_dict`:
```python
data_dict = {
    "your_dataset_name": DATASET_NAME,
    # ... other datasets
}
```

#### Step 3: Configure Sampling Rate (Optional)

Control the amount of data used by appending `%X` to the dataset name:

| Syntax | Effect |
|--------|--------|
| `"dataset_name%50"` | Use 50% of the data |
| `"dataset_name%20"` | Use 20% of the data |
| `"dataset_name"` | Use 100% of the data (default) |

**Usage Example**:
```python
dataset_names = ["my_dataset%50"]  # Will use 50% of your dataset
configs = data_list(dataset_names)
```

### 📌 Important Notes

- ✅ **Annotation Path**: Must point to a JSON or JSONL file with dataset annotations
- ✅ **Data Path**: Can be empty if image paths in annotations are absolute
- ✅ **Sampling Rates**: Applied per-dataset when using multiple datasets
- ✅ **Format Compliance**:
  - Each `<image>` tag must correspond to exactly one image file
  - Each `<video>` tag must correspond to exactly one video file

---

## 🎯 Usage

### 💾 Packed Data Training for Memory Efficiency

NEO adopts a **packed data training strategy** to maximize GPU memory utilization. This approach concatenates multiple samples into a single sequence, significantly improving training efficiency.

#### ⚡ Key Considerations:

| Aspect | Description |
|--------|-------------|
| **Flexible Batch Size** | Can be set to any value, but must align with dataset characteristics |
| **Length Control** | Total packed length must not exceed `max_seq_length` |
| **Global Batch Size** | Total samples processed across all GPUs in one optimizer step |

#### 📐 Configuration Guidelines:

1. **Monitor** your dataset's average sample length
2. **Calculate** safe batch size: `batch_size × average_sample_length ≤ max_seq_length`
3. **Adjust** based on GPU memory capacity and model size

#### 💡 Example Calculation:

```
Given:
  - max_seq_length = 18,432
  - average_sample_length ≈ 3,000 tokens

Recommended:
  - batch_size = 6 (safe: 6 × 3,000 = 18,000 < 18,432)

⚠️  If samples vary greatly in length, reduce batch_size to prevent overflow.
```

---

### 🔧 Training Script Configuration

```bash
#!/bin/bash

#==============================================================================
# 🌐 Distributed Training Configuration
#==============================================================================
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

#==============================================================================
# ⚡ DeepSpeed Configuration
#==============================================================================
deepspeed=./scripts/zero3.json

#==============================================================================
# 🤖 Model Configuration
# 📖 For detailed logic, refer to: neo/model/build.py build_model function
#==============================================================================
mllm=""       # 🔧 Path to pre-trained NEO model for SFT (Supervised Fine-Tuning)
llm=""        # 🏗️  Path to the base LLM model for training NEO from scratch
tokenizer=""  # 📝 Path to the tokenizer

#==============================================================================
# 🎯 Training Hyperparameters
#==============================================================================
lr=2e-4

# 💥💥💥 Global Batch Size Control
# Global Batch Size: The total number of samples processed across all GPUs in the entire 
# training cluster during a single optimizer step (parameter update).
# Formula: Global batch size = batch_size × grad_accum_steps × num_gpus
# When data_flatten is enabled, batch_size also controls the length of flattened data
batch_size=1         # 📦 Per-device batch size for controlling global batch size and flatten data length
grad_accum_steps=1   # 🔄 Gradient accumulation steps for controlling global batch size

#==============================================================================
# 🚀 Training Entry Point
#==============================================================================
entry_file=neo/train/train.py

#==============================================================================
# 📊 Dataset Configuration
#==============================================================================
datasets=""  # 📁 Replace with your dataset names (e.g., "dataset1,dataset2%50")

#==============================================================================
# 💾 Output Configuration
#==============================================================================
run_name="neo-baseline"
output_dir=./output

#==============================================================================
# ⚙️ Training Arguments
#==============================================================================
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${mllm}" \ 
    --dataset_use ${datasets} \
    --data_flatten True \
    --dtype bfloat16 \
    --output_dir ${output_dir} \
    --extra_num_layers 12 \   # 🧱 Number of pre-buffer layers
    --num_hidden_layers 28 \  # 🏗️  Total number of layers in the model
    --train_buffer \          # 🎓 Whether to train only the prebuffer layers
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 262144 \     # 🖼️  Maximum pixels for image processing
    --min_pixels 12544 \      # 🖼️  Minimum pixels for image processing
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.0 \
    --warmup_steps 1000 \     # 🔥 Learning rate warmup steps
    --max_grad_norm 1 \
    --logging_steps 1 \
    --max_seq_length 18432 \  # ✂️  Maximum length after data flattening; sequences exceeding this will be truncated (see FlattenedDataCollatorForSupervisedDataset in data_processor.py)
    --model_max_length 18432 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"  # 📈 Logging to TensorBoard

#==============================================================================
# 🔧 Environment Setup
#==============================================================================
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

#==============================================================================
# 🎬 Launch Training
#==============================================================================
torchrun --nproc_per_node=2 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
```

---

## 📝 Key Training Parameters Explained

### 🧱 Model Architecture
- **`extra_num_layers`**: Number of pre-buffer layers (default: 12)
- **`num_hidden_layers`**: Total layers in the model (default: 28)
- **`train_buffer`**: Whether to train only the pre-buffer layers

### 💾 Batch Size & Memory
- **`per_device_train_batch_size`**: Samples per GPU per forward pass
- **`gradient_accumulation_steps`**: Accumulate gradients over N steps
- **Global Batch Size Formula**: `batch_size × grad_accum_steps × num_gpus`

### 📏 Sequence Length
- **`max_seq_length`**: Maximum length after data packing (default: 18432)
- **`model_max_length`**: Model's maximum context length (should match above)

### 🖼️ Image Processing
- **`max_pixels`**: Maximum pixels for image processing (default: 262144)
- **`min_pixels`**: Minimum pixels for image processing (default: 12544)

### 🎓 Learning Parameters
- **`learning_rate`**: Optimizer learning rate (default: 2e-4)
- **`warmup_steps`**: Learning rate warmup steps (default: 1000)
- **`max_grad_norm`**: Gradient clipping threshold (default: 1)

---

## 🎉 You're All Set!

Start your training journey with NEO and build powerful vision-language models! 

For issues or questions, please visit our [GitHub repository](https://github.com/EvolvingLMMs-Lab/NEO).