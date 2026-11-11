from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class ModelArguments:
    # Basic model paths
    model_name_or_path: Optional[str] = field(default=None)
    llm_model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_max_length: int = field(default=8096)
    # position embedding settings
    rope_theta_hw: float = field(default=10000.0)
    max_position_embeddings_hw: int = field(default=10000)
    rope_theta_vision: float = field(default=1000.0)
    max_position_embeddings_vision: int = field(default=10000)
    vision_hidden_size: int = field(default=1024)
    vision_num_channels: int = field(default=3)
    extra_num_layers: int = field(default=1)
    num_hidden_layers: int = field(default=2)
    # Other model configs
    use_cache: bool = field(default=False)
    # Training control flags
    train_buffer: bool = field(default=False)


@dataclass
class DataArguments:
    image_size: int = field(default=512)
    patch_size: int = field(default=16)
    max_pixels: int = field(default=262144)
    min_pixels: int = field(default=12544)
    dataset_use: str = field(default="sbu_captions")
    data_flatten: bool = field(default=True)
    downsample_ratio: float = field(default=0.5)
    dynamic_image_size: str = field(default="native_resolution")
    max_seq_length: int = field(default=8192)
    loss_reduction: str = field(default="square")


@dataclass
class TrainingArguments(HfTrainingArguments):
    seed: int = field(default=42)
    cache_dir: Optional[str] = field(default=None)
    dataloader_num_workers: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_epsilon: float = field(default=1e-8)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    min_lr_ratio: float = field(default=0.0)
    warmup_ratio: float = field(default=0.0)
    warmup_steps: int = field(default=0)
    max_steps: int = field(default=200000)
    gradient_accumulation_steps: int = field(default=1)
    max_grad_norm: float = field(default=1.0)
    save_steps: int = field(default=5000)
    save_total_limit: Optional[int] = field(default=None)
    logging_steps: int = field(default=10)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.lr_scheduler_type == "cosine_with_min_lr":
            if self.lr_scheduler_kwargs is None:
                self.lr_scheduler_kwargs = {}
            self.lr_scheduler_kwargs["min_lr"] = self.learning_rate * self.min_lr_ratio
