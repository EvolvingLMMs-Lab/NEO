import json
import random
from dataclasses import dataclass
from functools import partial
from typing import Dict, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.utils import logging

from ..train.argument import DataArguments, TrainingArguments
from . import data_list
from .constants import (ALL_SPECIAL_TOKEN_LIST, IGNORE_INDEX,
                        IMG_CONTEXT_TOKEN, IMG_START_TOKEN)
from .utils import (build_transform, dynamic_preprocess_native_resolution,
                    len2weight, tokenize_mm_chat_conversations)

logger = logging.get_logger(__name__)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        is_train: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.is_train = False

        dataset_list = data_list(data_args.dataset_use.split(","))
        list_data_dict = []
        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                logger.info(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                logger.info(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations
        logger.info(f"Total training samples: {len(list_data_dict)}")
        self.list_data_dict = list_data_dict
        self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                for key, value in sample.items():
                    if hasattr(value, "shape"):
                        logger.info(f"  {key}: shape={value.shape}")
                return sample
            except Exception as e:
                logger.warning(
                    f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception: {e}"
                )

        # If all retries failed, try a random sample
        logger.error(
            f"All {num_base_retries} retries failed for sample {i}, using random fallback"
        )
        random_idx = random.randint(0, len(self.list_data_dict) - 1)
        sources = self.list_data_dict[random_idx]
        if isinstance(sources, dict):
            sources = [sources]
        sample = self.item_fn(sources)
        logger.info(
            f"__getitem__({i}) fallback returning sample with keys: {sample.keys()}"
        )
        return sample

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        assert len(sources) == 1, "Only single source is supported."
        assert (
            self.data_args.dynamic_image_size == "native_resolution"
        ), "Only native resolution is supported."

        patch_size = self.data_args.patch_size
        downsample_ratio = self.data_args.downsample_ratio
        min_pixels = self.data_args.min_pixels
        max_pixels = self.data_args.max_pixels
        source = sources[0]

        if "image" in source:
            image_path_list = (
                source["image"]
                if isinstance(source["image"], list)
                else [source["image"]]
            )
        elif "images" in source:
            image_path_list = source["images"]
        else:
            image_path_list = []

        num_image = len(image_path_list)

        if num_image > 0:
            # hack code to ensure the first human round has image tag
            if num_image == 1:
                for conv in source["conversations"]:
                    if conv["from"] == "human":
                        # the first round for human should have an image
                        if "<image>" not in conv["value"]:
                            conv["value"] = "<image>\n" + conv["value"]
                    break

            transform = build_transform(
                input_size=self.data_args.image_size,
                is_train=self.is_train,
                resize=False,
            )
            images, num_tiles = [], []
            for image_path in image_path_list:
                image = Image.open(image_path).convert("RGB")
                patch = dynamic_preprocess_native_resolution(
                    image,
                    min_pixels=min_pixels,
                    max_pixels=(
                        max_pixels
                        if num_image == 1
                        else max(
                            max_pixels * 2 // num_image,
                            min_pixels,
                        )
                    ),
                    size_factor=int(patch_size / downsample_ratio),
                )
                images.append(patch)
                w, h = patch.size
                num_tiles.append(
                    int(w * h // patch_size**2 * downsample_ratio**2)
                )  # 192 patches
            pixel_values = [transform(image) for image in images]
            num_image_tokens = [num_tile for num_tile in num_tiles]
        else:
            pixel_values = []
            num_image_tokens = []

        res = tokenize_mm_chat_conversations(
            conversations=source["conversations"],
            tokenizer=self.tokenizer,
            num_image_token_list=num_image_tokens,
            num_image=num_image,
        )

        return dict(
            input_ids=res["input_ids"][0],
            labels=res["labels"][0],
            pixel_values=pixel_values,
        )


@dataclass
class FlattenedDataCollatorForSupervisedDataset:
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: PreTrainedTokenizer
    data_args: DataArguments
    training_args: TrainingArguments
    len2weight: callable = None
    total_samples: int = 0
    abnormal_samples: int = 0

    def __post_init__(self):
        if self.len2weight is None:
            self.len2weight = partial(
                len2weight, loss_reduction=self.data_args.loss_reduction
            )

    def compute_packed_sequence_metadata(
        self,
        data_index: torch.LongTensor,  # (seq_len,)
        input_ids: torch.LongTensor,  # (seq_len,)
        labels: torch.LongTensor,  # (seq_len,)
        img_start_token_id: int,
        img_token_id: int,
    ):
        indexes = []
        seq_boundaries = [0]
        loss_weight = []

        start = data_index.min().item()
        end = data_index.max().item() + 1
        for i in range(start, end):
            num_tokens = (data_index == i).sum().item()
            seq_boundaries.append(seq_boundaries[-1] + num_tokens)
            assert num_tokens > 0, "num_tokens should be greater than 0"
            tmp_input_ids = input_ids[seq_boundaries[-2] : seq_boundaries[-1]]
            tmp_img_start_shift = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long),
                    (tmp_input_ids == img_start_token_id).long(),
                ],
                dim=0,
            )[:-1]
            tmp_not_img_token = (tmp_input_ids != img_token_id).long()
            tmp_indexes = (
                (tmp_img_start_shift + tmp_not_img_token).cumsum(0) - 1
            ).tolist()
            indexes.extend(tmp_indexes)

            curr_data_index = data_index[seq_boundaries[-2] : seq_boundaries[-1]]
            assert (curr_data_index == i).all(), data_index

            curr_labels = labels[seq_boundaries[-2] : seq_boundaries[-1]]
            num_effective_tokens = (curr_labels != IGNORE_INDEX).sum().item()
            loss_weight.extend([self.len2weight(num_effective_tokens)] * num_tokens)

        assert len(indexes) == data_index.size(
            0
        ), f"{len(indexes)}, {data_index.size(0)}"

        loss_weight = torch.tensor(loss_weight, dtype=torch.float32)

        return seq_boundaries, indexes, loss_weight

    @staticmethod
    def preprocess_pixel_values(pixel_values, patch_size=16):
        all_flatten_pixel_values = []
        all_grid_hw = []
        for idx, px in enumerate(pixel_values):
            c, h, w = px.shape
            grid_h = h // patch_size
            grid_w = w // patch_size
            flatten_pixel_values = (
                px.view(c, grid_h, patch_size, grid_w, patch_size)
                .permute(1, 3, 0, 2, 4)  # [grid_h, grid_w, c, patch_size, patch_size]
                .reshape(grid_h * grid_w, c * patch_size**2)
            )
            all_flatten_pixel_values.append(flatten_pixel_values)
            all_grid_hw.append([grid_h, grid_w])
        all_flatten_pixel_values = (
            torch.concat(all_flatten_pixel_values, dim=0)
            if len(all_flatten_pixel_values) > 0
            else torch.empty((0, 0))
        )
        all_grid_hw = (
            torch.tensor(all_grid_hw) if len(all_grid_hw) > 0 else torch.empty((0, 2))
        )

        return all_flatten_pixel_values, all_grid_hw

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        current_data_index = 0
        self.total_samples += 1
        (
            input_ids,
            labels,
            pixel_values,
            data_index,
            image_to_instance_map,
        ) = ([], [], [], [], [])

        for instance in instances:
            instance_len = instance["input_ids"].size(0)
            current_data_index_tensor = torch.full(
                (instance_len,), current_data_index, dtype=torch.long
            )
            input_ids.append(instance["input_ids"])
            labels.append(instance["labels"])
            data_index.append(current_data_index_tensor)

            instance_pixel_values = instance["pixel_values"]
            if instance_pixel_values is not None and len(instance_pixel_values) > 0:
                num_images_in_instance = len(instance_pixel_values)
                pixel_values.extend(instance_pixel_values)
                image_to_instance_map.extend(
                    [current_data_index] * num_images_in_instance
                )
            current_data_index += 1

        input_ids = torch.cat(input_ids, dim=-1)
        labels = torch.cat(labels, dim=-1)
        data_index = torch.cat(data_index, dim=-1)

        packed_seq_length = input_ids.size(0)
        if input_ids.size(0) > self.data_args.max_seq_length:
            cutoff_idx = data_index[self.data_args.max_seq_length]
            if cutoff_idx == 0:
                raise RuntimeError(
                    f" Sample length {input_ids.size(0)} exceeds max_seq_length {self.data_args.max_seq_length}"
                )
            else:
                truncate_pos = (
                    (data_index == cutoff_idx).nonzero(as_tuple=True)[0][0].item()
                )

            truncated_data_index = data_index[truncate_pos:]
            input_ids = input_ids[:truncate_pos]
            labels = labels[:truncate_pos]
            data_index = data_index[:truncate_pos]

            if len(truncated_data_index) > 0 and len(pixel_values) > 0:
                truncated_instances = set(truncated_data_index.unique().tolist())
                new_pixel_values = []
                for img_idx, instance_id in enumerate(image_to_instance_map):
                    if instance_id not in truncated_instances:
                        new_pixel_values.append(pixel_values[img_idx])

                num_removed_images = len(pixel_values) - len(new_pixel_values)
                pixel_values = new_pixel_values
                print(
                    f"Removed {num_removed_images} images from {len(truncated_instances)} instances affected by truncation"
                )

            self.abnormal_samples += 1

            print(
                f"Abnormal/Total: {self.abnormal_samples}/{self.total_samples}, "
                f"Batch Size: {self.training_args.per_device_train_batch_size}, "
                f"Packed sequence length: {packed_seq_length}, "
                f"max_seq_length: {self.data_args.max_seq_length}, "
                f"Truncate position: {truncate_pos}, "
                f"Final sequence length: {input_ids.size(0)}",
            )

        seq_boundaries, token_indexes, token_weights = (
            self.compute_packed_sequence_metadata(
                data_index=data_index,
                input_ids=input_ids,
                labels=labels,
                img_token_id=self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
                img_start_token_id=self.tokenizer.convert_tokens_to_ids(
                    IMG_START_TOKEN
                ),
            )
        )
        labels = torch.cat(
            [labels[1:], torch.full(size=(1,), fill_value=IGNORE_INDEX)], dim=-1
        )
        token_weights = torch.cat(
            [
                token_weights[1:],
                torch.zeros(size=(1,), dtype=token_weights.dtype),
            ],
            dim=-1,
        )
        indexes = torch.tensor(token_indexes, dtype=torch.long)
        seq_boundaries = torch.tensor(seq_boundaries, dtype=torch.int32)
        loss_weight = torch.where(
            labels == IGNORE_INDEX, torch.zeros_like(token_weights), token_weights
        )

        if len(pixel_values) > 0:
            images, image_grid_hw = [], []
            flatten_pixel_values, grid_hw = self.preprocess_pixel_values(
                pixel_values, self.data_args.patch_size
            )
            images.append(flatten_pixel_values)
            image_grid_hw.append(grid_hw)
        else:
            images = None
            image_grid_hw = []

        return {
            "input_ids": input_ids.unsqueeze(0),
            "pixel_values": images,
            "labels": labels,
            "seq_boundaries": seq_boundaries,
            "indexes": indexes,
            "image_grid_hw": image_grid_hw,
            "loss_weight": [loss_weight.tolist()],
        }


def make_supervised_data_module(tokenizer, data_args, training_args):
    train_dataset = LazySupervisedDataset(tokenizer, data_args=data_args)
    data_collator = FlattenedDataCollatorForSupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    from types import SimpleNamespace

    from torch.utils.data import DataLoader
    from transformers import AutoProcessor

    data_args = SimpleNamespace(
        dataset_use="sbu_captions%1",
        dynamic_image_size="native_resolution",
        patch_size=16,
        image_size=512,
        down_sample_ratio=0.5,
        max_pixels=262144,
        min_pixels=65536,
        max_seq_length=2048,
        data_flatten=True,
        loss_reduction="square",
    )
    tokenizer_path = ""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    num_new_tokens = tokenizer.add_tokens(ALL_SPECIAL_TOKEN_LIST, special_tokens=True)

    data_module = make_supervised_data_module(tokenizer, data_args)

    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    print(f"Dataset size: {len(train_dataset)}")

    # test single sample
    # print("\nTest getting a single sample:")
    # sample = train_dataset[0]
    # print(f"Sample keys: {sample.keys()}")
    # print(f"input_ids shape: {sample['input_ids'].shape}")
    # print(f"labels shape: {sample['labels'].shape}")
    # if "pixel_values" in sample:
    #     print(f"pixel_values shape: {sample['pixel_values'][0].shape}")

    # ===== test data_collator =====
    print("\nTest data_collator:")
    batch_samples = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    batch = data_collator(batch_samples)
    print(f"Batch keys: {batch.keys()}")
    # ===== test DataLoader =====
    print("\nTest DataLoader:")
    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=0,
    )

    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        if batch.get("pixel_values") is not None:
            print(f"  pixel_values: {batch['pixel_values'].shape}")
        if i >= 0:
            break

    print("\nTest completed!")
