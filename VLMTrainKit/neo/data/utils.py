import copy
import io
import math
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import PreTrainedTokenizer

from .constants import (
    CLIP_MEAN,
    CLIP_STD,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    SIGLIP_MEAN,
    SIGLIP_STD,
)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(
                output
            ).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {
    quality: simulate_jpeg_degradation(quality) for quality in qualities
}


def build_transform(
    is_train: bool,
    input_size,
    pad2square: bool = False,
    normalize_type: str = "imagenet",
    resize: bool = True,
):
    if normalize_type == "imagenet":
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:
        if resize:
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.RandomChoice(
                        [
                            T.Lambda(jpeg_degrade_functions[quality])
                            for quality in qualities
                        ]
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:  # only for native resolution
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.RandomChoice(
                        [
                            T.Lambda(jpeg_degrade_functions[quality])
                            for quality in qualities
                        ]
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
    else:
        if pad2square is False:
            if resize:
                transform = T.Compose(
                    [
                        T.Lambda(
                            lambda img: img.convert("RGB") if img.mode != "RGB" else img
                        ),
                        T.Resize(
                            (input_size, input_size),
                            interpolation=InterpolationMode.BICUBIC,
                        ),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD),
                    ]
                )
            else:  # only for native resolution
                transform = T.Compose(
                    [
                        T.Lambda(
                            lambda img: img.convert("RGB") if img.mode != "RGB" else img
                        ),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD),
                    ]
                )

        else:
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Lambda(
                        lambda img: expand2square(
                            img, tuple(int(x * 255) for x in MEAN)
                        )
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
    return transform


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 256 * 32 * 32,
    max_pixels: int = 16384 * 32 * 32,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {200}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def dynamic_preprocess_native_resolution(
    image, size_factor=32, min_pixels=4 * 32 * 32, max_pixels=16384 * 32 * 32, **kwargs
):
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))
    return image


def tokenize_mm_chat_conversations(
    conversations,
    tokenizer: PreTrainedTokenizer,
    num_image_token_list: List[int],
    truncation: bool = False,
    num_image: int = 1,
) -> Dict:
    roles = {
        "human": "user",
        "system": "system",
        "gpt": "assistant",
    }
    conversation_start = "<|im_start|>"
    conversation_end = "<|im_end|>\n"
    num_image = len(num_image_token_list)
    new_conversations = []
    cur_image_idx = 0
    for conv in conversations:
        while "<image>" in conv["value"] and cur_image_idx < num_image:
            image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[cur_image_idx]}{IMG_END_TOKEN}"
            conv["value"] = conv["value"].replace("<image>", image_tokens, 1)
            cur_image_idx += 1
        new_conversations.append(conv)
    conversations = new_conversations

    tokens, labels = [], []
    for _, conv in enumerate(conversations):
        if conv["from"] not in roles:
            print(f"WARNING: Unknown role, skip. {conv}")
            continue
        role = roles[conv["from"]]
        content = conv["value"]
        if role != "assistant" or (role == "assistant" and conv.get("is_input", False)):
            user_info = f"{conversation_start}{role}\n{content}{conversation_end}"
            tokenized_user = tokenizer(
                user_info, return_attention_mask=False, add_special_tokens=False
            )["input_ids"]
            tokens.extend(tokenized_user)
            labels.extend([IGNORE_INDEX] * len(tokenized_user))
        elif role == "assistant":
            assis_start = f"{conversation_start}{role}\n"
            tokens_assistant_start = tokenizer(
                assis_start, return_attention_mask=False, add_special_tokens=False
            )["input_ids"]
            tokens.extend(tokens_assistant_start)
            labels.extend([IGNORE_INDEX] * len(tokens_assistant_start))
            assis_info = f"{content}{conversation_end}"
            tokenized_assistant = tokenizer(
                assis_info, return_attention_mask=False, add_special_tokens=False
            )["input_ids"]
            tokens.extend(tokenized_assistant)
            labels.extend(copy.deepcopy(tokenized_assistant))
        else:
            print(f"Not processed role, skip. {conv}")
    if truncation and len(tokens) > tokenizer.model_max_length:
        tokens = tokens[: tokenizer.model_max_length]
        labels = labels[: tokenizer.model_max_length]
    input_ids = torch.LongTensor([tokens])
    targets = torch.LongTensor([labels])

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)
