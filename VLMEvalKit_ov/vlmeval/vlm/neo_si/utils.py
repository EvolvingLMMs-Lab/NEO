import math
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image

from ...dataset import DATASET_TYPE
from ...smp import *

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_local_rank_and_local_world_size():
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        return 0, 1

    if "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
        local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        return local_rank, local_world_size

    if "LOCAL_RANK" in os.environ and "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"])

    raise NotImplementedError(
        "Fail to get local_rank and local_world_size! "
        "Please ensure that you set the environment variable "
        "`LOCAL_RANK` and `LOCAL_WORLD_SIZE`"
    )


def build_mcq_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding multiple choice question. The last line of your response should follow "
            "this format: 'Answer: \\boxed{$LETTER}' (without quotes), where LETTER is one of the options. "
            "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
            "information provided. Avoid repeating steps indefinitely—provide your best guess even if "
            "unsure. Think step by step logically, considering all relevant information before answering."
        )
    prompt = prompt.replace(
        "Answer with the option's letter from the given choices directly.", ""
    ).strip()
    prompt = prompt + "\n" + cot_prompt

    return prompt


def build_qa_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding question. The last line of your response should follow this format: "
            "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
            "based on the reasoning provided. If you are uncertain or the problem is too complex, make "
            "a reasoned guess based on the information provided. Avoid repeating steps indefinitely—"
            "provide your best guess even if unsure. Think step by step logically, considering all "
            "relevant information before answering."
        )
    prompt = prompt + "\n" + cot_prompt

    return prompt


def build_multi_choice_prompt(line, dataset=None):
    question = line["question"]
    # For MUIRBench, keep <image> placeholders to maintain image positions
    if dataset is None or not listinstr(["MUIRBench"], dataset):
        question = question.replace("<image>", "").strip()
    hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
    if hint is not None:
        question = hint + "\n" + question

    # Try to get options from individual columns (A, B, C, D, ...)
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }

    # If no options found in columns, try to parse from 'options' field (list format)
    if not options and "options" in line and not pd.isna(line["options"]):
        options_data = line["options"]
        # Check if it's a list
        if isinstance(options_data, list):
            for idx, option_text in enumerate(options_data):
                option_letter = string.ascii_uppercase[idx]
                options[option_letter] = str(option_text)
        # If it's a string representation of a list, try to evaluate it
        elif isinstance(options_data, str):
            try:
                import ast

                options_list = ast.literal_eval(options_data)
                if isinstance(options_list, list):
                    for idx, option_text in enumerate(options_list):
                        option_letter = string.ascii_uppercase[idx]
                        options[option_letter] = str(option_text)
            except (ValueError, SyntaxError):
                pass

    # If no options found in columns, try to parse from 'candidates' field
    if not options and "candidates" in line and not pd.isna(line["candidates"]):
        candidates_text = line["candidates"]
        # Parse format like "A. back-left\nB. back\nC. front\nD. back-right"
        for line_text in candidates_text.split("\n"):
            line_text = line_text.strip()
            if (
                line_text
                and len(line_text) > 2
                and line_text[0].isalpha()
                and line_text[1] == "."
            ):
                option_letter = line_text[0].upper()
                option_text = line_text[2:].strip()
                options[option_letter] = option_text

    for key, item in options.items():
        question += f"\n{key}. {item}"
    prompt = question

    if len(options):
        prompt += (
            "\n请直接回答选项字母。"
            if cn_string(prompt)
            else "\nAnswer with the option's letter from the given choices directly."
        )
    else:
        prompt += (
            "\n请直接回答问题。"
            if cn_string(prompt)
            else "\nAnswer the question directly."
        )

    return prompt


def build_video_prompt(prompt, dataset=None, max_frames=64):
    for start in range(0, max_frames, 8):
        images_to_remove = "".join(
            [f"<Image-{i}>" for i in range(start + 1, start + 9)]
        )
        prompt = prompt.replace(images_to_remove, "")
    for i in range(max_frames):
        prompt = prompt.replace(f"Image-{i + 1}", f"Frame-{i + 1}")
    if listinstr(["MMBench-Video"], dataset):
        prompt = prompt.replace("\nAnswer:", "")
    elif listinstr(["Video-MME", "WorldSense"], dataset):
        prompt = prompt.replace("\nAnswer:", "")
        prompt += "\nAnswer with the option's letter from the given choices directly."
    elif listinstr(["MVBench"], dataset):
        prompt = prompt.replace("Best option:(", "")

    return prompt


def reorganize_prompt(message, image_num, dataset=None):
    if dataset is not None and listinstr(["MUIRBench"], dataset):
        # Option 1: Keep interleaved structure (currently commented out)
        # prompt = ""
        # image_idx = 1
        # for x in message:
        #     if x["type"] == "text":
        #         prompt += x["value"]
        #     elif x["type"] == "image":
        #         prompt += f"<image>"
        #         image_idx += 1

        # Option 2: Put all images at the beginning with numbered placeholders (currently active)
        prompt = ""
        image_idx = 1
        for x in message:
            if x["type"] == "text":
                prompt += x["value"]
            elif x["type"] == "image":
                prompt += f"<Image-{image_idx}>"
                image_idx += 1

        # Add all images at the beginning
        prompt = (
            "".join([f"<Image-{i + 1}>: <image>\n" for i in range(image_num)]) + prompt
        )
    elif dataset is not None and listinstr(["bmmr"], dataset.lower()):
        if image_num == 1:
            prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
        else:
            prompt, image_idx = "", 1
            for x in message:
                if x["type"] == "text":
                    prompt += x["value"]
                elif x["type"] == "image":
                    image_idx += 1
    elif image_num == 1:
        prompt = "<image>\n" + "\n".join(
            [x["value"] for x in message if x["type"] == "text"]
        )
    else:
        try:
            ### only for video training
            prompt = ""
            image_idx = 1
            pairs = []  # list of (timestamp, image_tag)

            i = 0
            while i < len(message):
                if message[i]["type"] == "image":
                    img_tag = f"<Image-{image_idx}>"
                    image_idx += 1

                    # 下一个必须是 text（timestamp）
                    ts = message[i + 1]["value"].strip()
                    pairs.append((ts, img_tag))
                    i += 2
                else:
                    # 正常文本
                    if message[i]["value"].startswith(
                        "The time range of this video is"
                    ):
                        temporal_instruction = message[i]["value"]
                    else:
                        question = message[i]["value"]
                    i += 1

            # 构造 special tokens
            special_tokens = "\n".join(
                [
                    f"Frame{i+1} is at [{ts}]:<image>"
                    for i, (ts, img_tag) in enumerate(pairs)
                ]
            )
            prompt = temporal_instruction + special_tokens + "\n" + question

        except Exception as e:
            prompt, image_idx = "", 1
            for x in message:
                if x["type"] == "text":
                    prompt += x["value"]
                elif x["type"] == "image":
                    prompt += f"<Image-{image_idx}>"
                    image_idx += 1

            prompt = (
                "".join([f"Image-{i + 1}: <image>\n" for i in range(image_num)])
                + prompt
            )
            images_to_remove = "".join([f"<Image-{i + 1}>" for i in range(image_num)])
            prompt = prompt.replace(images_to_remove, "")

    return prompt


mpo_prompt_with_final_answer = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    'please use the format "Final answer: .."'
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)

mpo_prompt_without_final_answer = (
    "Your task is to answer the question below. "
    "Give step by step reasoning. "
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)


def mpo_post_processing(response, dataset):

    def extract_answer(text):
        match = re.search(r"(Final answer:|Answer:)\s*(.*)", text, re.IGNORECASE)
        if match:
            return match.group(2).strip()
        return text

    if dataset is not None and (
        DATASET_TYPE(dataset) in ["Y/N", "MCQ"] or listinstr(["CRPE"], dataset)
    ):
        response = extract_answer(response).strip()
    return response


def parse_bbox_vl(response):
    # 使用正则表达式匹配bounding box
    # pattern = r"<box>\[\[(\d+), (\d+), (\d+), (\d+)\]\]</box>"
    pattern = r"\[\[(\d+), (\d+), (\d+), (\d+)\]\]"
    match = re.search(pattern, response)
    if match:
        # 提取匹配到的坐标值并转换为整数
        x1, y1, x2, y2 = map(int, match.groups())
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    else:
        return response


def build_mpo_prompt(message, line, dataset):
    if listinstr(["LLaVABench", "MMVet"], dataset):
        return message

    question_orig = line["question"]
    if listinstr(["MathVerse", "MathVision"], dataset):
        question_orig = question_orig.split("Question:", 1)[-1].strip()
        question_orig = question_orig.replace("Choices:\n", "").strip()
    if listinstr(["WeMath"], dataset):
        question_orig = question_orig.replace(
            "Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n<Thought process>: <<your thought process>> <Answer>: <<your option>>",
            "",
        ).strip()  # noqa: E501

    # Try to get options from individual columns (A, B, C, D, ...)
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }

    # If no options found in columns, try to parse from 'candidates' field
    if not options and "candidates" in line and not pd.isna(line["candidates"]):
        candidates_text = line["candidates"]
        # Parse format like "A. back-left\nB. back\nC. front\nD. back-right"
        for line_text in candidates_text.split("\n"):
            line_text = line_text.strip()
            if (
                line_text
                and len(line_text) > 2
                and line_text[0].isalpha()
                and line_text[1] == "."
            ):
                option_letter = line_text[0].upper()
                option_text = line_text[2:].strip()
                options[option_letter] = option_text

    options_prompt = ""
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"

    if options_prompt.strip():
        question_orig = f"{question_orig}\n{options_prompt}"

    cot_prompt = mpo_prompt_with_final_answer
    prompt = cot_prompt.format(question=question_orig).strip()
    message[0]["value"] = prompt
    return message


def format_nav_prompt(template, placeholders, **kwargs):
    prompt = template
    for placeholder in placeholders:
        value = kwargs.get(placeholder, "")
        prompt = prompt.replace(f"{{{placeholder}}}", str(value))
    return prompt


def pile_action_history(history, max_num=4):
    if len(history) > 0:
        return "\n".join(history[-max_num:])
    else:
        return "None"


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


# copy from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L60
def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
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
    image, size_factor=32, min_pixels=65536, max_pixels=4194304, **kwargs
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


def preprocess_pixel_values(pixel_values, patch_size=16):
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size

    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)  # [grid_h, grid_w, c, patch_size, patch_size]
        .reshape(grid_h * grid_w, c * patch_size**2)
    )

    grid_hw = torch.tensor([[grid_h, grid_w]]).to(device=pixel_values.device)

    return flatten_pixel_values, grid_hw


def get_contrasting_background(image):
    """
    Calculate the color (white or black) that is different from the average foreground color
    to use as the background color
    """
    image_np = np.array(image)
    if (image_np[:, :, 3] == 0).any():
        non_transparent_pixels = image_np[:, :, :3][image_np[:, :, 3] > 0]
        if non_transparent_pixels.size == 0:
            return None
        pixel_mean = non_transparent_pixels.mean()
        contrasting_color = (0, 0, 0) if pixel_mean > 382.5 else (255, 255, 255)
        return contrasting_color
    else:
        return None


def load_image_native(
    image_file,
    patch_size=16,
    downsample_ratio=0.5,
    min_pixels=65536,
    max_pixels=4194304,
    upscale=False,
):
    """
    Load and preprocess an image file, converting it to RGB mode,
    resizing, normalizing, and optionally adding a thumbnail version.
    """
    image = Image.open(image_file)
    if image.mode == "RGBA":
        bg_color = get_contrasting_background(image)
        if bg_color:
            background = Image.new("RGB", image.size, bg_color)
            background.paste(image, mask=image.split()[3])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    new_image = dynamic_preprocess_native_resolution(
        image,
        size_factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    pixel_values, grid_hw = preprocess_pixel_values(
        transform(new_image).to(torch.float32), patch_size=patch_size
    )

    print(
        f"Transfer image_size from ({image.height, image.width}) to ({new_image.height, new_image.width})"
    )

    return pixel_values, grid_hw
