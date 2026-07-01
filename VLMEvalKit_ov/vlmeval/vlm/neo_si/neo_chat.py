import re
import warnings

import torch
import transformers
import yaml
from transformers import AutoModel, AutoTokenizer

from ...dataset import (
    DATASET_MODALITY,
    DATASET_TYPE,
    build_dataset,
    infer_dataset_basename,
)
from ...smp import *
from ..base import BaseModel
from .utils import (
    build_mcq_cot_prompt,
    build_mpo_prompt,
    build_multi_choice_prompt,
    build_qa_cot_prompt,
    build_video_prompt,
    format_nav_prompt,
    load_image_native,
    mpo_post_processing,
    parse_bbox_vl,
    pile_action_history,
    reorganize_prompt,
)

# load all the gui templates
upper_path = Path(__file__).parent
with open(os.path.join(upper_path, "gui_template.yaml"), "r") as f:
    GUI_TEMPLATE = yaml.load(f, Loader=yaml.FullLoader)

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()


def extract_boxed_content(ans: str):
    idx = ans.rfind(r"\boxed{")
    if idx == -1:
        return ans

    idx += len(r"\boxed{")
    brace_level = 1
    content_start = idx
    i = idx

    while i < len(ans):
        if ans[i] == "{":
            brace_level += 1
        elif ans[i] == "}":
            brace_level -= 1
            if brace_level == 0:
                break
        i += 1

    if brace_level != 0:
        # Unbalanced braces
        return ans

    content = ans[content_start:i]
    return content


class NEOChatSI(BaseModel):
    """NEO chat model configured for spatial-intelligence (SI) benchmarks.

    This is a self-contained variant of the understanding-oriented ``NEOChat``.
    It differs in its dataset prompt handling (spatial MCQ ``candidates``/
    ``options`` parsing, MUIRBench interleaved messages, spatial video prompts)
    and is kept separate so the two evaluation paths never mix.
    """

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path=None,
        load_in_8bit=False,
        use_mpo_prompt=False,
        screen_parse=True,
        # model parameters
        patch_size=16,
        min_pixels=65536,
        max_pixels=4194304,
        downsample_ratio=0.5,
        # Best-of-N parameters
        best_of_n=1,
        reward_model_path=None,
        # R1 parameters
        cot_prompt_version="v1",
        # inference parameters
        use_postprocess=False,
        max_new_tokens=4096,
        **kwargs,
    ):

        assert best_of_n == 1
        assert model_path is not None
        assert version_cmp(transformers.__version__, "4.37.2", "ge")

        self.cot_prompt_version = cot_prompt_version
        self.use_mpo_prompt = use_mpo_prompt
        self.use_cot = os.getenv("USE_COT") == "1"
        self.use_postprocess = use_postprocess

        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if cot_prompt_version == "r1":
            self.system_prompt = R1_SYSTEM_PROMPT
            self.cot_prompt = (
                "Please answer the question and put the final answer within \\boxed{}."
            )
        elif cot_prompt_version == "v2":
            self.system_prompt = None
            self.cot_prompt = "Answer the preceding multiple-choice question \
            by carefully analyzing the provided image. \nPlease answer with \
            carefully thought step by step. Apply the thinking process \
            recursively at both macro and micro levels. \nVerify consistency \
            of reasoning and look for potential flaws or gaps during \
            thinking. \nWhen realize mistakes, explain why the previous \
            thinking was incorrect, fix it and then continue thinking.\nThe \
            last line of your response should follow this format: 'Answer: \
            \\boxed{$LETTER}' (without quotes), where LETTER is one of the \
            options\n\n"
        else:
            assert cot_prompt_version == "v1"
            self.system_prompt = ""
            self.cot_prompt = None

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r"Image(\d+)"
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r"Image-\1"

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r"Image-(\d+)"
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r"Image\1"

        self.screen_parse = screen_parse

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        self.device = "cuda"

        if best_of_n > 1:
            assert reward_model_path is not None

            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                reward_model_path, trust_remote_code=True, use_fast=False
            )
            self.reward_model = AutoModel.from_pretrained(
                reward_model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",
            ).eval()

            if not self.use_cot:
                os.environ["USE_COT"] = "1"
                self.use_cot = True
                print(
                    "[Warning] Since Best-of-N is enabled, USE_COT is forced to be set to 1."
                )

            print(f"Enable Best-of-N evaluation with PRM: {reward_model_path}")

        self.best_of_n = best_of_n
        kwargs_default = dict(
            do_sample=False, max_new_tokens=max_new_tokens, top_p=None
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(
            [
                "MMSIBench_wo_circular",
                "ViewSpatialBench",
                "MUIRBench",
                "SiteBenchImage",
                "OmniSpatialBench_manual_cot",
                "MUIRBench_EASI",
            ],
            dataset,
        ):
            return False
        if DATASET_MODALITY(dataset) == "VIDEO":
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_mpo_prompt = self.use_mpo_prompt and (
            self.use_cot or dataset in ["MMStar", "HallusionBench", "OCRBench"]
        )

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        if dataset is not None and listinstr(["BMMR"], dataset):
            self.kwargs["max_new_tokens"] = max(
                self.kwargs.get("max_new_tokens", 4096), 8196
            )
            print(
                f'[Warning] BMMR dataset requires a larger max_new_tokens, set to {self.kwargs["max_new_tokens"]}'
            )

        if dataset is not None and DATASET_TYPE(dataset) == "Y/N":
            question = line["question"]
            if listinstr(["MME"], dataset):
                prompt = (
                    question + " Answer the question using a single word or phrase."
                )
            elif listinstr(["HallusionBench", "AMBER"], dataset):
                prompt = (
                    question
                    + " Please answer yes or no. Answer the question using a single word or phrase."
                )
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == "MCQ":
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv("USE_COT") == "1":
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == "VQA":
            question = line["question"]
            if listinstr(["LLaVABench", "WildVision"], dataset):
                prompt = question + "\nAnswer this question in detail."
            elif listinstr(
                [
                    "OCRVQA",
                    "TextVQA",
                    "ChartQA",
                    "DocVQA",
                    "InfoVQA",
                    "OCRBench",
                    "DUDE",
                    "SLIDEVQA",
                    "GQA",
                    "MMLongBench_DOC",
                ],
                dataset,
            ):
                prompt = (
                    question + "\nAnswer the question using a single word or phrase."
                )
            elif listinstr(["MathVerse"], dataset):
                question = question.replace(
                    "please directly answer the question and", "please"
                )
                prompt = question
                if os.getenv("USE_COT") == "1":
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            elif listinstr(
                [
                    "MathVista",
                    "MathVision",
                    "VCR",
                    "MTVQA",
                    "MMVet",
                    "MMDU",
                    "CRPE",
                    "MIA-Bench",
                    "MM-Math",
                    "DynaMath",
                    "QSpatial",
                    "WeMath",
                    "LogicVista",
                    "MM-IFEval",
                    "ChartMimic",
                ],
                dataset,
            ):
                prompt = question
                if os.getenv("USE_COT") == "1":
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = (
                    question + "\nAnswer the question using a single word or phrase."
                )
        elif dataset is not None and DATASET_TYPE(dataset) == "GUI":
            ds_basename = infer_dataset_basename(dataset)
            ds = build_dataset(dataset, skeleton=True)
            action_space = ds.get_action_space()
            traj_dict = ds.get_trajectory(line)

            prompt_config = GUI_TEMPLATE[ds_basename]
            if "history" in prompt_config["placeholders"]:
                traj_dict["history"] = pile_action_history(traj_dict["history"])
            prompt = format_nav_prompt(
                (
                    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{task}</ref>"  # noqa: E501
                    if self.screen_parse
                    else prompt_config["template"]
                ),
                prompt_config["placeholders"],
                action_space=action_space,
                **traj_dict,
            )
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line["question"]
            if os.getenv("USE_COT") == "1":
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        # For MUIRBench, build interleaved message based on <image> placeholders
        if dataset is not None and listinstr(["MUIRBench"], dataset):
            message = []
            parts = prompt.split("<image>")
            image_idx = 0

            for i, part in enumerate(parts):
                if part:  # Add non-empty text parts
                    message.append(dict(type="text", value=part))
                # Add image after each text part (except the last one)
                if i < len(parts) - 1 and image_idx < len(tgt_path):
                    message.append(dict(type="image", value=tgt_path[image_idx]))
                    image_idx += 1
        else:
            message = [dict(type="text", value=prompt)]
            message.extend([dict(type="image", value=s) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)
        return message

    def set_max_num(self, dataset):
        # The total limit on the number of images processed, set to avoid Out-of-Memory issues.
        if dataset is None:
            return None

        if DATASET_MODALITY(dataset) == "VIDEO":
            return None

        if listinstr(["OCRBench"], dataset):
            self.min_pixels = 512 * 512  # 256 * 256    10 * 10 * 32 * 32
            print(f"transfer min_pixels to {self.min_pixels}")

    @torch.no_grad()
    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        use_mpo_prompt = self.use_mpo_prompt and (
            self.use_cot or dataset in ["MMStar", "HallusionBench", "OCRBench"]
        )

        image_num = len([x for x in message if x["type"] == "image"])
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        dataset_modality = DATASET_MODALITY(dataset) if dataset is not None else None

        if dataset is not None and dataset_modality == "VIDEO":
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x["value"] for x in message if x["type"] == "image"]
            grid_hw_list, pixel_values_list = [], []

            for image_idx, file_name in enumerate(image_path):
                upscale_flag = (
                    image_idx == 0
                    and dataset is not None
                    and listinstr(["MMMU"], dataset)
                )
                curr_pixel_values, curr_grid_hw = load_image_native(
                    file_name,
                    patch_size=self.patch_size,
                    downsample_ratio=self.downsample_ratio,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    upscale=upscale_flag,
                )
                grid_hw_list.append(curr_grid_hw.to(self.device))
                pixel_values_list.append(
                    curr_pixel_values.to(self.device).to(torch.bfloat16)
                )
            grid_hw = torch.cat(grid_hw_list, dim=0)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x["value"] for x in message if x["type"] == "image"][0]
            upscale_flag = dataset is not None and listinstr(["MMMU"], dataset)
            pixel_values, grid_hw = load_image_native(
                image_path,
                patch_size=self.patch_size,
                downsample_ratio=self.downsample_ratio,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                upscale=upscale_flag,
            )
            grid_hw = grid_hw.to(self.device)
            pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
        else:
            grid_hw = None
            pixel_values = None

        response_list = []
        for idx in range(self.best_of_n):
            kwargs_default = self.kwargs.copy()

            if self.system_prompt is not None:
                self.model.system_message = self.system_prompt
            _non_generate_keys = {
                "add_special_tokens",
                "system_prompt",
                "remove_think",
                "enable_thinking",
            }
            gen_config = {
                k: v
                for k, v in kwargs_default.items()
                if k not in _non_generate_keys
            }
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
                question=prompt,
                generation_config=gen_config,
                verbose=idx == 0,
            )
            response_list.append(response)

        if self.best_of_n > 1:
            response_list = self.reward_model.select_best_response(
                tokenizer=self.reward_tokenizer,
                question=prompt,
                response_list=response_list,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
            )
        response = response_list[0]

        if dataset is not None and not listinstr(["WeMath"], dataset):
            if use_mpo_prompt:
                response = mpo_post_processing(response, dataset)
            elif self.use_cot and self.use_postprocess:
                response = extract_boxed_content(response)

        if dataset is not None and DATASET_TYPE(dataset) == "GUI" and self.screen_parse:
            # Parse the bounding box coordinates from the response
            response = parse_bbox_vl(response)
            # Normalize the coordinates to the range [0, 1]
            if isinstance(response, list):
                response = [item / 1000 for item in response]
                # Convert the coordinates to the format required by the GUI
                response = f"x={response[0]}, y={response[1]}"

        return response
