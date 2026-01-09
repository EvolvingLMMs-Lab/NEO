from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_neo_chat import NEOChatConfig
from .conversation import get_conv_template
from .modeling_neo_qwen3 import Qwen3ForCausalLM
from .modeling_neo_vit import NEOVisionModel

try:
    from torch.nn.attention.flex_attention import (_mask_mod_signature,
                                                   and_masks,
                                                   create_block_mask, or_masks)
except ImportError:
    print("To enable flexattention, please install torch>=2.5.0")

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op="eq"):
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def calculate_pad_length(seqlen, div_num):
    """
    calculate the min padding_length,  make (seqlen + padding_length) can be divisible by div_num

    :param seqlen: int, 序列长度
    :param div_num: int, 整数
    :return: int, 最小填充长度
    """
    if seqlen % div_num == 0:
        return 0
    else:
        padding_length = div_num - (seqlen % div_num)
        return padding_length


def _offsets_to_doc_ids_tensor(offsets, has_pad, split_size=1024):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]

    if has_pad:
        tmp_counts = counts[:-1]
        last_counts = counts[-1]

        num_split = last_counts // split_size
        remainder = last_counts % split_size

        split_counts = [split_size] * num_split
        if remainder > 0:
            split_counts.append(remainder)

        counts = torch.cat(
            [
                tmp_counts,
                torch.LongTensor(split_counts).to(
                    dtype=tmp_counts.dtype, device=tmp_counts.device
                ),
            ],
            dim=-1,
        )

    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def create_flex_mask_padding(document_ids, modality_indicators, div_num):
    """
    Current version:
    1. document attention
    2. within each document, causal attention. Within a same image, full attention
    seqlen padded to divisable by some number
    """
    slen = document_ids.size(-1)
    padding_length = calculate_pad_length(seqlen=slen, div_num=div_num)
    if padding_length > 0:
        pad_doc_id = document_ids.max() + 1
        document_ids = F.pad(document_ids, (0, padding_length), value=pad_doc_id)
        modality_indicators = F.pad(modality_indicators, (0, padding_length), value=-1)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def samedoc_mask(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx]

    def sameimg_mask(b, h, q_idx, kv_idx):
        is_image = modality_indicators[q_idx] > 0
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        return (
            is_image
            & (modality_indicators[q_idx] == modality_indicators[kv_idx])
            & same_doc
        )

    samedoc_causal_mask = and_masks(causal_mask, samedoc_mask)
    mask_mod = or_masks(samedoc_causal_mask, sameimg_mask)
    
    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=slen + padding_length,
        KV_LEN=slen + padding_length,
        BLOCK_SIZE=128,
        # disable torch compile in debugging mode to avoid potential issues
        _compile=False,
    )

    return block_mask, padding_length


def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """
    Compute patch coordinates (x, y)

    Args:
        grid_hw: (B, 2) tensor representing (H, W) per image
    """
    device = grid_hw.device
    B = grid_hw.shape[0]

    # Get the number of patches per image
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(
        torch.arange(B, device=device), N
    )  # (N_total,)

    # Generate intra-image patch index (row-major order)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = (
        patch_id_within_image
        - torch.cumsum(torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0)[
            patch_to_sample
        ]
    )

    # Get H/W for each patch according to its image
    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


class NEOChatModel(PreTrainedModel):
    config_class = NEOChatConfig
    # main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "NEOVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ""

    def __init__(
        self,
        config: NEOChatConfig,
        vision_model=None,
        language_model=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")

        self.patch_size = config.vision_config.patch_size
        self.downsample_ratio = config.vision_config.downsample_ratio

        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = NEOVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = Qwen3ForCausalLM(config.llm_config)

        self.img_context_token_id = None
        self.img_start_token_id = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # input_ids: (Seq_len,)
        indexes: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[List[torch.Tensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seq_boundaries: Optional[torch.LongTensor] = None,
        image_grid_hw: Optional[torch.LongTensor] = None,
        loss_weight: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        assert (
            self.img_context_token_id is not None
        ), "img_context_token_id should not be None."
        assert image_grid_hw is not None, "image_grid_hw should not be None."
        if pixel_values is None:
            grid_size = int(1 / self.downsample_ratio)
            pixel_values = [
                torch.rand(
                    grid_size**2,
                    3 * self.patch_size * self.patch_size,
                    device=self.device,
                    dtype=self.dtype,
                )
            ]
            grid_hw = torch.tensor([[grid_size, grid_size]], device=self.device)
        else:
            grid_hw = image_grid_hw[0]

        pixel_values = pixel_values[0].to(self.dtype)
        vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)
        hidden_states = self.language_model.get_input_embeddings()(input_ids)
        selected = input_ids == self.img_context_token_id
        vit_embeds = vit_embeds.reshape((-1, vit_embeds.shape[-1]))

        abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
            grid_hw // int(1 / self.downsample_ratio), device=self.device
        )
        pos_h = torch.zeros_like(indexes)
        pos_w = torch.zeros_like(indexes)
        pos_h[selected[0]] = abs_pos_h.to(dtype=pos_h.dtype)
        pos_w[selected[0]] = abs_pos_w.to(dtype=pos_w.dtype)
        indexes = torch.stack([indexes, pos_h, pos_w], dim=-1)

        img_start_flags = (input_ids[0] == self.img_start_token_id).long()
        shifted_flags = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=input_ids.device),
                img_start_flags,
            ],
            dim=0,
        )[:-1]

        hidden_states = hidden_states.clone()
        hidden_states[selected] = vit_embeds

        modality_indicators = shifted_flags.cumsum(0)
        modality_indicators[input_ids[0] != self.img_context_token_id] = -1

        document_ids = _offsets_to_doc_ids_tensor(
            seq_boundaries, has_pad=False
        )
        attention_mask, padding_length = create_flex_mask_padding(
            document_ids, modality_indicators=modality_indicators, div_num=128
        )

        llm_outputs = self.language_model(
            inputs_embeds=hidden_states,
            labels=labels,
            indexes=indexes,
            padding_length=padding_length,
            attention_mask=attention_mask,
            inference_params=None,
            loss_weight=loss_weight,
        )

        return llm_outputs

    def extract_feature(self, pixel_values, grid_hw=None):
        return self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
            grid_hw=grid_hw,
        ).last_hidden_state

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        grid_hw=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            print(f"dynamic image size: {grid_hw * self.patch_size}")

        for i in range(grid_hw.shape[0]):
            num_patch_token = int(
                grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2
            )
            image_tokens = (
                IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            grid_hw=grid_hw,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        grid_hw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        assert input_ids.shape[0] == 1
        assert self.img_context_token_id is not None
        indexes = self.get_thw_indexes(input_ids[0], grid_hw)
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)

    def get_thw_indexes(self, input_ids, grid_hw):
        img_start_shift = torch.cat(
            [
                torch.zeros(1, dtype=torch.long).to(input_ids.device),
                (input_ids == self.img_start_token_id).long(),
            ],
            dim=0,
        )[:-1]
        not_img_token = (input_ids != self.img_context_token_id).long()
        t_indexes = (img_start_shift + not_img_token).cumsum(0) - 1
        h_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)
        w_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)

        selected = input_ids == self.img_context_token_id
        if selected.long().sum() > 0:
            abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                grid_hw // int(1 / self.downsample_ratio), device=t_indexes.device
            )
            h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
            w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
        return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)


if __name__ == "__main__":
    config = NEOChatConfig()
    model = NEOChatModel(config)
    print(model)
