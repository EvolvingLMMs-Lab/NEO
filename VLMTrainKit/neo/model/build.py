from transformers import AutoTokenizer
from transformers.utils import logging

from ..data.constants import (ALL_SPECIAL_TOKEN_LIST, IMG_CONTEXT_TOKEN,
                              IMG_START_TOKEN)
from .configuration_neo_chat import NEOChatConfig
from .configuration_neo_llm import NEOLLMConfig
from .configuration_neo_vit import NEOVisionConfig
from .modeling_neo_chat import NEOChatModel
from .modeling_neo_qwen3 import Qwen3ForCausalLM
from .modeling_neo_vit import NEOVisionModel

# Set logging level to INFO to see all messages in console
logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def build_model(model_args, data_args, tokenizer):
    if model_args.model_name_or_path is not None:
        model = NEOChatModel.from_pretrained(
            model_args.model_name_or_path, dtype=model_args.dtype, device="auto"
        )
        logger.info("Loaded model from pretrained path.")
    else:
        assert (
            model_args.llm_model_name_or_path is not None
        ), "LLM model path must be provided."

        llm_config = NEOLLMConfig.from_pretrained(
            model_args.llm_model_name_or_path,
            rope_theta_hw=model_args.rope_theta_hw,
            max_position_embeddings_hw=model_args.max_position_embeddings_hw,
            extra_num_layers=model_args.extra_num_layers,
            num_hidden_layers=model_args.num_hidden_layers,
        )
        vision_config = NEOVisionConfig(
            llm_hidden_size=llm_config.hidden_size,
            downsample_ratio=data_args.downsample_ratio,
            hidden_size=model_args.vision_hidden_size,
            rope_theta_vision=model_args.rope_theta_vision,
            max_position_embeddings_vision=model_args.max_position_embeddings_vision,
            num_channels=model_args.vision_num_channels,
            patch_size=data_args.patch_size,
            min_pixels=data_args.min_pixels,
            max_pixels=data_args.max_pixels,
        )
        model_config = NEOChatConfig(
            vision_config=vision_config,
            llm_config=llm_config,
            img_start_token_id=tokenizer.convert_tokens_to_ids(IMG_START_TOKEN),
            img_context_token_id=tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
        )

        vision_model = NEOVisionModel(vision_config)
        language_model = Qwen3ForCausalLM.from_pretrained(
            model_args.llm_model_name_or_path,
            config=llm_config,
        )
        model = NEOChatModel(
            config=model_config,
            vision_model=vision_model,
            language_model=language_model,
            dtype=model_args.dtype,
            device="auto",
        )
        logger.info("Building model from configuration.")
    return model


def build_tokenizer(model_args, data_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = data_args.max_seq_length
    tokenizer.add_tokens(ALL_SPECIAL_TOKEN_LIST, special_tokens=True)
    return tokenizer


def build_model_and_tokenizer(model_args, data_args):
    tokenizer = build_tokenizer(model_args, data_args)
    model = build_model(model_args, data_args, tokenizer)

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    return model, tokenizer


if __name__ == "__main__":
    from types import SimpleNamespace

    data_args = SimpleNamespace(
        dataset_use="",
        dynamic_image_size="native_resolution",
        patch_size=16,
        image_size=512,
        downsample_ratio=0.5,
        max_pixels=262144,
        min_pixels=65536,
        max_seq_length=2048,
        data_flatten=True,
        loss_reduction="square",
    )
    model_args = SimpleNamespace(
        model_name_or_path=None,
        tokenizer_path="",
        llm_model_name_or_path="",
        vision_num_channels=3,
        vision_hidden_size=1024,
        vision_llm_hidden_size=2048,
        dtype="bfloat16",
    )
    model = build_model(model_args, data_args)
