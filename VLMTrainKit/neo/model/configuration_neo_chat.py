import copy

from transformers import Qwen3Config, Qwen3VLConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_neo_llm import NEOLLMConfig
from .configuration_neo_vit import NEOVisionConfig

logger = logging.get_logger(__name__)


class NEOChatConfig(PretrainedConfig):
    model_type = "neo_chat"
    sub_configs = {"vision_config": NEOVisionConfig, "llm_config": NEOLLMConfig}

    def __init__(
        self,
        img_start_token_id=None,
        img_context_token_id=None,
        vision_config: NEOVisionConfig = None,
        llm_config: NEOLLMConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {"architectures": ["NEOVisionModel"]}
            logger.info(
                "vision_config is None. Initializing the NEOVisionConfig with default values."
            )
        if llm_config is None:
            llm_config = {"architectures": ["Qwen3ForCausalLM"]}
            logger.info(
                "llm_config is None. Initializing the NeoLLMConfig config with default values (`Qwen3ForCausalLM`)."
            )
        assert (
            "architectures" in llm_config
        ), "Should specify architecture in llm_config"

        if isinstance(vision_config, dict):
            self.vision_config = NEOVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
        if isinstance(llm_config, dict):
            self.llm_config = NEOLLMConfig(**llm_config)
        else:
            self.llm_config = llm_config

        self.img_start_token_id = img_start_token_id
        self.img_context_token_id = img_context_token_id
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

        # Expose hidden_size from llm_config for DeepSpeed auto configuration
        self.hidden_size = self.llm_config.hidden_size

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type

        return output


if __name__ == "__main__":
    vision_config = NEOVisionConfig()
    llm_config = NEOLLMConfig()
    chat_config = NEOChatConfig(
        vision_config=vision_config,
        llm_config=llm_config,
        use_backbone_lora=True,
        use_llm_lora=False,
        downsample_ratio=0.5,
    )

    print(chat_config)
