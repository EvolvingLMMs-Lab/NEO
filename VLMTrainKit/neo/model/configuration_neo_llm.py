from transformers import Qwen3Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NEOLLMConfig(Qwen3Config):
    def __init__(
        self,
        rope_theta_hw=10000.0,
        max_position_embeddings_hw=10000,
        extra_num_layers: int = 2,
        num_hidden_layers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        self.extra_num_layers = extra_num_layers
        self.num_hidden_layers = num_hidden_layers


if __name__ == "__main__":
    model_path = ""
    llm_config = NEOLLMConfig.from_pretrained(model_path, rope_theta_hw=20000.0)
    print(llm_config)
