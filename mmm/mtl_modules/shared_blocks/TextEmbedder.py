import torch

from .SharedBlock import SharedBlock

try:
    from transformers import AutoTokenizer, CLIPTextModel
except ImportError:
    AutoTokenizer, CLIPTextModel = None, None


class TextEmbedder(SharedBlock):
    class Config(SharedBlock.Config):
        module_name: str = "textembedder"
        modelname: str = "openai/clip-vit-base-patch32"
        # pretrained: str = "laion2b_s34b_b79k"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.model = CLIPTextModel.from_pretrained(cfg.modelname)
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.modelname)
        self.make_mtl_compatible()

    def get_hidden_dim(self):
        return self.model.text_model.final_layer_norm.normalized_shape[0]

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """returns a dict of tensors with keys: input_ids, attention_mask"""
        return self._tokenizer(texts, padding=True, return_tensors="pt")

    def forward(self, input_tokens: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_tokens, attention_mask=attention_mask)
        # outputs.last_hidden_state exists as well
        return outputs.pooler_output
