"""Model registry: which foundation models Scalpel can operate on, and loading.

The initial product surface targets CLIP-family dual encoders (text tower).
The registry abstraction exists so additional architectures (e.g. SigLIP,
open_clip checkpoints, LLM decoders) can be onboarded without touching the
pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import CLIPModel, CLIPTokenizerFast

_SUPPORTED: dict[str, dict] = {
    "openai/clip-vit-base-patch32": {
        "family": "clip",
        "description": "CLIP ViT-B/32 dual encoder (OpenAI)",
    },
    "openai/clip-vit-base-patch16": {
        "family": "clip",
        "description": "CLIP ViT-B/16 dual encoder (OpenAI)",
    },
    "openai/clip-vit-large-patch14": {
        "family": "clip",
        "description": "CLIP ViT-L/14 dual encoder (OpenAI)",
    },
}


def supported_models() -> dict[str, dict]:
    """Catalog of models the platform officially supports."""
    return dict(_SUPPORTED)


@dataclass
class LoadedModel:
    """A loaded CLIP model plus the handles the pipeline needs."""

    model_id: str
    model: CLIPModel
    tokenizer: CLIPTokenizerFast
    device: torch.device

    @property
    def text_layers(self):
        return self.model.text_model.encoder.layers

    @property
    def num_layers(self) -> int:
        return len(self.text_layers)

    @property
    def num_heads(self) -> int:
        return self.model.config.text_config.num_attention_heads

    @property
    def d_model(self) -> int:
        return self.model.config.text_config.hidden_size

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def eot_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Index of the EOT token per sequence (CLIP pools text at EOT).

        Uses the tokenizer's eos id: legacy CLIP checkpoints ship a stale
        ``text_config.eos_token_id`` (2 instead of 49407). CLIP pads with the
        EOT token too, so we take the *first* occurrence per row.
        """
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:  # pragma: no cover - defensive
            eos_id = self.model.config.text_config.eos_token_id
        return (input_ids == eos_id).int().argmax(dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """L2-normalized text embeddings in the shared CLIP embedding space."""
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenize(texts[start : start + batch_size])
            features = self.model.get_text_features(**batch)
            if not isinstance(features, torch.Tensor):
                # transformers >= 5 returns a model output whose pooler_output
                # holds the projected text features.
                features = features.pooler_output
            chunks.append(features / features.norm(dim=-1, keepdim=True))
        return torch.cat(chunks, dim=0)


def load_model(
    model_id: str,
    device: str = "cpu",
    model: CLIPModel | None = None,
    tokenizer: CLIPTokenizerFast | None = None,
) -> LoadedModel:
    """Load a supported model (or wrap injected instances, used in tests)."""
    if model is None or tokenizer is None:
        if model_id not in _SUPPORTED:
            raise ValueError(
                f"Model '{model_id}' is not in the supported catalog: {sorted(_SUPPORTED)}"
            )
        model = CLIPModel.from_pretrained(model_id)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    return LoadedModel(model_id=model_id, model=model, tokenizer=tokenizer, device=torch_device)
