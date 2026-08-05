from __future__ import annotations

import pytest
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPTokenizerFast, CLIPVisionConfig

from scalpel.models.registry import LoadedModel, load_model


@pytest.fixture(scope="session")
def tokenizer() -> CLIPTokenizerFast:
    # Small (~2 MB) tokenizer files; weights are never downloaded in tests.
    return CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")


@pytest.fixture(scope="session")
def tiny_clip(tokenizer) -> LoadedModel:
    torch.manual_seed(0)
    config = CLIPConfig(
        text_config=CLIPTextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            projection_dim=32,
        ).to_dict(),
        vision_config=CLIPVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
            projection_dim=32,
        ).to_dict(),
        projection_dim=32,
    )
    model = CLIPModel(config)
    model.eval()
    return load_model("tiny-clip-test", model=model, tokenizer=tokenizer)
