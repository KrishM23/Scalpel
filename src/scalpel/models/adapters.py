"""Architecture adapters: uniform text-tower interface across HF model families.

Scalpel's surgery pipeline needs a residual stream, attention out-projections,
and MLP down-projections. Hugging Face names those modules differently per
architecture; adapters normalize them into ``LayerView`` handles so the rest
of the platform stays architecture-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
import torch.nn as nn

WeightLayout = Literal["linear", "conv1d"]
Family = Literal["clip", "causal_lm", "text_encoder"]


@dataclass(frozen=True)
class LayerView:
    """One transformer block's editable write modules."""

    attn_out: nn.Module
    mlp_out: nn.Module
    attn_path: str
    mlp_path: str
    layout: WeightLayout = "linear"

    @property
    def attn_weight(self) -> torch.Tensor:
        return self.attn_out.weight

    @property
    def mlp_weight(self) -> torch.Tensor:
        return self.mlp_out.weight

    @property
    def attn_bias(self) -> torch.Tensor | None:
        return getattr(self.attn_out, "bias", None)

    @property
    def mlp_bias(self) -> torch.Tensor | None:
        return getattr(self.mlp_out, "bias", None)


@dataclass(frozen=True)
class ArchitectureSpec:
    """How to walk a loaded HF module tree for a given ``model_type``."""

    family: Family
    layers_of: Callable[[nn.Module], Any]
    attn_out_of: Callable[[nn.Module], nn.Module]
    mlp_out_of: Callable[[nn.Module], nn.Module]
    layer_prefix: str
    attn_name: str
    mlp_name: str
    layout: WeightLayout = "linear"
    num_heads_of: Callable[[Any], int] | None = None
    hidden_of: Callable[[Any], int] | None = None


def _clip_layers(model: nn.Module):
    return model.text_model.encoder.layers


def _gpt2_layers(model: nn.Module):
    return model.transformer.h


def _llama_layers(model: nn.Module):
    # Covers llama / mistral / qwen2 / gemma / phi3-style roots.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("could not locate transformer layers")


def _bert_layers(model: nn.Module):
    return model.encoder.layer


def _roberta_layers(model: nn.Module):
    return model.encoder.layer


# model_type → how to find editable modules
_SPECS: dict[str, ArchitectureSpec] = {
    "clip": ArchitectureSpec(
        family="clip",
        layers_of=_clip_layers,
        attn_out_of=lambda layer: layer.self_attn.out_proj,
        mlp_out_of=lambda layer: layer.mlp.fc2,
        layer_prefix="text_model.encoder.layers",
        attn_name="self_attn.out_proj",
        mlp_name="mlp.fc2",
    ),
    "gpt2": ArchitectureSpec(
        family="causal_lm",
        layers_of=_gpt2_layers,
        attn_out_of=lambda layer: layer.attn.c_proj,
        mlp_out_of=lambda layer: layer.mlp.c_proj,
        layer_prefix="transformer.h",
        attn_name="attn.c_proj",
        mlp_name="mlp.c_proj",
        layout="conv1d",
    ),
    "llama": ArchitectureSpec(
        family="causal_lm",
        layers_of=_llama_layers,
        attn_out_of=lambda layer: layer.self_attn.o_proj,
        mlp_out_of=lambda layer: layer.mlp.down_proj,
        layer_prefix="model.layers",
        attn_name="self_attn.o_proj",
        mlp_name="mlp.down_proj",
    ),
    "mistral": ArchitectureSpec(
        family="causal_lm",
        layers_of=_llama_layers,
        attn_out_of=lambda layer: layer.self_attn.o_proj,
        mlp_out_of=lambda layer: layer.mlp.down_proj,
        layer_prefix="model.layers",
        attn_name="self_attn.o_proj",
        mlp_name="mlp.down_proj",
    ),
    "qwen2": ArchitectureSpec(
        family="causal_lm",
        layers_of=_llama_layers,
        attn_out_of=lambda layer: layer.self_attn.o_proj,
        mlp_out_of=lambda layer: layer.mlp.down_proj,
        layer_prefix="model.layers",
        attn_name="self_attn.o_proj",
        mlp_name="mlp.down_proj",
    ),
    "qwen3": ArchitectureSpec(
        family="causal_lm",
        layers_of=_llama_layers,
        attn_out_of=lambda layer: layer.self_attn.o_proj,
        mlp_out_of=lambda layer: layer.mlp.down_proj,
        layer_prefix="model.layers",
        attn_name="self_attn.o_proj",
        mlp_name="mlp.down_proj",
    ),
    "gemma": ArchitectureSpec(
        family="causal_lm",
        layers_of=_llama_layers,
        attn_out_of=lambda layer: layer.self_attn.o_proj,
        mlp_out_of=lambda layer: layer.mlp.down_proj,
        layer_prefix="model.layers",
        attn_name="self_attn.o_proj",
        mlp_name="mlp.down_proj",
    ),
    "gemma2": ArchitectureSpec(
        family="causal_lm",
        layers_of=_llama_layers,
        attn_out_of=lambda layer: layer.self_attn.o_proj,
        mlp_out_of=lambda layer: layer.mlp.down_proj,
        layer_prefix="model.layers",
        attn_name="self_attn.o_proj",
        mlp_name="mlp.down_proj",
    ),
    "phi": ArchitectureSpec(
        family="causal_lm",
        layers_of=_llama_layers,
        attn_out_of=lambda layer: layer.self_attn.dense,
        mlp_out_of=lambda layer: layer.mlp.fc2
        if hasattr(layer.mlp, "fc2")
        else layer.mlp.down_proj,
        layer_prefix="model.layers",
        attn_name="self_attn.dense",
        mlp_name="mlp.fc2",
    ),
    "phi3": ArchitectureSpec(
        family="causal_lm",
        layers_of=_llama_layers,
        attn_out_of=lambda layer: layer.self_attn.o_proj,
        mlp_out_of=lambda layer: layer.mlp.down_proj,
        layer_prefix="model.layers",
        attn_name="self_attn.o_proj",
        mlp_name="mlp.down_proj",
    ),
    "gpt_neox": ArchitectureSpec(
        family="causal_lm",
        layers_of=lambda m: m.gpt_neox.layers,
        attn_out_of=lambda layer: layer.attention.dense,
        mlp_out_of=lambda layer: layer.mlp.dense_4h_to_h,
        layer_prefix="gpt_neox.layers",
        attn_name="attention.dense",
        mlp_name="mlp.dense_4h_to_h",
    ),
    "opt": ArchitectureSpec(
        family="causal_lm",
        layers_of=lambda m: m.model.decoder.layers,
        attn_out_of=lambda layer: layer.self_attn.out_proj,
        mlp_out_of=lambda layer: layer.fc2,
        layer_prefix="model.decoder.layers",
        attn_name="self_attn.out_proj",
        mlp_name="fc2",
    ),
    "bert": ArchitectureSpec(
        family="text_encoder",
        layers_of=_bert_layers,
        attn_out_of=lambda layer: layer.attention.output.dense,
        mlp_out_of=lambda layer: layer.output.dense,
        layer_prefix="encoder.layer",
        attn_name="attention.output.dense",
        mlp_name="output.dense",
    ),
    "roberta": ArchitectureSpec(
        family="text_encoder",
        layers_of=_roberta_layers,
        attn_out_of=lambda layer: layer.attention.output.dense,
        mlp_out_of=lambda layer: layer.output.dense,
        layer_prefix="encoder.layer",
        attn_name="attention.output.dense",
        mlp_name="output.dense",
    ),
    "distilbert": ArchitectureSpec(
        family="text_encoder",
        layers_of=lambda m: m.transformer.layer,
        attn_out_of=lambda layer: layer.attention.out_lin,
        mlp_out_of=lambda layer: layer.ffn.lin2,
        layer_prefix="transformer.layer",
        attn_name="attention.out_lin",
        mlp_name="ffn.lin2",
    ),
    "mpnet": ArchitectureSpec(
        family="text_encoder",
        layers_of=lambda m: m.encoder.layer,
        attn_out_of=lambda layer: layer.attention.o,
        mlp_out_of=lambda layer: layer.output.dense,
        layer_prefix="encoder.layer",
        attn_name="attention.o",
        mlp_name="output.dense",
    ),
}

# Aliases: HF model_types that share a layout with a canonical entry.
_ALIASES: dict[str, str] = {
    "gpt_neo": "gpt2",
    "gptj": "gpt2",
    "mpt": "llama",
    "stablelm": "llama",
    "cohere": "llama",
    "command-r": "llama",
    "dbrx": "llama",
    "deepseek_v3": "llama",
    "deepseek_v2": "llama",
    "mistral3": "mistral",
    "qwen2_moe": "qwen2",
    "phi-msft": "phi",
    "gemma3_text": "gemma2",
    "modernbert": "bert",
    "deberta": "bert",
    "deberta-v2": "bert",
    "albert": "bert",
    "electra": "bert",
    "xlm-roberta": "roberta",
    "chinese_clip": "clip",
    "align": "clip",
    "altclip": "clip",
    "clip_vision_model": "clip",
}


def normalize_model_type(model_type: str) -> str:
    key = (model_type or "").lower().replace("-", "_")
    if key in _SPECS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    # Soft match: qwen2_5 → qwen2, llama3 → llama, etc.
    for canonical in _SPECS:
        if key.startswith(canonical):
            return canonical
    for alias, canonical in _ALIASES.items():
        if key.startswith(alias.replace("-", "_")):
            return canonical
    return key


def get_architecture_spec(model_type: str) -> ArchitectureSpec | None:
    canonical = normalize_model_type(model_type)
    return _SPECS.get(canonical)


def supported_families() -> dict[str, dict]:
    """Public description of architecture families businesses can target."""
    types_by_family: dict[str, list[str]] = {
        "clip": [],
        "causal_lm": [],
        "text_encoder": [],
    }
    for name, spec in _SPECS.items():
        types_by_family[spec.family].append(name)
    return {
        "clip": {
            "description": "CLIP dual encoders (text tower surgery)",
            "model_types": sorted(types_by_family["clip"]),
            "examples": [
                "openai/clip-vit-base-patch32",
                "patrickjohncyh/fashion-clip",
                "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            ],
        },
        "causal_lm": {
            "description": "Decoder LLMs (Llama, Qwen, Mistral, GPT-2, Phi, Gemma, …)",
            "model_types": sorted(types_by_family["causal_lm"]),
            "examples": [
                "gpt2",
                "meta-llama/Llama-3.2-1B",
                "Qwen/Qwen2.5-0.5B",
                "microsoft/phi-2",
            ],
        },
        "text_encoder": {
            "description": "Bidirectional encoders (BERT, RoBERTa, DistilBERT, MPNet)",
            "model_types": sorted(types_by_family["text_encoder"]),
            "examples": [
                "bert-base-uncased",
                "roberta-base",
                "sentence-transformers/all-mpnet-base-v2",
            ],
        },
    }


def build_layer_views(model: nn.Module, spec: ArchitectureSpec) -> list[LayerView]:
    layers = list(spec.layers_of(model))
    views: list[LayerView] = []
    for i, layer in enumerate(layers):
        views.append(
            LayerView(
                attn_out=spec.attn_out_of(layer),
                mlp_out=spec.mlp_out_of(layer),
                attn_path=f"{spec.layer_prefix}.{i}.{spec.attn_name}",
                mlp_path=f"{spec.layer_prefix}.{i}.{spec.mlp_name}",
                layout=spec.layout,
            )
        )
    return views
