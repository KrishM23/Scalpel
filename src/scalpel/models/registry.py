"""Model registry: probe any Hugging Face id, load via architecture adapters.

Businesses pass any HF model id (CLIP, causal LM, or text encoder). Scalpel
reads ``config.json`` via ``AutoConfig``, classifies the architecture, and
loads through a family-specific path. Featured models are suggestions only —
not a whitelist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPModel,
    CLIPTokenizerFast,
    PreTrainedTokenizerBase,
)

from scalpel.models.adapters import (
    ArchitectureSpec,
    Family,
    LayerView,
    build_layer_views,
    get_architecture_spec,
    normalize_model_type,
    supported_families,
)

# Curated suggestions shown in the console /docs — any other compatible HF id works.
_FEATURED: dict[str, dict] = {
    "openai/clip-vit-base-patch32": {
        "family": "clip",
        "description": "CLIP ViT-B/32 dual encoder (OpenAI) — fast default",
    },
    "openai/clip-vit-base-patch16": {
        "family": "clip",
        "description": "CLIP ViT-B/16 dual encoder (OpenAI)",
    },
    "openai/clip-vit-large-patch14": {
        "family": "clip",
        "description": "CLIP ViT-L/14 dual encoder (OpenAI)",
    },
    "patrickjohncyh/fashion-clip": {
        "family": "clip",
        "description": "FashionCLIP — CLIP fine-tuned on fashion products",
    },
    "gpt2": {
        "family": "causal_lm",
        "description": "GPT-2 small — lightweight causal LM for LLM surgery demos",
    },
    "bert-base-uncased": {
        "family": "text_encoder",
        "description": "BERT base — classic bidirectional text encoder",
    },
    "Qwen/Qwen2.5-0.5B": {
        "family": "causal_lm",
        "description": "Qwen2.5 0.5B — small modern instruction-capable LLM",
    },
}


class UnsupportedArchitectureError(ValueError):
    """Raised when a HF config's model_type has no surgery adapter yet."""


@dataclass(frozen=True)
class ModelProbe:
    """Result of inspecting a HF model id without downloading weights."""

    model_id: str
    family: Family
    model_type: str
    architecture_key: str
    description: str


def featured_models() -> dict[str, dict]:
    return dict(_FEATURED)


def supported_models() -> dict[str, dict]:
    """Backward-compatible alias: featured catalog (not an allow-list)."""
    return featured_models()


@dataclass
class LoadedModel:
    """Architecture-agnostic text tower handle used by the surgery pipeline."""

    model_id: str
    family: Family
    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    layers: list[LayerView]
    model_type: str
    architecture_key: str

    # --- dimensions ---------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_heads(self) -> int:
        cfg = self._text_config()
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        raise AttributeError("could not determine num_attention_heads")

    @property
    def d_model(self) -> int:
        cfg = self._text_config()
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        raise AttributeError("could not determine hidden_size")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def text_layers(self):
        """Deprecated alias kept for older call sites; prefer ``layers``."""
        return self.layers

    def _text_config(self):
        cfg = self.model.config
        if self.family == "clip" and hasattr(cfg, "text_config"):
            return cfg.text_config
        return cfg

    # --- tokenization / pooling ---------------------------------------------

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        max_length = getattr(self.tokenizer, "model_max_length", 512) or 512
        # Some HF tokenizers ship model_max_length=int(1e30); clamp for safety.
        if max_length > 4096:
            max_length = 512
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def pool_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token index used as the residual-stream readout per sequence."""
        if self.family == "clip":
            return self._clip_eot_positions(input_ids)
        if self.family == "text_encoder":
            # CLS / first token.
            return torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        # Causal LM: last non-pad token.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return torch.full(
                (input_ids.shape[0],),
                input_ids.shape[1] - 1,
                dtype=torch.long,
                device=input_ids.device,
            )
        mask = input_ids != pad_id
        # length-1 index of last real token
        lengths = mask.long().sum(dim=-1).clamp(min=1) - 1
        return lengths

    def eot_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Alias used by older CLIP-centric code."""
        return self.pool_positions(input_ids)

    def _clip_eot_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.model.config.text_config.eos_token_id
        return (input_ids == eos_id).int().argmax(dim=-1)

    # --- forward / encode ---------------------------------------------------

    def forward_hidden_states(self, batch: dict[str, torch.Tensor]):
        """Run the text stack; return an object with ``.hidden_states``."""
        if self.family == "clip":
            return self.model.text_model(**batch, output_hidden_states=True)
        kwargs: dict = {"output_hidden_states": True}
        if self.family == "causal_lm":
            kwargs["use_cache"] = False
        return self.model(**batch, **kwargs)

    @torch.no_grad()
    def encode_text(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """L2-normalized text embeddings used for WEAT / retention."""
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenize(texts[start : start + batch_size])
            if self.family == "clip":
                features = self.model.get_text_features(**batch)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
            else:
                outputs = self.forward_hidden_states(batch)
                hidden = outputs.hidden_states[-1]
                pos = self.pool_positions(batch["input_ids"])
                rows = torch.arange(pos.shape[0], device=self.device)
                features = hidden[rows, pos]
            chunks.append(features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        return torch.cat(chunks, dim=0)

    # --- surgery helpers ----------------------------------------------------

    def has_readout_hardening(self) -> bool:
        return (
            self.family == "clip"
            and hasattr(self.model, "text_projection")
            and hasattr(self.model, "text_model")
        )

    def snapshot_editable_state(self) -> dict[str, torch.Tensor]:
        """Clone parameters the calibration sweep may rewrite."""
        if self.family == "clip":
            prefixes = ("text_model.", "text_projection.")
            return {
                name: tensor.detach().clone()
                for name, tensor in self.model.state_dict().items()
                if name.startswith(prefixes)
            }
        return {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}

    def restore_editable_state(self, snapshot: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(snapshot, strict=False)

    def save_pretrained(self, path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def probe_model(model_id: str, trust_remote_code: bool = True) -> ModelProbe:
    """Classify a HF model id from its config alone (no weight download)."""
    model_id = model_id.strip()
    if not model_id:
        raise UnsupportedArchitectureError("model_id must be non-empty")

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        raise UnsupportedArchitectureError(
            f"Could not load Hugging Face config for '{model_id}': {exc}"
        ) from exc

    model_type = getattr(config, "model_type", None) or ""
    # CLIP dual-encoder configs sometimes nest text_config.
    if not model_type and hasattr(config, "text_config"):
        model_type = "clip"
    # Architecture list hint (e.g. ["CLIPModel"]).
    architectures = getattr(config, "architectures", None) or []
    if not model_type and any("CLIP" in a for a in architectures):
        model_type = "clip"

    arch_key = normalize_model_type(model_type)
    spec = get_architecture_spec(model_type)
    if spec is None:
        families = supported_families()
        supported_types = sorted(
            {t for meta in families.values() for t in meta["model_types"]}
        )
        raise UnsupportedArchitectureError(
            f"Model '{model_id}' has model_type='{model_type}', which Scalpel "
            f"cannot edit yet. Supported families: clip, causal_lm, text_encoder "
            f"(types include {', '.join(supported_types[:12])}, …). "
            f"Open an issue to add this architecture."
        )

    featured = _FEATURED.get(model_id, {})
    description = featured.get(
        "description",
        f"{spec.family} model ({arch_key}) — surgery on text residual stream",
    )
    return ModelProbe(
        model_id=model_id,
        family=spec.family,
        model_type=model_type or arch_key,
        architecture_key=arch_key,
        description=description,
    )


def _load_weights(probe: ModelProbe, trust_remote_code: bool) -> nn.Module:
    if probe.family == "clip":
        return CLIPModel.from_pretrained(
            probe.model_id, trust_remote_code=trust_remote_code
        )
    if probe.family == "causal_lm":
        return AutoModelForCausalLM.from_pretrained(
            probe.model_id, trust_remote_code=trust_remote_code
        )
    # text_encoder — prefer base AutoModel (sentence-transformers repos often
    # wrap a bert/mpnet encoder).
    return AutoModel.from_pretrained(probe.model_id, trust_remote_code=trust_remote_code)


def _load_tokenizer(probe: ModelProbe, trust_remote_code: bool) -> PreTrainedTokenizerBase:
    if probe.family == "clip":
        tok = CLIPTokenizerFast.from_pretrained(
            probe.model_id, trust_remote_code=trust_remote_code
        )
    else:
        tok = AutoTokenizer.from_pretrained(
            probe.model_id, trust_remote_code=trust_remote_code, use_fast=True
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def _wrap(
    model_id: str,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    probe: ModelProbe | None = None,
) -> LoadedModel:
    if probe is None:
        # Injected models (tests): infer from the module class / config.
        cfg = model.config
        model_type = getattr(cfg, "model_type", None) or "clip"
        if isinstance(model, CLIPModel):
            model_type = "clip"
        probe = ModelProbe(
            model_id=model_id,
            family=get_architecture_spec(model_type).family,  # type: ignore[union-attr]
            model_type=model_type,
            architecture_key=normalize_model_type(model_type),
            description="injected",
        )
    spec: ArchitectureSpec = get_architecture_spec(probe.model_type)  # type: ignore[assignment]
    if spec is None:
        raise UnsupportedArchitectureError(probe.model_type)
    # CLIP weights live on the dual-encoder root; layer walk uses full model.
    root = model
    layers = build_layer_views(root, spec)
    return LoadedModel(
        model_id=model_id,
        family=probe.family,
        model=model,
        tokenizer=tokenizer,
        device=device,
        layers=layers,
        model_type=probe.model_type,
        architecture_key=probe.architecture_key,
    )


def load_model(
    model_id: str,
    device: str = "cpu",
    model: nn.Module | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    trust_remote_code: bool = True,
) -> LoadedModel:
    """Load any supported HF model id (or wrap injected instances for tests)."""
    torch_device = torch.device(device)
    if model is not None and tokenizer is not None:
        lm = _wrap(model_id, model, tokenizer, torch_device, probe=None)
        lm.model.to(torch_device)
        lm.model.eval()
        return lm

    probe = probe_model(model_id, trust_remote_code=trust_remote_code)
    model = _load_weights(probe, trust_remote_code=trust_remote_code)
    tokenizer = _load_tokenizer(probe, trust_remote_code=trust_remote_code)
    model.to(torch_device)
    model.eval()
    return _wrap(model_id, model, tokenizer, torch_device, probe=probe)
