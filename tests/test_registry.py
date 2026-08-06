"""Architecture detection + multi-family loading (offline where possible)."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from scalpel.editing.surgeon import SurgeryConfig
from scalpel.models.adapters import get_architecture_spec, normalize_model_type, supported_families
from scalpel.models.registry import (
    LoadedModel,
    UnsupportedArchitectureError,
    load_model,
    probe_model,
)
from scalpel.pipelines.debias import run_debias_pipeline


def test_normalize_model_type_aliases():
    assert normalize_model_type("llama") == "llama"
    assert normalize_model_type("qwen2") == "qwen2"
    assert normalize_model_type("Qwen2_5") == "qwen2" or normalize_model_type("qwen2_5") == "qwen2"
    assert normalize_model_type("xlm-roberta") == "roberta"
    assert get_architecture_spec("mistral").family == "causal_lm"
    assert get_architecture_spec("clip").family == "clip"
    assert get_architecture_spec("bert").family == "text_encoder"
    assert get_architecture_spec("totally_made_up_xyz") is None


def test_supported_families_cover_business_stacks():
    families = supported_families()
    assert set(families) == {"clip", "causal_lm", "text_encoder"}
    assert "gpt2" in families["causal_lm"]["examples"]
    assert any("fashion-clip" in e for e in families["clip"]["examples"])


def test_injected_clip_exposes_layer_views(tiny_clip: LoadedModel):
    assert tiny_clip.family == "clip"
    assert tiny_clip.num_layers == 4
    assert len(tiny_clip.layers) == 4
    assert tiny_clip.layers[0].attn_out is not None
    emb = tiny_clip.encode_text(["a photo of a cat", "a photo of a dog"])
    assert emb.shape[0] == 2
    assert torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-4)


class _FakeCausalTokenizer:
    """Deterministic stand-in so causal-LM tests stay offline and cache-independent."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 64

    def __call__(
        self, texts, padding=True, truncation=True, return_tensors="pt", max_length=64
    ):
        rows = []
        for text in texts:
            ids = [2 + (ord(ch) % 50) for ch in text[: max_length - 1]] or [2]
            ids = ids + [self.eos_token_id]
            rows.append(ids)
        width = max(len(r) for r in rows)
        input_ids = torch.tensor(
            [r + [self.pad_token_id] * (width - len(r)) for r in rows], dtype=torch.long
        )
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, path):  # pragma: no cover - unused in unit test
        pass


@pytest.fixture(scope="module")
def tiny_gpt2() -> LoadedModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_embd=64,
        n_layer=3,
        n_head=4,
        n_positions=64,
        vocab_size=64,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return load_model("tiny-gpt2-test", model=model, tokenizer=_FakeCausalTokenizer())


def test_gpt2_pipeline_audit_and_edit(tiny_gpt2, tmp_path):
    result = run_debias_pipeline(
        model_id="tiny-gpt2-test",
        bias="gender_profession",
        config=SurgeryConfig(max_components=4, harden_projection=False, device="cpu"),
        save_dir=tmp_path,
        audit_only=False,
        model=copy.deepcopy(tiny_gpt2.model),
        tokenizer=tiny_gpt2.tokenizer,
    )
    assert result.report["model_family"] == "causal_lm"
    assert result.report["surgery"]["num_edits"] >= 1
    assert "bias_reduction" in result.report["metrics"]


def test_probe_rejects_unknown_architecture(monkeypatch):
    from transformers import AutoConfig

    class _Cfg:
        model_type = "totally_unsupported_arch_xyz"
        architectures: tuple = ()

    monkeypatch.setattr(
        AutoConfig, "from_pretrained", classmethod(lambda cls, *a, **k: _Cfg())
    )
    with pytest.raises(UnsupportedArchitectureError, match="cannot edit yet"):
        probe_model("org/custom-arch")
