import torch

from scalpel.biases.catalog import get_bias_spec
from scalpel.interpretability.activations import (
    ComponentActivations,
    record_component_activations,
)
from scalpel.interpretability.directions import find_bias_direction


def _synthetic_acts(direction: torch.Tensor, sign: float, n: int = 32) -> ComponentActivations:
    """Activations whose layer-2 residual stream shifts by +/- direction."""
    torch.manual_seed(42)
    d = direction.shape[0]
    base = torch.randn(3, n, d) * 0.1  # streams: embeddings + 2 layers
    shifted = base.clone()
    shifted[2] += sign * direction
    return ComponentActivations(
        resid=shifted,
        attn_head_inputs=torch.randn(2, n, d) * 0.1,
        mlp_out=torch.randn(2, n, d) * 0.1,
    )


def test_diff_in_means_recovers_planted_direction():
    d = 16
    v_true = torch.zeros(d)
    v_true[3] = 1.0
    acts_a = _synthetic_acts(v_true, sign=+1.0)
    acts_b = _synthetic_acts(v_true, sign=-1.0)
    direction = find_bias_direction(acts_a, acts_b)
    assert direction.layer == 2
    cosine = float(direction.vector @ v_true)
    assert abs(cosine) > 0.99
    assert direction.separation > 5.0


def test_recorded_activation_shapes(tiny_clip):
    spec = get_bias_spec("gender_profession")
    prompts = [pair[0] for pair in spec.paired_prompts[:6]]
    acts = record_component_activations(tiny_clip, prompts)
    layers, d = tiny_clip.num_layers, tiny_clip.d_model
    assert acts.resid.shape == (layers + 1, 6, d)
    assert acts.attn_head_inputs.shape == (layers, 6, d)
    assert acts.mlp_out.shape == (layers, 6, d)


def test_head_writes_sum_to_attention_output(tiny_clip):
    """Per-head write decomposition must reconstruct out_proj (minus bias)."""
    prompts = ["a photo of a man", "a photo of a woman", "a red car"]
    acts = record_component_activations(tiny_clip, prompts)
    layer = 1
    writes = acts.head_writes(tiny_clip, layer)  # [H, N, d]
    out_proj = tiny_clip.text_layers[layer].self_attn.out_proj
    expected = acts.attn_head_inputs[layer] @ out_proj.weight.T
    assert torch.allclose(writes.sum(dim=0), expected, atol=1e-4)
