"""Bias circuit isolation.

Given a bias direction ``v`` in the residual stream, we attribute how strongly
each component (attention head or MLP block) writes onto ``v`` when the group
attribute flips. For paired prompts (a_i, b_i) and component write ``w_c``:

    score(c) = | mean_i [ (w_c(a_i) - w_c(b_i)) . v ] |

Because prompt pairs are minimally contrastive, everything unrelated to the
bias attribute cancels in the difference; components with large scores are
precisely the ones transporting the bias signal into the pooled representation.
The ranked set of components above a cumulative-mass threshold is the isolated
"bias circuit" that the editor then severs.
"""

from __future__ import annotations

from dataclasses import dataclass

from scalpel.interpretability.activations import ComponentActivations
from scalpel.interpretability.directions import BiasDirection
from scalpel.models.registry import LoadedModel


@dataclass
class CircuitComponent:
    kind: str  # "attn_head" | "mlp"
    layer: int  # encoder layer index (0-based)
    head: int | None  # head index for attn_head, None for mlp
    score: float  # attributed write onto the bias direction
    share: float  # fraction of total attributed effect

    @property
    def label(self) -> str:
        if self.kind == "attn_head":
            return f"L{self.layer}.H{self.head}"
        return f"L{self.layer}.MLP"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "layer": self.layer,
            "head": self.head,
            "label": self.label,
            "score": round(self.score, 6),
            "share": round(self.share, 6),
        }


@dataclass
class BiasCircuit:
    direction: BiasDirection
    components: list[CircuitComponent]  # ranked, descending score
    selected: list[CircuitComponent]  # the subset chosen for surgery

    def to_dict(self) -> dict:
        return {
            "direction_layer": self.direction.layer,
            "num_directions": self.direction.num_directions,
            "direction_separation": round(self.direction.separation, 4),
            "per_layer_separation": [round(s, 4) for s in self.direction.per_layer_separation],
            "selected_components": [c.to_dict() for c in self.selected],
            "all_components_top20": [c.to_dict() for c in self.components[:20]],
        }


def isolate_bias_circuit(
    lm: LoadedModel,
    acts_a: ComponentActivations,
    acts_b: ComponentActivations,
    direction: BiasDirection,
    max_components: int = 12,
    cumulative_share: float = 0.8,
) -> BiasCircuit:
    """Rank all components by their write onto the bias subspace and select
    the smallest prefix covering ``cumulative_share`` of the total attributed
    effect (capped at ``max_components``).

    With a multi-directional basis V [k, d], a component's effect is the L2
    norm of its mean paired-difference write projected into the subspace,
    which reduces to |Δwrite · v| for k = 1.
    """
    basis = direction.basis  # [k, d]
    raw: list[tuple[str, int, int | None, float]] = []

    for layer in range(lm.num_layers):
        writes_a = acts_a.head_writes(lm, layer)  # [H, N, d]
        writes_b = acts_b.head_writes(lm, layer)
        head_effect = ((writes_a - writes_b) @ basis.T).mean(dim=1).norm(dim=-1)  # [H]
        for head in range(lm.num_heads):
            raw.append(("attn_head", layer, head, head_effect[head].item()))

        mlp_diff = acts_a.mlp_out[layer] - acts_b.mlp_out[layer]
        mlp_effect = (mlp_diff @ basis.T).mean(dim=0).norm().item()
        raw.append(("mlp", layer, None, mlp_effect))

    total = sum(abs(score) for *_, score in raw) + 1e-12
    components = sorted(
        (
            CircuitComponent(kind=kind, layer=layer, head=head, score=abs(score),
                             share=abs(score) / total)
            for kind, layer, head, score in raw
        ),
        key=lambda c: c.score,
        reverse=True,
    )

    selected: list[CircuitComponent] = []
    covered = 0.0
    for component in components:
        if len(selected) >= max_components or covered >= cumulative_share:
            break
        selected.append(component)
        covered += component.share

    return BiasCircuit(direction=direction, components=components, selected=selected)
