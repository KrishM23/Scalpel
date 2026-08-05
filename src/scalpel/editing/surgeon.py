"""The surgeon: applies rank-one projection edits to an isolated bias circuit.

For every selected circuit component the corresponding weight matrix is edited
in closed form so it can no longer write onto the bias direction:

- attention head ``(layer, head)``: project the head's column-slice of
  ``self_attn.out_proj.weight`` (a rank-one update per head),
- MLP block ``layer``: project ``mlp.fc2.weight`` and its bias,
- optionally, harden ``text_projection`` so any residual bias signal produced
  by unedited components cannot be read into the shared embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from scalpel.editing.rank_one import (
    project_out_of_columns_,
    project_out_of_input_,
    project_out_of_output_,
    project_vector_,
)
from scalpel.interpretability.circuits import BiasCircuit
from scalpel.models.registry import LoadedModel


@dataclass
class SurgeryConfig:
    """Tunable knobs for one surgery.

    max_components / cumulative_share control how much of the attributed bias
    circuit is severed; harden_projection additionally projects the bias
    direction out of the text_projection read-out map.
    """

    max_components: int = 12
    cumulative_share: float = 0.8
    num_directions: int = 1  # dimensionality of the erased bias subspace
    calibrate: bool = False  # sweep erasure strength to minimize residual |WEAT|
    harden_projection: bool = True
    edit_bias_terms: bool = True
    direction_layer: int | None = None  # None = auto-select most separable layer
    device: str = "cpu"

    def to_dict(self) -> dict:
        return {
            "max_components": self.max_components,
            "cumulative_share": self.cumulative_share,
            "num_directions": self.num_directions,
            "calibrate": self.calibrate,
            "harden_projection": self.harden_projection,
            "edit_bias_terms": self.edit_bias_terms,
            "direction_layer": self.direction_layer,
            "device": self.device,
        }


@dataclass
class SurgeryRecord:
    """Audit trail of every parameter tensor touched."""

    edits: list[dict] = field(default_factory=list)
    strength: float = 1.0

    def log(self, target: str, kind: str, rank: int = 1) -> None:
        self.edits.append({"target": target, "edit": kind, "rank": rank})

    def to_dict(self) -> dict:
        return {
            "num_edits": len(self.edits),
            "strength": round(self.strength, 4),
            "edits": self.edits,
        }


@torch.no_grad()
def perform_surgery(
    lm: LoadedModel, circuit: BiasCircuit, config: SurgeryConfig, strength: float = 1.0
) -> SurgeryRecord:
    """Apply closed-form low-rank projection edits severing every selected
    component (rank k per matrix, where k = the erased subspace dimension).

    ``strength`` scales the projection: 1.0 fully erases the subspace,
    fractional values attenuate it (used by calibrated surgery)."""
    v = circuit.direction.basis  # [k, d]
    rank = circuit.direction.num_directions
    record = SurgeryRecord()
    record.strength = strength
    dh = lm.head_dim
    edited_attn_layers: set[int] = set()

    for component in circuit.selected:
        layer = lm.text_layers[component.layer]
        if component.kind == "attn_head":
            cols = slice(component.head * dh, (component.head + 1) * dh)
            project_out_of_columns_(layer.self_attn.out_proj.weight, v, cols, strength)
            record.log(
                f"text_model.encoder.layers.{component.layer}.self_attn.out_proj.weight"
                f"[:, {cols.start}:{cols.stop}]",
                "output-space projection (per-head)",
                rank=rank,
            )
            if config.edit_bias_terms and component.layer not in edited_attn_layers:
                project_vector_(layer.self_attn.out_proj.bias, v, strength)
                record.log(
                    f"text_model.encoder.layers.{component.layer}.self_attn.out_proj.bias",
                    "bias projection",
                    rank=rank,
                )
                edited_attn_layers.add(component.layer)
        elif component.kind == "mlp":
            project_out_of_output_(layer.mlp.fc2.weight, v, strength)
            record.log(
                f"text_model.encoder.layers.{component.layer}.mlp.fc2.weight",
                "output-space projection",
                rank=rank,
            )
            if config.edit_bias_terms:
                project_vector_(layer.mlp.fc2.bias, v, strength)
                record.log(
                    f"text_model.encoder.layers.{component.layer}.mlp.fc2.bias",
                    "bias projection",
                    rank=rank,
                )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown component kind: {component.kind}")

    if config.harden_projection:
        # text_projection reads the residual stream *after* final_layer_norm.
        # LN centers the stream and rescales coordinates by its gain, so a
        # perturbation alpha*v pre-LN surfaces post-LN (to first order) along
        # gamma (x) (v - mean(v)); harden against the transported subspace.
        gamma = lm.model.text_model.final_layer_norm.weight
        v_post = gamma * (v - v.mean(dim=-1, keepdim=True))
        project_out_of_input_(lm.model.text_projection.weight, v_post, strength)
        record.log("text_projection.weight", "input-space projection", rank=rank)

    return record
