"""The surgeon: applies rank-one projection edits to an isolated bias circuit.

For every selected circuit component the corresponding weight matrix is edited
in closed form so it can no longer write onto the bias direction:

- attention head ``(layer, head)``: project the head's slice of the attention
  output projection,
- MLP block ``layer``: project the MLP down-projection (+ bias),
- optionally (CLIP), harden ``text_projection`` so residual bias cannot leak
  into the shared embedding space.
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
    """Tunable knobs for one surgery."""

    max_components: int = 12
    cumulative_share: float = 0.8
    num_directions: int = 1
    calibrate: bool = False
    harden_projection: bool = True
    edit_bias_terms: bool = True
    direction_layer: int | None = None
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


def _project_attn_head_(
    weight: torch.Tensor,
    v: torch.Tensor,
    cols: slice,
    layout: str,
    strength: float,
) -> None:
    if layout == "conv1d":
        # y = x @ W; head owns input rows; erase v from output space of that block.
        basis_rows = weight[cols, :]
        # Reuse input-space style math on the [dh, d_out] block:
        # block' = block @ (I - V^T V)
        from scalpel.editing.rank_one import _as_orthonormal_basis

        basis = _as_orthonormal_basis(v, weight.dtype)
        weight[cols, :] = basis_rows - strength * ((basis_rows @ basis.T) @ basis)
    else:
        project_out_of_columns_(weight, v, cols, strength)


def _project_mlp_out_(
    weight: torch.Tensor, v: torch.Tensor, layout: str, strength: float
) -> None:
    if layout == "conv1d":
        # y = x @ W → erase output space via W' = W (I - V^T V)
        project_out_of_input_(weight, v, strength)
    else:
        project_out_of_output_(weight, v, strength)


@torch.no_grad()
def perform_surgery(
    lm: LoadedModel, circuit: BiasCircuit, config: SurgeryConfig, strength: float = 1.0
) -> SurgeryRecord:
    """Apply closed-form low-rank projection edits severing every selected component."""
    v = circuit.direction.basis  # [k, d]
    rank = circuit.direction.num_directions
    record = SurgeryRecord()
    record.strength = strength
    dh = lm.head_dim
    edited_attn_layers: set[int] = set()

    for component in circuit.selected:
        view = lm.layers[component.layer]
        if component.kind == "attn_head":
            cols = slice(component.head * dh, (component.head + 1) * dh)
            _project_attn_head_(view.attn_weight, v, cols, view.layout, strength)
            record.log(
                f"{view.attn_path}.weight[:, {cols.start}:{cols.stop}]"
                if view.layout == "linear"
                else f"{view.attn_path}.weight[{cols.start}:{cols.stop}, :]",
                "output-space projection (per-head)",
                rank=rank,
            )
            if config.edit_bias_terms and component.layer not in edited_attn_layers:
                if view.attn_bias is not None:
                    project_vector_(view.attn_bias, v, strength)
                    record.log(f"{view.attn_path}.bias", "bias projection", rank=rank)
                edited_attn_layers.add(component.layer)
        elif component.kind == "mlp":
            _project_mlp_out_(view.mlp_weight, v, view.layout, strength)
            record.log(
                f"{view.mlp_path}.weight",
                "output-space projection",
                rank=rank,
            )
            if config.edit_bias_terms and view.mlp_bias is not None:
                project_vector_(view.mlp_bias, v, strength)
                record.log(f"{view.mlp_path}.bias", "bias projection", rank=rank)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown component kind: {component.kind}")

    if config.harden_projection and lm.has_readout_hardening():
        gamma = lm.model.text_model.final_layer_norm.weight
        v_post = gamma * (v - v.mean(dim=-1, keepdim=True))
        project_out_of_input_(lm.model.text_projection.weight, v_post, strength)
        record.log("text_projection.weight", "input-space projection", rank=rank)

    return record
