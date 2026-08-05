"""Bias direction estimation in the residual stream.

We use the difference-of-means ("diff-in-means") estimator over minimally
contrastive prompt pairs, which is the standard closed-form probe for linear
concept directions (Marks & Tegmark 2023; Arditi et al. 2024). For each layer
we compute the unit vector separating group-A from group-B activations at the
pooled token, then select the layer where that direction is most linearly
separable (largest standardized mean difference).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scalpel.interpretability.activations import ComponentActivations


@dataclass
class BiasDirection:
    """A unit-norm bias direction in residual stream space."""

    vector: torch.Tensor  # [d], unit norm
    layer: int  # residual stream index it was estimated at (1..L)
    separation: float  # standardized mean difference along the direction
    per_layer_separation: list[float]


def find_bias_direction(
    acts_a: ComponentActivations,
    acts_b: ComponentActivations,
    layer: int | None = None,
) -> BiasDirection:
    """Estimate the bias direction from paired group-A / group-B activations.

    ``acts_a`` and ``acts_b`` must come from paired prompt lists of equal
    length (pair i differs only in the group attribute).
    """
    if acts_a.resid.shape != acts_b.resid.shape:
        raise ValueError("Group A and B activations must be paired (same shape)")

    num_streams = acts_a.resid.shape[0]  # L + 1
    candidates: list[tuple[torch.Tensor, float]] = []
    for stream in range(num_streams):
        h_a, h_b = acts_a.resid[stream], acts_b.resid[stream]  # [N, d]
        delta = h_a.mean(dim=0) - h_b.mean(dim=0)
        norm = delta.norm()
        if norm < 1e-8:
            candidates.append((torch.zeros_like(delta), 0.0))
            continue
        v = delta / norm
        proj_a, proj_b = h_a @ v, h_b @ v
        pooled_std = torch.cat([proj_a - proj_a.mean(), proj_b - proj_b.mean()]).std()
        separation = ((proj_a.mean() - proj_b.mean()) / (pooled_std + 1e-8)).item()
        candidates.append((v, separation))

    separations = [sep for _, sep in candidates]
    if layer is None:
        # Skip the raw embedding stream (index 0): edits act on layer writes.
        layer = max(range(1, num_streams), key=lambda i: separations[i])
    vector, separation = candidates[layer]
    if vector.norm() < 1e-6:
        raise ValueError("Degenerate bias direction: groups are not separable")
    return BiasDirection(
        vector=vector,
        layer=layer,
        separation=separation,
        per_layer_separation=separations,
    )
