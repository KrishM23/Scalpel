"""Bias subspace estimation in the residual stream.

The first basis vector is the difference-of-means ("diff-in-means") direction
over minimally contrastive prompt pairs — the standard closed-form probe for
linear concept directions (Marks & Tegmark 2023; Arditi et al. 2024).

A single direction captures the *dominant* linear bias component, but real
bias concepts are often spread over a small subspace. When ``num_directions``
> 1 we augment the mean-difference direction with the top principal components
of the residual paired differences (after removing the mean-difference
component), yielding an orthonormal basis for the bias subspace. Erasing the
whole subspace attacks higher-order association structure that survives
single-direction erasure.

For each layer we compute the candidate direction and select the layer where
group separation is largest (standardized mean difference).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scalpel.interpretability.activations import ComponentActivations


@dataclass
class BiasDirection:
    """An orthonormal bias basis in residual stream space.

    ``basis`` has shape [k, d]; row 0 is the diff-in-means direction, rows
    1..k-1 are principal components of the residual paired differences.
    """

    basis: torch.Tensor  # [k, d], orthonormal rows
    layer: int  # residual stream index it was estimated at (1..L)
    separation: float  # standardized mean difference along basis[0]
    per_layer_separation: list[float]

    @property
    def vector(self) -> torch.Tensor:
        """The dominant (diff-in-means) direction."""
        return self.basis[0]

    @property
    def num_directions(self) -> int:
        return self.basis.shape[0]


def find_bias_direction(
    acts_a: ComponentActivations,
    acts_b: ComponentActivations,
    layer: int | None = None,
    num_directions: int = 1,
) -> BiasDirection:
    """Estimate the bias subspace from paired group-A / group-B activations.

    ``acts_a`` and ``acts_b`` must come from paired prompt lists of equal
    length (pair i differs only in the group attribute).
    """
    if acts_a.resid.shape != acts_b.resid.shape:
        raise ValueError("Group A and B activations must be paired (same shape)")
    if num_directions < 1:
        raise ValueError("num_directions must be >= 1")

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
    v1, separation = candidates[layer]
    if v1.norm() < 1e-6:
        raise ValueError("Degenerate bias direction: groups are not separable")

    basis_rows = [v1]
    if num_directions > 1:
        # Principal components of paired differences, orthogonal to v1.
        diffs = acts_a.resid[layer] - acts_b.resid[layer]  # [N, d]
        residual = diffs - torch.outer(diffs @ v1, v1)
        residual = residual - residual.mean(dim=0)
        max_extra = min(num_directions - 1, min(residual.shape) - 1)
        if max_extra > 0:
            _, svals, vh = torch.linalg.svd(residual, full_matrices=False)
            for i in range(max_extra):
                if svals[i] < 1e-6:
                    break
                basis_rows.append(vh[i])

    basis = torch.stack(basis_rows, dim=0)
    # Numerical safety: re-orthonormalize the basis.
    q, _ = torch.linalg.qr(basis.T)
    basis = q.T[: len(basis_rows)]

    return BiasDirection(
        basis=basis,
        layer=layer,
        separation=separation,
        per_layer_separation=separations,
    )
