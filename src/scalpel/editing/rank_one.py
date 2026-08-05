"""Closed-form rank-one weight edits.

All edits are orthogonal-projection updates of the form

    W' = (I - v v^T) W  =  W - v (v^T W)

for a unit vector ``v``. This is a rank-one modification (the subtracted term
``v (v^T W)`` has rank one), it is closed-form (no gradients, no fine-tuning),
and it is *permanent*: after the edit, the component can no longer write any
signal along ``v`` into the residual stream, for every possible input. Because
the projection only removes a single dimension out of ``d_model``, everything
the component computes orthogonally to ``v`` is untouched — this is what makes
the edit surgical rather than a retrain.

The same construction underlies concept-erasure methods (LEACE, Belrose et al.
2023) and directional ablation (Arditi et al. 2024); ROME/MEMIT use the
analogous closed-form rank-one structure for additive edits.
"""

from __future__ import annotations

import torch


def _check_unit(v: torch.Tensor) -> torch.Tensor:
    if v.dim() != 1:
        raise ValueError("direction must be a 1-D vector")
    norm = v.norm()
    if not torch.isfinite(norm) or norm < 1e-8:
        raise ValueError("direction must have non-zero finite norm")
    return v / norm


@torch.no_grad()
def project_out_of_output_(weight: torch.Tensor, v: torch.Tensor) -> None:
    """In-place: remove direction ``v`` from the *output* (row) space of a
    linear map ``y = W x``. ``weight`` has shape [d_out, d_in]; ``v`` lives in
    the d_out space. Afterwards ``v . (W' x) == 0`` for all x."""
    v = _check_unit(v.to(weight.dtype))
    weight.sub_(torch.outer(v, v @ weight))


@torch.no_grad()
def project_out_of_columns_(weight: torch.Tensor, v: torch.Tensor, cols: slice) -> None:
    """In-place: remove ``v`` from the output space of a column-slice of ``W``.

    Used for per-head attention edits: head ``h`` only touches columns
    ``[h*dh, (h+1)*dh)`` of ``out_proj.weight``, so projecting that slice
    silences head ``h`` along ``v`` while leaving every other head untouched.
    """
    v = _check_unit(v.to(weight.dtype))
    block = weight[:, cols]
    weight[:, cols] = block - torch.outer(v, v @ block)


@torch.no_grad()
def project_vector_(bias: torch.Tensor, v: torch.Tensor) -> None:
    """In-place: remove the component of a bias/vector along ``v``."""
    v = _check_unit(v.to(bias.dtype))
    bias.sub_(v * (v @ bias))


@torch.no_grad()
def project_out_of_input_(weight: torch.Tensor, v: torch.Tensor) -> None:
    """In-place: make a linear map ``y = W x`` blind to direction ``v`` in its
    *input* (column) space: W' = W (I - v v^T). Used to harden read-out maps
    (e.g. CLIP's text_projection) so any residual trace of ``v`` reaching them
    cannot leak into the shared embedding space."""
    v = _check_unit(v.to(weight.dtype))
    weight.sub_(torch.outer(weight @ v, v))
