"""Closed-form low-rank projection edits.

All edits are orthogonal-projection updates of the form

    W' = (I - V^T V) W  =  W - V^T (V W)

for an orthonormal basis ``V`` with k rows (k = 1 gives the classic rank-one
update ``W - v(vᵀW)``). The subtracted term has rank k, the edit is
closed-form (no gradients, no fine-tuning), and it is *permanent*: after the
edit, the component cannot write any signal inside the erased subspace, for
every possible input. Because only k dimensions out of d_model are removed,
everything the component computes orthogonally to the subspace is untouched —
this is what makes the edit surgical rather than a retrain.

The same construction underlies concept-erasure methods (LEACE, Belrose et al.
2023) and directional ablation (Arditi et al. 2024); ROME/MEMIT use the
analogous closed-form rank-one structure for additive edits.
"""

from __future__ import annotations

import torch


def _as_orthonormal_basis(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Normalize input to an orthonormal [k, d] basis (accepts a bare [d])."""
    if v.dim() == 1:
        v = v.unsqueeze(0)
    if v.dim() != 2:
        raise ValueError("direction must be a vector [d] or basis [k, d]")
    norms = v.norm(dim=1)
    if not torch.isfinite(norms).all() or (norms < 1e-8).any():
        raise ValueError("every basis vector must have non-zero finite norm")
    q, _ = torch.linalg.qr(v.T.to(torch.float32))
    return q.T[: v.shape[0]].to(dtype)


@torch.no_grad()
def project_out_of_output_(
    weight: torch.Tensor, v: torch.Tensor, strength: float = 1.0
) -> None:
    """In-place: remove the subspace ``v`` from the *output* (row) space of a
    linear map ``y = W x``. ``weight`` has shape [d_out, d_in]; ``v`` lives in
    the d_out space. At ``strength=1.0`` (full erasure) ``V (W' x) == 0`` for
    all x; fractional strengths attenuate the subspace instead — used by
    calibrated surgery to target neutrality rather than annihilation."""
    basis = _as_orthonormal_basis(v, weight.dtype)
    weight.sub_(strength * (basis.T @ (basis @ weight)))


@torch.no_grad()
def project_out_of_columns_(
    weight: torch.Tensor, v: torch.Tensor, cols: slice, strength: float = 1.0
) -> None:
    """In-place: remove ``v`` from the output space of a column-slice of ``W``.

    Used for per-head attention edits: head ``h`` only touches columns
    ``[h*dh, (h+1)*dh)`` of ``out_proj.weight``, so projecting that slice
    silences head ``h`` inside the subspace while leaving every other head
    untouched.
    """
    basis = _as_orthonormal_basis(v, weight.dtype)
    block = weight[:, cols]
    weight[:, cols] = block - strength * (basis.T @ (basis @ block))


@torch.no_grad()
def project_vector_(bias: torch.Tensor, v: torch.Tensor, strength: float = 1.0) -> None:
    """In-place: remove the component of a bias/vector inside subspace ``v``."""
    basis = _as_orthonormal_basis(v, bias.dtype)
    bias.sub_(strength * (basis.T @ (basis @ bias)))


@torch.no_grad()
def project_out_of_input_(
    weight: torch.Tensor, v: torch.Tensor, strength: float = 1.0
) -> None:
    """In-place: make a linear map ``y = W x`` blind to subspace ``v`` in its
    *input* (column) space: W' = W (I - V^T V). Used to harden read-out maps
    (e.g. CLIP's text_projection) so any residual trace of the subspace
    reaching them cannot leak into the shared embedding space."""
    basis = _as_orthonormal_basis(v, weight.dtype)
    weight.sub_(strength * ((weight @ basis.T) @ basis))
