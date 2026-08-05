import torch

from scalpel.editing.rank_one import (
    project_out_of_columns_,
    project_out_of_input_,
    project_out_of_output_,
    project_vector_,
)


def _unit(d: int, seed: int = 0) -> torch.Tensor:
    v = torch.randn(d, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def test_output_projection_annihilates_direction():
    torch.manual_seed(1)
    w = torch.randn(16, 32)
    v = _unit(16)
    original = w.clone()
    project_out_of_output_(w, v)
    x = torch.randn(32, 100)
    # No input can produce any output along v anymore.
    assert (v @ (w @ x)).abs().max() < 1e-5
    # The update is exactly rank one.
    assert torch.linalg.matrix_rank(original - w, tol=1e-5) == 1
    # Orthogonal complement is untouched.
    u = torch.randn(16)
    u = u - (u @ v) * v
    assert torch.allclose(u @ w, u @ original, atol=1e-5)


def test_output_projection_is_idempotent():
    torch.manual_seed(2)
    w = torch.randn(8, 8)
    v = _unit(8, seed=3)
    project_out_of_output_(w, v)
    once = w.clone()
    project_out_of_output_(w, v)
    assert torch.allclose(w, once, atol=1e-6)


def test_column_slice_projection_is_surgical():
    torch.manual_seed(4)
    w = torch.randn(16, 16)
    original = w.clone()
    v = _unit(16, seed=5)
    cols = slice(4, 8)
    project_out_of_columns_(w, v, cols)
    # Edited head slice can no longer write along v...
    x = torch.randn(4, 50)
    assert (v @ (w[:, cols] @ x)).abs().max() < 1e-5
    # ...while every other head's columns are bit-identical.
    untouched = [c for c in range(16) if not (4 <= c < 8)]
    assert torch.equal(w[:, untouched], original[:, untouched])


def test_input_projection_blinds_direction():
    torch.manual_seed(6)
    w = torch.randn(12, 24)
    v = _unit(24, seed=7)
    project_out_of_input_(w, v)
    assert (w @ v).abs().max() < 1e-5


def test_bias_vector_projection():
    b = torch.randn(10)
    v = _unit(10, seed=8)
    project_vector_(b, v)
    assert abs(float(b @ v)) < 1e-6


def test_multidirectional_basis_projection():
    """A [k, d] basis erases the whole subspace with a rank-k update."""
    torch.manual_seed(9)
    w = torch.randn(20, 30)
    original = w.clone()
    basis = torch.linalg.qr(torch.randn(20, 3))[0].T  # orthonormal [3, 20]
    project_out_of_output_(w, basis)
    x = torch.randn(30, 100)
    # No input can produce output anywhere inside the 3-D subspace.
    assert (basis @ (w @ x)).abs().max() < 1e-5
    # The update has rank exactly k = 3.
    assert torch.linalg.matrix_rank(original - w, tol=1e-5) == 3
    # Orthogonal complement untouched.
    u = torch.randn(20)
    u = u - basis.T @ (basis @ u)
    assert torch.allclose(u @ w, u @ original, atol=1e-4)


def test_non_orthonormal_basis_is_orthonormalized():
    """Passing correlated (non-orthonormal) directions must still work."""
    torch.manual_seed(10)
    w = torch.randn(12, 12)
    v1 = torch.randn(12)
    v2 = v1 + 0.1 * torch.randn(12)  # nearly parallel to v1
    project_out_of_output_(w, torch.stack([v1, v2]))
    x = torch.randn(12, 50)
    assert ((v1 / v1.norm()) @ (w @ x)).abs().max() < 1e-4
    assert ((v2 / v2.norm()) @ (w @ x)).abs().max() < 1e-4
