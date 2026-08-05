"""Commercial performance retention metrics.

The surgery removes exactly one direction out of d_model from a handful of
component write-spaces; these metrics verify that everything else survived:

- embedding_cosine_retention: mean cosine similarity between the original and
  edited embeddings of bias-neutral retention prompts (1.0 = unchanged).
- geometry_retention: Pearson correlation between the off-diagonal pairwise
  cosine-similarity matrices of the retention set before vs after surgery.
  This is what downstream ranking/retrieval quality depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RetentionReport:
    embedding_cosine_retention: float
    geometry_retention: float
    worst_prompt_cosine: float

    def to_dict(self) -> dict:
        return {
            "embedding_cosine_retention": round(self.embedding_cosine_retention, 6),
            "geometry_retention": round(self.geometry_retention, 6),
            "worst_prompt_cosine": round(self.worst_prompt_cosine, 6),
        }


def _offdiag(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return matrix[mask]


@torch.no_grad()
def evaluate_retention(
    original_embeddings: torch.Tensor, edited_embeddings: torch.Tensor
) -> RetentionReport:
    """Both inputs: [N, d] L2-normalized embeddings of the same prompts."""
    per_prompt = (original_embeddings * edited_embeddings).sum(dim=-1)

    sims_before = _offdiag(original_embeddings @ original_embeddings.T)
    sims_after = _offdiag(edited_embeddings @ edited_embeddings.T)
    stacked = torch.stack([sims_before, sims_after])
    geometry = torch.corrcoef(stacked)[0, 1].item()

    return RetentionReport(
        embedding_cosine_retention=per_prompt.mean().item(),
        geometry_retention=geometry,
        worst_prompt_cosine=per_prompt.min().item(),
    )
