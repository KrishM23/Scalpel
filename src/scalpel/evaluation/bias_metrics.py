"""Bias measurement on final (shared-space) embeddings.

Two complementary metrics, both computed on L2-normalized CLIP text features:

- WEAT effect size (Caliskan et al. 2017): standardized difference between the
  two stereotyped probe sets in their differential association with the two
  attribute groups. |d| near 0 means no measurable association bias; values
  around 1.0+ are large.
- Association gap statistics: for every probe prompt, the difference in cosine
  similarity to the group-A vs group-B attribute centroids. Reported as mean
  absolute gap and standard deviation across probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scalpel.biases.catalog import BiasSpec
from scalpel.models.registry import LoadedModel


@dataclass
class BiasReport:
    weat_effect_size: float
    mean_abs_association_gap: float
    association_gap_std: float
    per_probe_gaps: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "weat_effect_size": round(self.weat_effect_size, 4),
            "mean_abs_association_gap": round(self.mean_abs_association_gap, 6),
            "association_gap_std": round(self.association_gap_std, 6),
            "per_probe_gaps": {k: round(v, 6) for k, v in self.per_probe_gaps.items()},
        }


def _differential_association(
    targets: torch.Tensor, attrs_a: torch.Tensor, attrs_b: torch.Tensor
) -> torch.Tensor:
    """s(t, A, B) = mean_a cos(t, a) - mean_b cos(t, b), per target row."""
    return (targets @ attrs_a.T).mean(dim=1) - (targets @ attrs_b.T).mean(dim=1)


@torch.no_grad()
def evaluate_bias(lm: LoadedModel, spec: BiasSpec) -> BiasReport:
    prompts_a = [pair[0] for pair in spec.paired_prompts]
    prompts_b = [pair[1] for pair in spec.paired_prompts]
    emb_a = lm.encode_text(prompts_a)
    emb_b = lm.encode_text(prompts_b)
    emb_t1 = lm.encode_text(spec.probe_set_1)
    emb_t2 = lm.encode_text(spec.probe_set_2)

    # WEAT effect size.
    s_t1 = _differential_association(emb_t1, emb_a, emb_b)
    s_t2 = _differential_association(emb_t2, emb_a, emb_b)
    pooled = torch.cat([s_t1, s_t2])
    effect_size = ((s_t1.mean() - s_t2.mean()) / (pooled.std() + 1e-12)).item()

    # Per-probe association gaps against group centroids.
    centroid_a = emb_a.mean(dim=0)
    centroid_a = centroid_a / centroid_a.norm()
    centroid_b = emb_b.mean(dim=0)
    centroid_b = centroid_b / centroid_b.norm()
    probes = spec.probe_set_1 + spec.probe_set_2
    emb_probes = torch.cat([emb_t1, emb_t2], dim=0)
    gaps = emb_probes @ centroid_a - emb_probes @ centroid_b
    per_probe = {probe: gaps[i].item() for i, probe in enumerate(probes)}

    return BiasReport(
        weat_effect_size=effect_size,
        mean_abs_association_gap=gaps.abs().mean().item(),
        association_gap_std=gaps.std().item(),
        per_probe_gaps=per_probe,
    )
