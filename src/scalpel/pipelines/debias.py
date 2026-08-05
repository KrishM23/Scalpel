"""End-to-end debiasing pipeline.

    load model -> baseline audit -> record activations -> estimate bias
    direction -> isolate circuit -> closed-form surgery -> post audit ->
    retention check -> (optionally) persist edited weights + report

The whole pipeline is gradient-free and deterministic: every edit is a
closed-form rank-one projection, so results are exactly reproducible and the
audit report doubles as a compliance artifact.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from transformers import CLIPModel, CLIPTokenizerFast

from scalpel.biases.catalog import BiasSpec, get_bias_spec
from scalpel.editing.surgeon import SurgeryConfig, perform_surgery
from scalpel.evaluation.bias_metrics import evaluate_bias
from scalpel.evaluation.performance import evaluate_retention
from scalpel.interpretability.activations import record_component_activations
from scalpel.interpretability.circuits import isolate_bias_circuit
from scalpel.interpretability.directions import find_bias_direction
from scalpel.models.registry import LoadedModel, load_model


@dataclass
class DebiasResult:
    report: dict
    edited: LoadedModel
    artifact_path: Path | None


def run_debias_pipeline(
    model_id: str,
    bias: str | BiasSpec = "gender_profession",
    config: SurgeryConfig | None = None,
    save_dir: str | Path | None = None,
    model: CLIPModel | None = None,
    tokenizer: CLIPTokenizerFast | None = None,
) -> DebiasResult:
    """Run one full surgery. ``model``/``tokenizer`` may be injected to skip
    the Hugging Face download (tests, pre-warmed workers)."""
    config = config or SurgeryConfig()
    spec = get_bias_spec(bias) if isinstance(bias, str) else bias
    spec.validate()
    started = time.time()

    lm = load_model(model_id, device=config.device, model=model, tokenizer=tokenizer)

    # 1. Baseline audit (bias + retention reference embeddings).
    bias_before = evaluate_bias(lm, spec)
    retention_before = lm.encode_text(spec.retention_prompts)

    # 2. Mechanistic pass: record component activations on contrastive pairs.
    prompts_a = [pair[0] for pair in spec.paired_prompts]
    prompts_b = [pair[1] for pair in spec.paired_prompts]
    acts_a = record_component_activations(lm, prompts_a)
    acts_b = record_component_activations(lm, prompts_b)

    # 3. Estimate the bias direction and isolate the circuit writing onto it.
    direction = find_bias_direction(acts_a, acts_b, layer=config.direction_layer)
    circuit = isolate_bias_circuit(
        lm,
        acts_a,
        acts_b,
        direction,
        max_components=config.max_components,
        cumulative_share=config.cumulative_share,
    )

    # 4. Surgery: closed-form rank-one edits.
    surgery = perform_surgery(lm, circuit, config)

    # 5. Post-surgery audit.
    bias_after = evaluate_bias(lm, spec)
    retention_after = lm.encode_text(spec.retention_prompts)
    retention = evaluate_retention(retention_before, retention_after)

    report = {
        "platform": "scalpel",
        "pipeline": "debias.v1",
        "model_id": model_id,
        "bias_spec": {
            "name": spec.name,
            "description": spec.description,
            "groups": [spec.group_a_label, spec.group_b_label],
            "num_contrastive_pairs": len(spec.paired_prompts),
        },
        "config": config.to_dict(),
        "circuit": circuit.to_dict(),
        "surgery": surgery.to_dict(),
        "metrics": {
            "bias_before": bias_before.to_dict(),
            "bias_after": bias_after.to_dict(),
            "bias_reduction": {
                "weat_effect_size": _reduction(
                    bias_before.weat_effect_size, bias_after.weat_effect_size
                ),
                "mean_abs_association_gap": _reduction(
                    bias_before.mean_abs_association_gap,
                    bias_after.mean_abs_association_gap,
                ),
            },
            "retention": retention.to_dict(),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }

    artifact_path: Path | None = None
    if save_dir is not None:
        artifact_path = Path(save_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        lm.model.save_pretrained(artifact_path / "model")
        lm.tokenizer.save_pretrained(artifact_path / "model")
        (artifact_path / "report.json").write_text(json.dumps(report, indent=2))

    return DebiasResult(report=report, edited=lm, artifact_path=artifact_path)


def _reduction(before: float, after: float) -> dict:
    """Relative reduction in |metric|; guards the divide-by-zero edge."""
    pct = 100.0 * (1.0 - abs(after) / abs(before)) if abs(before) > 1e-9 else 0.0
    return {"before": round(before, 4), "after": round(after, 4), "reduction_pct": round(pct, 1)}
