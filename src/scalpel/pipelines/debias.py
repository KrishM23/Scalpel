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
    audit_only: bool = False,
    model: CLIPModel | None = None,
    tokenizer: CLIPTokenizerFast | None = None,
) -> DebiasResult:
    """Run one full surgery — or, with ``audit_only``, a diagnostic audit that
    measures the bias and localizes the responsible circuit without touching
    any weights. ``model``/``tokenizer`` may be injected to skip the Hugging
    Face download (tests, pre-warmed workers)."""
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

    # 3. Estimate the bias subspace and isolate the circuit writing onto it.
    direction = find_bias_direction(
        acts_a,
        acts_b,
        layer=config.direction_layer,
        num_directions=config.num_directions,
    )
    circuit = isolate_bias_circuit(
        lm,
        acts_a,
        acts_b,
        direction,
        max_components=config.max_components,
        cumulative_share=config.cumulative_share,
    )

    report = {
        "platform": "scalpel",
        "pipeline": "debias.v1",
        "mode": "audit" if audit_only else "edit",
        "model_id": model_id,
        "bias_spec": {
            "name": spec.name,
            "description": spec.description,
            "groups": [spec.group_a_label, spec.group_b_label],
            "num_contrastive_pairs": len(spec.paired_prompts),
        },
        "config": config.to_dict(),
        "circuit": circuit.to_dict(),
    }

    if audit_only:
        report["metrics"] = {"bias_before": bias_before.to_dict()}
    else:
        # 4. Surgery: closed-form low-rank projection edits. With calibration
        # enabled, sweep the erasure strength on the same isolated circuit and
        # keep the strength minimizing the residual |WEAT| — full projection
        # can overshoot past neutrality on some benchmarks.
        if config.calibrate:
            surgery, bias_after, trials = _calibrated_surgery(lm, circuit, config, spec)
            report["calibration"] = trials
        else:
            surgery = perform_surgery(lm, circuit, config)
            bias_after = evaluate_bias(lm, spec)

        # 5. Post-surgery audit.
        retention_after = lm.encode_text(spec.retention_prompts)
        retention = evaluate_retention(retention_before, retention_after)

        report["surgery"] = surgery.to_dict()
        report["metrics"] = {
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
        }

    report["runtime_seconds"] = round(time.time() - started, 2)

    artifact_path: Path | None = None
    if save_dir is not None:
        artifact_path = Path(save_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        if not audit_only:
            lm.model.save_pretrained(artifact_path / "model")
            lm.tokenizer.save_pretrained(artifact_path / "model")
        (artifact_path / "report.json").write_text(json.dumps(report, indent=2))

    return DebiasResult(report=report, edited=lm, artifact_path=artifact_path)


_CALIBRATION_STRENGTHS = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25)


def _calibrated_surgery(lm, circuit, config, spec):
    """Sweep erasure strengths and keep the one minimizing residual |WEAT|.

    The edit is linear in strength, so each trial restores the affected text
    tower from a snapshot and re-applies the projections scaled — no gradient
    steps, still fully deterministic.
    """
    snapshot = {
        name: tensor.detach().clone()
        for name, tensor in lm.model.state_dict().items()
        if name.startswith(("text_model.", "text_projection."))
    }

    trials = []
    best = None  # (|weat|, strength, surgery, bias_after)
    for strength in _CALIBRATION_STRENGTHS:
        lm.model.load_state_dict(snapshot, strict=False)
        surgery = perform_surgery(lm, circuit, config, strength=strength)
        bias_after = evaluate_bias(lm, spec)
        score = abs(bias_after.weat_effect_size)
        trials.append(
            {
                "strength": strength,
                "weat_effect_size": round(bias_after.weat_effect_size, 4),
                "mean_abs_association_gap": round(bias_after.mean_abs_association_gap, 6),
            }
        )
        if best is None or score < best[0]:
            best = (score, strength, surgery, bias_after)

    _, strength, surgery, bias_after = best
    # Leave the model in the winning state.
    lm.model.load_state_dict(snapshot, strict=False)
    surgery = perform_surgery(lm, circuit, config, strength=strength)
    return surgery, bias_after, {"selected_strength": strength, "trials": trials}


def _reduction(before: float, after: float) -> dict:
    """Relative reduction in |metric|; guards the divide-by-zero edge."""
    pct = 100.0 * (1.0 - abs(after) / abs(before)) if abs(before) > 1e-9 else 0.0
    return {"before": round(before, 4), "after": round(after, 4), "reduction_pct": round(pct, 1)}
