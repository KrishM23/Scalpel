import copy
import json

import torch

from scalpel.editing.surgeon import SurgeryConfig
from scalpel.pipelines.debias import run_debias_pipeline


def _run(tiny_clip, tmp_path=None, audit_only=False, **config_kwargs):
    config = SurgeryConfig(max_components=6, **config_kwargs)
    return run_debias_pipeline(
        model_id="tiny-clip-test",
        bias="gender_profession",
        config=config,
        save_dir=tmp_path,
        audit_only=audit_only,
        # Surgery edits weights in place; copy so the shared fixture survives.
        model=copy.deepcopy(tiny_clip.model),
        tokenizer=tiny_clip.tokenizer,
    )


def test_pipeline_report_structure(tiny_clip, tmp_path):
    result = _run(tiny_clip, tmp_path=tmp_path)
    report = result.report

    assert report["model_id"] == "tiny-clip-test"
    assert report["bias_spec"]["name"] == "gender_profession"
    assert 1 <= len(report["circuit"]["selected_components"]) <= 6
    assert report["surgery"]["num_edits"] >= 1
    metrics = report["metrics"]
    assert set(metrics) == {"bias_before", "bias_after", "bias_reduction", "retention"}
    assert -1.0 <= metrics["retention"]["geometry_retention"] <= 1.0

    # Artifacts persisted: edited weights + audit report.
    assert (tmp_path / "model" / "config.json").exists()
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["surgery"]["num_edits"] == report["surgery"]["num_edits"]


def test_surgery_annihilates_selected_component_writes(tiny_clip):
    result = _run(tiny_clip)
    lm = result.edited
    circuit = result.report["circuit"]
    v = torch.zeros(lm.d_model)

    # Reconstruct v via the direction layer's stored metadata is not exposed;
    # instead verify structurally: every selected component's weight can no
    # longer produce output along ANY direction it was projected against by
    # checking the projector fixed point W = (I - vv^T) W  =>  the matrix
    # (I - vv^T) applied again must be a no-op. We recover v from the edit:
    # for MLP, v is the left-null direction created by the edit.
    for comp in circuit["selected_components"]:
        if comp["kind"] == "mlp":
            w = lm.text_layers[comp["layer"]].mlp.fc2.weight
            # Smallest left singular value direction should annihilate outputs.
            _, s, _ = torch.linalg.svd(w)
            assert s[-1] < 1e-4  # rank deficiency introduced by the projection
            v = torch.linalg.svd(w)[0][:, -1]
            x = torch.randn(w.shape[1], 20)
            assert (v @ (w @ x)).abs().max() < 1e-3


def test_direction_layer_override(tiny_clip):
    result = _run(tiny_clip, direction_layer=2)
    assert result.report["circuit"]["direction_layer"] == 2


def test_audit_only_mode_edits_nothing(tiny_clip, tmp_path):
    baseline = {
        name: p.clone() for name, p in tiny_clip.model.named_parameters()
    }
    result = _run(tiny_clip, tmp_path=tmp_path, audit_only=True)
    report = result.report

    assert report["mode"] == "audit"
    assert "surgery" not in report
    assert set(report["metrics"]) == {"bias_before"}
    # Circuit diagnosis is still included (the free-tier teaser).
    assert report["circuit"]["selected_components"]
    # No model weights are persisted, only the report.
    assert not (tmp_path / "model").exists()
    assert (tmp_path / "report.json").exists()
    # And the source model is bit-identical (audit is non-destructive).
    for name, param in tiny_clip.model.named_parameters():
        assert torch.equal(param, baseline[name]), name


def test_multidirectional_surgery(tiny_clip):
    result = _run(tiny_clip, num_directions=3)
    report = result.report
    assert report["circuit"]["num_directions"] == 3
    assert all(edit["rank"] == 3 for edit in report["surgery"]["edits"])


def test_calibrated_surgery_selects_a_strength(tiny_clip):
    result = _run(tiny_clip, calibrate=True)
    report = result.report
    calibration = report["calibration"]
    strengths = [t["strength"] for t in calibration["trials"]]
    assert calibration["selected_strength"] in strengths
    assert report["surgery"]["strength"] == calibration["selected_strength"]
    # The selected trial must have the smallest residual |WEAT|.
    best = min(calibration["trials"], key=lambda t: abs(t["weat_effect_size"]))
    assert best["strength"] == calibration["selected_strength"]


def test_html_report_renders(tiny_clip):
    from scalpel.reporting import render_report_html

    for audit_only in (False, True):
        html = render_report_html(_run(tiny_clip, audit_only=audit_only).report)
        assert html.startswith("<!doctype html>")
        assert "Isolated bias circuit" in html
        if audit_only:
            assert "Bias Audit Report" in html
        else:
            assert "Edit manifest" in html


def test_all_catalog_specs_run(tiny_clip):
    """Every built-in bias benchmark must run end to end."""
    from scalpel.biases.catalog import bias_catalog

    for name, spec in bias_catalog().items():
        spec.validate()
        result = run_debias_pipeline(
            model_id="tiny-clip-test",
            bias=name,
            config=SurgeryConfig(max_components=3),
            audit_only=True,
            model=copy.deepcopy(tiny_clip.model),
            tokenizer=tiny_clip.tokenizer,
        )
        assert result.report["bias_spec"]["name"] == name
