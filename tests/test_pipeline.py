import copy
import json

import torch

from scalpel.editing.surgeon import SurgeryConfig
from scalpel.pipelines.debias import run_debias_pipeline


def _run(tiny_clip, tmp_path=None, **config_kwargs):
    config = SurgeryConfig(max_components=6, **config_kwargs)
    return run_debias_pipeline(
        model_id="tiny-clip-test",
        bias="gender_profession",
        config=config,
        save_dir=tmp_path,
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
