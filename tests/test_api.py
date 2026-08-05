import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import scalpel.api.jobs as jobs_module
from scalpel.api.app import create_app
from scalpel.config import Settings

API_KEY = "acme:sk_test_12345"
HEADERS = {"X-API-Key": "sk_test_12345"}

FAKE_REPORT = {
    "platform": "scalpel",
    "pipeline": "debias.v1",
    "mode": "edit",
    "model_id": "openai/clip-vit-base-patch32",
    "bias_spec": {
        "name": "gender_profession",
        "description": "test",
        "groups": ["male", "female"],
        "num_contrastive_pairs": 48,
    },
    "config": {},
    "circuit": {
        "direction_layer": 9,
        "num_directions": 1,
        "direction_separation": 9.5,
        "selected_components": [
            {"kind": "attn_head", "layer": 10, "head": 3, "label": "L10.H3",
             "score": 0.5, "share": 0.2},
        ],
    },
    "surgery": {"num_edits": 3, "edits": [
        {"target": "text_projection.weight", "edit": "input-space projection", "rank": 1},
    ]},
    "metrics": {
        "bias_before": {"weat_effect_size": 1.7, "mean_abs_association_gap": 0.0157,
                        "association_gap_std": 0.017, "per_probe_gaps": {"a probe": 0.01}},
        "bias_after": {"weat_effect_size": 1.4, "mean_abs_association_gap": 0.0029,
                       "association_gap_std": 0.004, "per_probe_gaps": {"a probe": 0.001}},
        "bias_reduction": {
            "weat_effect_size": {"before": 1.7, "after": 1.4, "reduction_pct": 16.0},
            "mean_abs_association_gap": {"before": 0.0157, "after": 0.0029,
                                         "reduction_pct": 81.6},
        },
        "retention": {"embedding_cosine_retention": 0.9995, "geometry_retention": 0.9997,
                      "worst_prompt_cosine": 0.9976},
    },
    "runtime_seconds": 3.3,
}


def _make_client(tmp_path, monkeypatch, **settings_overrides):
    """API client with the pipeline stubbed out (no model downloads)."""

    def fake_pipeline(model_id, bias, config, save_dir=None, audit_only=False, **_kw):
        report = dict(FAKE_REPORT, model_id=model_id, mode="audit" if audit_only else "edit")
        if audit_only:
            report = dict(report)
            report.pop("surgery")
            report["metrics"] = {"bias_before": FAKE_REPORT["metrics"]["bias_before"]}
        # The real pipeline creates save_dir; the stub skips weights entirely.
        return SimpleNamespace(report=report, artifact_path=None)

    monkeypatch.setattr(jobs_module, "run_debias_pipeline", fake_pipeline)
    settings = Settings(
        api_keys=[API_KEY],
        artifact_dir=tmp_path / "artifacts",
        database_path=tmp_path / "scalpel.db",
        tenant_plans=settings_overrides.pop("tenant_plans", {}),
        **settings_overrides,
    )
    return TestClient(create_app(settings))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as test_client:
        yield test_client


def _wait_for_terminal(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/edit-jobs/{job_id}", headers=HEADERS).json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job never reached a terminal state")


def _submit(client, **overrides):
    payload = {"model_id": "openai/clip-vit-base-patch32", "bias": "gender_profession"}
    payload.update(overrides)
    return client.post("/v1/edit-jobs", headers=HEADERS, json=payload)


def test_health_is_public(client):
    assert client.get("/health").json()["status"] == "ok"


def test_web_console_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Scalpel" in response.text


def test_auth_required(client):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401


def test_catalogs(client):
    models = client.get("/v1/models", headers=HEADERS).json()
    assert any(m["model_id"] == "openai/clip-vit-base-patch32" for m in models)
    biases = {b["name"] for b in client.get("/v1/biases", headers=HEADERS).json()}
    assert {"gender_profession", "age_competence", "ethnicity_valence"} <= biases


def test_edit_job_lifecycle(client):
    response = _submit(client)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert job["tenant"] == "acme"
    assert job["mode"] == "edit"

    final = _wait_for_terminal(client, job["id"])
    assert final["status"] == "succeeded"

    report = client.get(f"/v1/edit-jobs/{job['id']}/report", headers=HEADERS).json()
    assert report["surgery"]["num_edits"] == 3

    html = client.get(f"/v1/edit-jobs/{job['id']}/report.html", headers=HEADERS)
    assert html.status_code == 200
    assert "Bias Surgery Report" in html.text

    listing = client.get("/v1/edit-jobs", headers=HEADERS).json()
    assert any(item["id"] == job["id"] for item in listing)


def test_audit_mode_job(client):
    job = _submit(client, mode="audit").json()
    final = _wait_for_terminal(client, job["id"])
    assert final["status"] == "succeeded"
    report = client.get(f"/v1/edit-jobs/{job['id']}/report", headers=HEADERS).json()
    assert report["mode"] == "audit"
    assert "surgery" not in report
    html = client.get(f"/v1/edit-jobs/{job['id']}/report.html", headers=HEADERS)
    assert "Bias Audit Report" in html.text


def test_artifact_missing_returns_404(client):
    job = _submit(client, save_artifact=False).json()
    _wait_for_terminal(client, job["id"])
    response = client.get(f"/v1/edit-jobs/{job['id']}/artifact", headers=HEADERS)
    assert response.status_code == 404


def test_free_plan_is_audit_only(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch, tenant_plans={"acme": "free"}) as client:
        edit = _submit(client)
        assert edit.status_code == 402
        assert "audit-only" in edit.json()["detail"]
        audit = _submit(client, mode="audit")
        assert audit.status_code == 202


def test_monthly_quota_enforced(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch, tenant_plans={"acme": "free"}) as client:
        for _ in range(10):  # free plan: 10 jobs/month
            assert _submit(client, mode="audit").status_code == 202
        over = _submit(client, mode="audit")
        assert over.status_code == 402
        assert "quota" in over.json()["detail"]


def test_usage_endpoint(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch, tenant_plans={"acme": "pro"}) as client:
        usage = client.get("/v1/usage", headers=HEADERS).json()
        assert usage == {
            "tenant": "acme", "plan": "pro", "jobs_this_month": 0,
            "monthly_job_limit": 100, "allows_edit": True,
        }
        _submit(client)
        usage = client.get("/v1/usage", headers=HEADERS).json()
        assert usage["jobs_this_month"] == 1


def test_webhook_fired_on_completion(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        jobs_module.httpx, "post",
        lambda url, json, timeout: calls.append((url, json)),
    )
    with _make_client(tmp_path, monkeypatch) as client:
        job = _submit(client, webhook_url="https://client.example/hook").json()
        _wait_for_terminal(client, job["id"])
    assert len(calls) == 1
    url, payload = calls[0]
    assert url == "https://client.example/hook"
    assert payload["job_id"] == job["id"]
    assert payload["status"] == "succeeded"


def test_rejects_unknown_model_and_bias(client):
    assert _submit(client, model_id="evil/unknown").status_code == 422
    assert _submit(client, bias="nonexistent").status_code == 422


def test_custom_bias_spec_accepted(client):
    spec = {
        "name": "custom_age",
        "paired_prompts": [["a young person", "an old person"]] * 4,
        "probe_set_1": ["a photo of an athlete"],
        "probe_set_2": ["a photo of a retiree"],
        "retention_prompts": ["a dog", "a car", "a tree", "a house"],
    }
    response = _submit(client, bias=spec)
    assert response.status_code == 202
    assert response.json()["bias_name"] == "custom_age"


def test_unknown_job_is_404(client):
    assert client.get("/v1/edit-jobs/job_doesnotexist", headers=HEADERS).status_code == 404


def test_artifact_packaging(tmp_path):
    """HTML report + weights zip are written next to saved artifacts."""
    from scalpel.api.jobs import JobRunner

    artifact_dir = tmp_path / "job"
    (artifact_dir / "model").mkdir(parents=True)
    (artifact_dir / "model" / "weights.bin").write_bytes(b"\x00" * 128)

    JobRunner._package_artifacts(artifact_dir, FAKE_REPORT)

    assert (artifact_dir / "report.html").exists()
    assert "Bias Surgery Report" in (artifact_dir / "report.html").read_text()
    assert (artifact_dir / "model.zip").exists()
