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
        api_keys=settings_overrides.pop("api_keys", [API_KEY]),
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


def test_marketing_and_console_pages(client):
    landing = client.get("/")
    assert landing.status_code == 200
    assert "text/html" in landing.headers["content-type"]
    assert "Cut bias out" in landing.text
    assert "/static/site.css" in landing.text
    assert client.get("/pricing").status_code == 404

    for path, needle in [
        ("/product", "From association gap"),
        ("/developers", "Two calls to your first"),
        ("/security", "Tenant isolation"),
        ("/privacy", "What we collect"),
        ("/terms", "Terms of service"),
    ]:
        page = client.get(path)
        assert page.status_code == 200, path
        assert needle in page.text, path

    assert client.get("/static/favicon.svg").status_code == 200
    assert client.get("/static/site.css").status_code == 200
    assert client.get("/static/session.js").status_code == 200
    assert "ScalpelSession" in client.get("/static/session.js").text
    assert "/static/session.js" in landing.text
    assert 'id="liveDemo"' in landing.text
    assert "/static/live-demo.js" in landing.text
    assert client.get("/static/live-demo.js").status_code == 200
    assert "startLiveDemo" in client.get("/static/live-demo.js").text


    console = client.get("/app")
    assert console.status_code == 200
    assert "Bias Operations" in console.text

    login = client.get("/login")
    assert login.status_code == 200
    assert 'data-mode="login"' in login.text

    signup = client.get("/signup")
    assert signup.status_code == 200
    assert 'data-mode="signup"' in signup.text


def test_signup_login_issues_api_key(client):
    payload = {
        "email": "ops@acme.test",
        "password": "securepass1",
        "name": "Ops Lead",
        "company": "Acme",
    }
    created = client.post("/v1/auth/signup", json=payload)
    assert created.status_code == 200
    body = created.json()
    assert body["api_key"].startswith("sk_live_")
    assert body["tenant"]
    assert body["plan"] == "free"

    dup = client.post("/v1/auth/signup", json=payload)
    assert dup.status_code == 400

    bad = client.post(
        "/v1/auth/login",
        json={"email": payload["email"], "password": "wrong-password"},
    )
    assert bad.status_code == 401

    logged_in = client.post(
        "/v1/auth/login",
        json={"email": payload["email"], "password": payload["password"]},
    )
    assert logged_in.status_code == 200
    assert logged_in.json()["api_key"] == body["api_key"]

    headers = {"X-API-Key": body["api_key"]}
    usage = client.get("/v1/usage", headers=headers)
    assert usage.status_code == 200
    assert usage.json()["tenant"] == body["tenant"]
    assert usage.json()["plan"] == "free"
    assert usage.json()["allows_edit"] is False

    # Free workspaces may audit but not edit.
    blocked = client.post(
        "/v1/edit-jobs",
        headers=headers,
        json={
            "model_id": "openai/clip-vit-base-patch32",
            "bias": "gender_profession",
            "mode": "edit",
        },
    )
    assert blocked.status_code == 402


def test_auth_required(client):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401


def test_bearer_auth_accepted(client):
    response = client.get(
        "/v1/models", headers={"Authorization": "Bearer sk_test_12345"}
    )
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_open_keys_mode_isolates_tenants(tmp_path, monkeypatch):
    monkeypatch.setenv("SCALPEL_OPEN_KEYS", "1")
    with _make_client(tmp_path, monkeypatch, api_keys=[]) as client:
        a = {"X-API-Key": "sk_finova_aaa"}
        b = {"X-API-Key": "sk_northstar_bbb"}
        assert client.get("/v1/usage", headers=a).json()["tenant"] == "finova_aaa"
        assert client.get("/v1/usage", headers=b).json()["tenant"] == "northstar_bbb"
        # Open-mode tenants get full edit capability.
        assert client.get("/v1/usage", headers=a).json()["allows_edit"] is True
        assert client.get("/v1/usage", headers=a).json()["plan"] == "enterprise"
        job = client.post(
            "/v1/edit-jobs",
            headers=a,
            json={"model_id": "openai/clip-vit-base-patch32", "bias": "gender_profession"},
        ).json()
        listing_a = client.get("/v1/edit-jobs", headers=a).json()
        listing_b = client.get("/v1/edit-jobs", headers=b).json()
        assert any(item["id"] == job["id"] for item in listing_a)
        assert listing_b == []


def test_multi_configured_keys_have_isolated_jobs(tmp_path, monkeypatch):
    with _make_client(
        tmp_path,
        monkeypatch,
        api_keys=["acme:sk_acme_1", "beta:sk_beta_1"],
        tenant_plans={"acme": "pro", "beta": "enterprise"},
    ) as client:
        ha, hb = {"X-API-Key": "sk_acme_1"}, {"X-API-Key": "sk_beta_1"}
        job = client.post(
            "/v1/edit-jobs",
            headers=ha,
            json={"model_id": "openai/clip-vit-base-patch32", "bias": "gender_profession"},
        ).json()
        assert job["tenant"] == "acme"
        assert any(
            item["id"] == job["id"] for item in client.get("/v1/edit-jobs", headers=ha).json()
        )
        assert client.get("/v1/edit-jobs", headers=hb).json() == []
        assert client.get("/v1/usage", headers=hb).json()["allows_edit"] is True


def test_catalogs(client):
    catalog = client.get("/v1/models", headers=HEADERS).json()
    assert catalog["accepts_any_huggingface_id"] is True
    assert "clip" in catalog["families"] and "causal_lm" in catalog["families"]
    featured_ids = {m["model_id"] for m in catalog["featured"]}
    assert "openai/clip-vit-base-patch32" in featured_ids
    assert "patrickjohncyh/fashion-clip" in featured_ids
    biases = {b["name"] for b in client.get("/v1/biases", headers=HEADERS).json()}
    assert {
        "gender_profession",
        "age_competence",
        "ethnicity_valence",
        "religion_valence",
        "disability_competence",
    } <= biases


def test_probe_accepts_any_classified_model(client, monkeypatch):
    import scalpel.api.app as app_mod
    from scalpel.models.registry import ModelProbe

    monkeypatch.setattr(
        app_mod,
        "probe_model",
        lambda model_id, trust_remote_code=True: ModelProbe(
            model_id=model_id,
            family="clip",
            model_type="clip",
            architecture_key="clip",
            description="stub",
        ),
    )
    body = client.get(
        "/v1/models/probe",
        params={"model_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"},
        headers=HEADERS,
    )
    assert body.status_code == 200
    assert body.json()["family"] == "clip"

    # Job create also uses probe — non-featured ids must be accepted.
    monkeypatch.setattr(
        app_mod,
        "probe_model",
        lambda model_id, trust_remote_code=True: ModelProbe(
            model_id=model_id,
            family="causal_lm",
            model_type="gpt2",
            architecture_key="gpt2",
            description="stub",
        ),
    )
    response = client.post(
        "/v1/edit-jobs",
        headers=HEADERS,
        json={"model_id": "gpt2", "bias": "gender_profession", "mode": "audit"},
    )
    assert response.status_code == 202


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

    def capture(url, content=None, headers=None, json=None, timeout=None):
        body = json if json is not None else __import__("json").loads(content)
        calls.append((url, body, headers or {}))

    monkeypatch.setattr(jobs_module.httpx, "post", capture)
    with _make_client(
        tmp_path, monkeypatch, webhook_secret="whsec_test"
    ) as client:
        job = _submit(client, webhook_url="https://client.example/hook").json()
        _wait_for_terminal(client, job["id"])
    assert len(calls) == 1
    url, payload, headers = calls[0]
    assert url == "https://client.example/hook"
    assert payload["job_id"] == job["id"]
    assert payload["status"] == "succeeded"
    assert headers.get("X-Scalpel-Signature", "").startswith("sha256=")


def test_alerts_for_failed_and_hot_audit(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as client:
        # Succeeded audit with high WEAT in FAKE_REPORT → bias_detected alert.
        job = _submit(client, mode="audit").json()
        _wait_for_terminal(client, job["id"])
        alerts = client.get("/v1/alerts", headers=HEADERS).json()
        assert any(a["kind"] == "bias_detected" and a["job_id"] == job["id"] for a in alerts)

        # Force a failure via a bad model that passes probe stub… use monkeypatch on runner.
        import scalpel.api.app as app_mod
        from scalpel.models.registry import ModelProbe

        monkeypatch.setattr(
            app_mod,
            "probe_model",
            lambda model_id, trust_remote_code=True: ModelProbe(
                model_id=model_id, family="clip", model_type="clip",
                architecture_key="clip", description="stub",
            ),
        )

        def boom(*_a, **_k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(jobs_module, "run_debias_pipeline", boom)
        failed = _submit(client, mode="audit").json()
        final = _wait_for_terminal(client, failed["id"])
        assert final["status"] == "failed"
        alerts = client.get("/v1/alerts", headers=HEADERS).json()
        assert any(a["kind"] == "job_failed" and a["job_id"] == failed["id"] for a in alerts)


def test_ready_and_health(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["users_db"] == "sqlite"
    assert "open_keys" not in health
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert "X-Request-ID" in ready.headers
    assert ready.headers["X-Content-Type-Options"] == "nosniff"


def test_ready_fails_without_auth_path(tmp_path, monkeypatch):
    monkeypatch.delenv("SCALPEL_OPEN_KEYS", raising=False)
    with _make_client(
        tmp_path,
        monkeypatch,
        api_keys=[],
        require_api_keys=True,
        public_signup=False,
    ) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        body = ready.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["auth"] == "no_api_keys"


def test_signup_disabled_when_public_signup_off(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch, public_signup=False) as client:
        response = client.post(
            "/v1/auth/signup",
            json={
                "email": "blocked@acme.test",
                "password": "securepass1",
                "name": "Blocked",
                "company": "Acme",
            },
        )
        assert response.status_code == 403


def test_rejects_unknown_model_and_bias(client, monkeypatch):
    import scalpel.api.app as app_mod
    from scalpel.models.registry import UnsupportedArchitectureError

    monkeypatch.setattr(
        app_mod,
        "probe_model",
        lambda model_id, trust_remote_code=True: (_ for _ in ()).throw(
            UnsupportedArchitectureError(f"Unsupported model '{model_id}'")
        ),
    )
    assert _submit(client, model_id="evil/unknown").status_code == 422
    # Restore a permissive probe for the bias check path.
    monkeypatch.setattr(
        app_mod,
        "probe_model",
        lambda model_id, trust_remote_code=True: __import__(
            "scalpel.models.registry", fromlist=["ModelProbe"]
        ).ModelProbe(
            model_id=model_id,
            family="clip",
            model_type="clip",
            architecture_key="clip",
            description="stub",
        ),
    )
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


def test_jobstore_migrates_v01_database(tmp_path):
    """Databases created before the 'mode' column existed must still work."""
    import sqlite3

    from scalpel.api.jobs import JobStore

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, tenant TEXT NOT NULL,"
            " model_id TEXT NOT NULL, bias_name TEXT NOT NULL, status TEXT NOT NULL,"
            " created_at TEXT NOT NULL, updated_at TEXT NOT NULL, error TEXT,"
            " artifact_dir TEXT, report_json TEXT)"
        )
        conn.execute(
            "INSERT INTO jobs VALUES ('job_old', 'acme', 'm', 'b', 'succeeded',"
            " '2026-08-04T00:00:00', '2026-08-04T00:00:00', NULL, NULL, NULL)"
        )

    store = JobStore(db)
    rows = store.list_for_tenant("acme")
    assert rows[0]["mode"] == "edit"  # backfilled default


def test_incomplete_jobs_without_payload_fail_on_recover(tmp_path, monkeypatch):
    from scalpel.api.jobs import JobRunner, JobStore

    store = JobStore(tmp_path / "db.sqlite")
    job_id = store.create("acme", "openai/clip-vit-base-patch32", "gender_profession")
    store.update(job_id, status="running")
    runner = JobRunner(
        store,
        Settings(api_keys=[API_KEY], artifact_dir=tmp_path / "a", database_path=tmp_path / "db.sqlite"),
    )
    assert runner.recover() == 0
    row = store.get(job_id, "acme")
    assert row["status"] == "failed"
    assert "restart" in row["error"].lower() or "timed out" in row["error"].lower()
    runner.shutdown()


def test_stale_queued_job_is_reaped(tmp_path):
    from datetime import datetime, timedelta, timezone

    from scalpel.api.jobs import JobStore

    store = JobStore(tmp_path / "db.sqlite")
    job_id = store.create("acme", "openai/clip-vit-base-patch32", "gender_profession")
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET updated_at = ?, status = 'queued' WHERE id = ?",
            (old, job_id),
        )
    failed = store.reap_stale(queued_timeout_s=60, running_timeout_s=3600)
    assert job_id in failed
    assert store.get(job_id, "acme")["status"] == "failed"


def test_incomplete_jobs_with_payload_requeue(tmp_path, monkeypatch):
    from scalpel.api.jobs import JobRunner, JobStore
    from scalpel.api.schemas import EditJobRequest

    def fake_pipeline(model_id, bias, config, save_dir=None, audit_only=False, **_kw):
        return SimpleNamespace(report=dict(FAKE_REPORT), artifact_path=None)

    monkeypatch.setattr(jobs_module, "run_debias_pipeline", fake_pipeline)
    store = JobStore(tmp_path / "db.sqlite")
    req = EditJobRequest(model_id="openai/clip-vit-base-patch32", bias="gender_profession")
    job_id = store.create(
        "acme",
        req.model_id,
        "gender_profession",
        request_json=req.model_dump_json(),
    )
    store.update(job_id, status="running")
    runner = JobRunner(
        store,
        Settings(api_keys=[API_KEY], artifact_dir=tmp_path / "a", database_path=tmp_path / "db.sqlite"),
    )
    assert runner.recover() == 1
    deadline = time.time() + 5
    while time.time() < deadline:
        row = store.get(job_id, "acme")
        if row["status"] == "succeeded":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("requeued job did not succeed")
    runner.shutdown()


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
