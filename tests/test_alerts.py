from scalpel.api.alerts import alerts_for_job, collect_alerts


def test_failed_job_alert():
    job = {
        "id": "job_1",
        "bias_name": "gender_profession",
        "mode": "edit",
        "status": "failed",
        "error": "RuntimeError: oom",
        "updated_at": "2026-08-05T00:00:00+00:00",
    }
    alerts = alerts_for_job(job, None)
    assert len(alerts) == 1
    assert alerts[0].kind == "job_failed"
    assert alerts[0].severity == "critical"


def test_audit_bias_detected():
    job = {
        "id": "job_2",
        "bias_name": "age_competence",
        "mode": "audit",
        "status": "succeeded",
        "updated_at": "2026-08-05T00:00:00+00:00",
    }
    report = {"metrics": {"bias_before": {"weat_effect_size": 1.2}}}
    alerts = alerts_for_job(job, report)
    assert [a.kind for a in alerts] == ["bias_detected"]


def test_strong_edit_does_not_residual_alert():
    """A successful surgery with large gap reduction must not keep nagging."""
    job = {
        "id": "job_3",
        "bias_name": "gender_profession",
        "mode": "edit",
        "status": "succeeded",
        "updated_at": "2026-08-05T00:00:00+00:00",
    }
    report = {
        "metrics": {
            "bias_reduction": {
                "weat_effect_size": {"before": 1.71, "after": 1.02, "reduction_pct": 40.0},
                "mean_abs_association_gap": {
                    "before": 0.015,
                    "after": 0.003,
                    "reduction_pct": 78.0,
                },
            }
        }
    }
    assert alerts_for_job(job, report) == []


def test_weak_edit_residual_alert():
    job = {
        "id": "job_3b",
        "bias_name": "ethnicity_valence",
        "mode": "edit",
        "status": "succeeded",
        "updated_at": "2026-08-05T00:00:00+00:00",
    }
    weak = {
        "metrics": {
            "bias_reduction": {
                "weat_effect_size": {"before": 1.0, "after": 0.9, "reduction_pct": 10.0},
                "mean_abs_association_gap": {
                    "before": 0.02,
                    "after": 0.018,
                    "reduction_pct": 10.0,
                },
            }
        }
    }
    assert alerts_for_job(job, weak)[0].kind == "residual_bias"


def test_overcorrection_when_gap_ok_but_sign_flips():
    job = {
        "id": "job_3c",
        "bias_name": "ethnicity_valence",
        "mode": "edit",
        "status": "succeeded",
        "updated_at": "2026-08-05T00:00:00+00:00",
    }
    flipped = {
        "metrics": {
            "bias_reduction": {
                "weat_effect_size": {"before": 0.9, "after": -0.35, "reduction_pct": 61.0},
                "mean_abs_association_gap": {
                    "before": 0.02,
                    "after": 0.004,
                    "reduction_pct": 80.0,
                },
            }
        }
    }
    assert alerts_for_job(job, flipped)[0].kind == "overcorrection"


def test_collect_suppresses_audit_after_edit():
    jobs = [
        {
            "id": "job_audit",
            "bias_name": "gender_profession",
            "mode": "audit",
            "status": "succeeded",
            "created_at": "2026-08-05T00:00:00+00:00",
            "updated_at": "2026-08-05T00:00:00+00:00",
            "report_json": '{"metrics":{"bias_before":{"weat_effect_size":1.7}}}',
        },
        {
            "id": "job_edit",
            "bias_name": "gender_profession",
            "mode": "edit",
            "status": "succeeded",
            "created_at": "2026-08-05T01:00:00+00:00",
            "updated_at": "2026-08-05T01:00:00+00:00",
            "report_json": (
                '{"metrics":{"bias_reduction":{'
                '"weat_effect_size":{"before":1.7,"after":1.0,"reduction_pct":40},'
                '"mean_abs_association_gap":{"before":0.01,"after":0.002,"reduction_pct":80}'
                "}}}"
            ),
        },
    ]
    kinds = [a.kind for a in collect_alerts(jobs)]
    assert "bias_detected" not in kinds
