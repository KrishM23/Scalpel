"""Workspace alerts derived from job status and audit metrics.

Alerts are actionable ops signals — not a scoreboard of every past WEAT value.
Successful surgeries that already cut lean substantially do not keep alerting.
Unremediated audits are suppressed once a later edit succeeds for that bias.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["critical", "warning"]
AlertKind = Literal["job_failed", "bias_detected", "residual_bias", "overcorrection"]


@dataclass(frozen=True)
class Alert:
    id: str
    job_id: str
    kind: AlertKind
    severity: Severity
    title: str
    detail: str
    bias_name: str
    mode: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "bias_name": self.bias_name,
            "mode": self.mode,
            "created_at": self.created_at,
        }


def _weat_before_after(report: dict | None) -> tuple[float | None, float | None]:
    if not report:
        return None, None
    metrics = report.get("metrics") or {}
    before = after = None
    if "bias_before" in metrics:
        before = float(metrics["bias_before"]["weat_effect_size"])
    reduction = metrics.get("bias_reduction") or {}
    weat = reduction.get("weat_effect_size") or {}
    if "before" in weat:
        before = float(weat["before"])
    if "after" in weat:
        after = float(weat["after"])
    elif "bias_after" in metrics:
        after = float(metrics["bias_after"]["weat_effect_size"])
    return before, after


def _gap_reduction_pct(report: dict | None) -> float | None:
    if not report:
        return None
    red = (report.get("metrics") or {}).get("bias_reduction") or {}
    gap = red.get("mean_abs_association_gap") or {}
    if "reduction_pct" in gap:
        return float(gap["reduction_pct"])
    return None


def _parse_report(job: dict, reports: dict[str, dict | None]) -> dict | None:
    report = reports.get(job["id"])
    if report is not None:
        return report
    raw = job.get("report_json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


def alerts_for_job(
    job: dict,
    report: dict | None = None,
    *,
    weat_threshold: float = 0.5,
    overcorrection_threshold: float = 0.3,
    min_gap_reduction_pct: float = 40.0,
) -> list[Alert]:
    """Return zero or more alerts for a single job row (+ optional report)."""
    job_id = job["id"]
    bias = job.get("bias_name") or "unknown"
    mode = job.get("mode") or "edit"
    created = job.get("updated_at") or job.get("created_at") or ""
    out: list[Alert] = []

    if job.get("status") == "failed":
        error = (job.get("error") or "unknown error").strip()
        out.append(
            Alert(
                id=f"{job_id}:job_failed",
                job_id=job_id,
                kind="job_failed",
                severity="critical",
                title="Surgery failed",
                detail=error[:240],
                bias_name=bias,
                mode=mode,
                created_at=created,
            )
        )
        return out

    if job.get("status") != "succeeded":
        return out

    before, after = _weat_before_after(report)

    if mode == "audit" and before is not None and abs(before) >= weat_threshold:
        out.append(
            Alert(
                id=f"{job_id}:bias_detected",
                job_id=job_id,
                kind="bias_detected",
                severity="warning",
                title="Bias detected (unremediated)",
                detail=f"Audit WEAT effect size {before:.3f} exceeds threshold {weat_threshold}",
                bias_name=bias,
                mode=mode,
                created_at=created,
            )
        )

    if mode == "edit" and before is not None and after is not None:
        gap_red = _gap_reduction_pct(report)
        # Residual only when surgery under-performed — not when WEAT is merely
        # non-zero after a strong gap reduction (common on gender benchmarks).
        underperformed = gap_red is not None and gap_red < min_gap_reduction_pct
        got_worse = abs(after) > abs(before) * 0.95 and abs(after) >= weat_threshold
        if underperformed or got_worse:
            out.append(
                Alert(
                    id=f"{job_id}:residual_bias",
                    job_id=job_id,
                    kind="residual_bias",
                    severity="warning",
                    title="Surgery under-performed",
                    detail=(
                        f"WEAT {before:.3f} → {after:.3f}"
                        + (
                            f"; association-gap reduction only {gap_red:.0f}%"
                            if gap_red is not None
                            else ""
                        )
                        + ". Re-run with higher k or calibration."
                    ),
                    bias_name=bias,
                    mode=mode,
                    created_at=created,
                )
            )
        elif (
            abs(before) >= weat_threshold
            and (before > 0) != (after > 0)
            and abs(after) >= overcorrection_threshold
        ):
            out.append(
                Alert(
                    id=f"{job_id}:overcorrection",
                    job_id=job_id,
                    kind="overcorrection",
                    severity="warning",
                    title="Overcorrection past neutrality",
                    detail=(
                        f"WEAT flipped {before:.3f} → {after:.3f}. "
                        "Enable calibration to target neutrality."
                    ),
                    bias_name=bias,
                    mode=mode,
                    created_at=created,
                )
            )

    return out


def collect_alerts(
    jobs: list[dict],
    reports: dict[str, dict | None] | None = None,
    *,
    weat_threshold: float = 0.5,
    overcorrection_threshold: float = 0.3,
    min_gap_reduction_pct: float = 40.0,
) -> list[Alert]:
    """Aggregate actionable alerts across a tenant's jobs, newest first.

    - Failures always alert.
    - Audit ``bias_detected`` is suppressed when a later successful *edit*
      exists for the same ``bias_name`` (already remediated).
    - Only the newest unremediated audit per bias is kept.
    """
    reports = reports or {}
    ordered = sorted(
        jobs,
        key=lambda j: j.get("created_at") or j.get("updated_at") or "",
    )

    # Biases that already have a successful edit — audits before/after are done.
    remediated: set[str] = set()
    for job in ordered:
        if job.get("status") == "succeeded" and job.get("mode") == "edit":
            remediated.add(job.get("bias_name") or "")

    alerts: list[Alert] = []
    seen_audit_bias: set[str] = set()

    # Walk newest-first so we keep the latest audit alert per bias.
    for job in reversed(ordered):
        report = _parse_report(job, reports)
        for alert in alerts_for_job(
            job,
            report,
            weat_threshold=weat_threshold,
            overcorrection_threshold=overcorrection_threshold,
            min_gap_reduction_pct=min_gap_reduction_pct,
        ):
            if alert.kind == "bias_detected":
                bias = alert.bias_name
                if bias in remediated or bias in seen_audit_bias:
                    continue
                seen_audit_bias.add(bias)
            alerts.append(alert)

    alerts.sort(key=lambda a: a.created_at, reverse=True)
    return alerts
