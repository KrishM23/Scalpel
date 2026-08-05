"""Standalone HTML compliance report generation.

Renders a pipeline audit report (the JSON produced by ``run_debias_pipeline``)
into a self-contained HTML document — no external assets, printable, suitable
for forwarding to legal/compliance stakeholders.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

_CSS = """
  body { font: 14px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif;
         color: #1a2233; max-width: 860px; margin: 0 auto; padding: 40px 24px; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  h2 { font-size: 16px; margin-top: 32px; border-bottom: 2px solid #e3e8f0;
       padding-bottom: 6px; }
  .sub { color: #64748b; font-size: 13px; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
          margin-top: 16px; }
  .card { border: 1px solid #e3e8f0; border-radius: 8px; padding: 14px; }
  .card .v { font-size: 22px; font-weight: 700; }
  .card .l { font-size: 12px; color: #64748b; }
  .good { color: #059669; } .bad { color: #dc2626; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eef1f6; }
  th { color: #64748b; font-weight: 600; }
  .bar { height: 10px; background: #eef1f6; border-radius: 5px; overflow: hidden; }
  .bar > div { height: 100%; background: #5b8cff; }
  .footer { margin-top: 40px; font-size: 12px; color: #94a3b8;
            border-top: 1px solid #e3e8f0; padding-top: 12px; }
"""


def _card(value: str, label: str, cls: str = "") -> str:
    return (
        f'<div class="card"><div class="v {cls}">{value}</div>'
        f'<div class="l">{html.escape(label)}</div></div>'
    )


def _reduction_card(name: str, entry: dict) -> str:
    pct = entry["reduction_pct"]
    cls = "good" if pct >= 0 else "bad"
    sign = "−" if pct >= 0 else "+"
    return _card(
        f"{sign}{abs(pct):.1f}%",
        f"{name} ({entry['before']} → {entry['after']})",
        cls,
    )


def render_report_html(report: dict) -> str:
    spec = report["bias_spec"]
    metrics = report["metrics"]
    circuit = report["circuit"]
    mode = report.get("mode", "edit")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []

    # --- Summary cards -----------------------------------------------------
    cards: list[str] = []
    if mode == "edit" and "bias_reduction" in metrics:
        reduction = metrics["bias_reduction"]
        cards.append(
            _reduction_card("association gap", reduction["mean_abs_association_gap"])
        )
        cards.append(_reduction_card("WEAT effect size", reduction["weat_effect_size"]))
        retention = metrics["retention"]
        cards.append(
            _card(
                f"{retention['embedding_cosine_retention'] * 100:.2f}%",
                "embedding retention on neutral prompts",
                "good",
            )
        )
        cards.append(_card(str(report["surgery"]["num_edits"]), "projection edits applied"))
    else:
        before = metrics["bias_before"]
        cards.append(_card(f"{before['weat_effect_size']}", "WEAT effect size (measured)"))
        cards.append(
            _card(f"{before['mean_abs_association_gap']}", "mean absolute association gap")
        )
        cards.append(
            _card(str(len(circuit["selected_components"])), "components carrying the bias")
        )
    cards.append(_card(f"{report['runtime_seconds']}s", "pipeline runtime"))
    cards.append(_card(str(circuit.get("num_directions", 1)), "erased subspace dimension"))
    sections.append(f'<h2>Summary</h2><div class="grid">{"".join(cards)}</div>')

    # --- Circuit table -----------------------------------------------------
    rows = "".join(
        f"<tr><td>{html.escape(c['label'])}</td><td>{c['kind']}</td>"
        f"<td>{c['score']:.4f}</td>"
        f'<td><div class="bar"><div style="width:{min(100, c["share"] * 400):.0f}%">'
        f"</div></div></td><td>{c['share'] * 100:.1f}%</td></tr>"
        for c in circuit["selected_components"]
    )
    sections.append(
        "<h2>Isolated bias circuit</h2>"
        f'<p class="sub">Bias subspace estimated at residual stream layer '
        f"{circuit['direction_layer']} (separation {circuit['direction_separation']}σ). "
        "Components ranked by attributed write onto the bias subspace.</p>"
        f"<table><tr><th>Component</th><th>Kind</th><th>Score</th>"
        f"<th>Share of effect</th><th></th></tr>{rows}</table>"
    )

    # --- Per-probe associations --------------------------------------------
    before_gaps = metrics["bias_before"]["per_probe_gaps"]
    after_gaps = metrics.get("bias_after", {}).get("per_probe_gaps")
    probe_rows = "".join(
        f"<tr><td>{html.escape(probe)}</td><td>{gap:+.4f}</td>"
        + (f"<td>{after_gaps[probe]:+.4f}</td>" if after_gaps else "")
        + "</tr>"
        for probe, gap in before_gaps.items()
    )
    after_header = "<th>After</th>" if after_gaps else ""
    sections.append(
        "<h2>Per-probe association gaps</h2>"
        f'<p class="sub">Cosine-similarity gap toward the '
        f"{html.escape(spec['groups'][0])} vs {html.escape(spec['groups'][1])} "
        "attribute centroid, per probe prompt.</p>"
        f"<table><tr><th>Probe</th><th>Before</th>{after_header}</tr>{probe_rows}</table>"
    )

    # --- Edit manifest -----------------------------------------------------
    if mode == "edit" and "surgery" in report:
        edit_rows = "".join(
            f"<tr><td><code>{html.escape(e['target'])}</code></td>"
            f"<td>{html.escape(e['edit'])}</td><td>{e['rank']}</td></tr>"
            for e in report["surgery"]["edits"]
        )
        sections.append(
            "<h2>Edit manifest</h2>"
            '<p class="sub">Every parameter tensor modified, with the projection '
            "type and rank. All edits are closed-form and deterministic.</p>"
            f"<table><tr><th>Parameter</th><th>Edit</th><th>Rank</th></tr>{edit_rows}</table>"
        )

    title = "Bias Surgery Report" if mode == "edit" else "Bias Audit Report"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Scalpel — {title}</title>
<style>{_CSS}</style></head><body>
<h1>Scalpel — {title}</h1>
<p class="sub">
  Model: <strong>{html.escape(report["model_id"])}</strong> ·
  Benchmark: <strong>{html.escape(spec["name"])}</strong>
  ({html.escape(spec["groups"][0])} vs {html.escape(spec["groups"][1])},
  {spec["num_contrastive_pairs"]} contrastive pairs) ·
  Generated {generated}
</p>
{"".join(sections)}
<div class="footer">
  Generated by Scalpel v0.1 · pipeline {html.escape(report["pipeline"])} ·
  method: difference-of-means bias subspace estimation, mechanistic circuit
  attribution, closed-form orthogonal-projection weight edits.
</div>
</body></html>"""
