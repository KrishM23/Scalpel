# Scalpel — Surgical Model Editing & Bias Mitigation Platform

Scalpel is a B2B enterprise API for automated model editing. Point it at **any
compatible Hugging Face model id** — CLIP dual-encoders, causal LLMs (Llama,
Qwen, Mistral, GPT-2, Phi, Gemma, …), or text encoders (BERT, RoBERTa, MPNet).
It uses mechanistic interpretability to **isolate the latent circuit that
transports a bias concept**, and applies **closed-form, low-rank projection
edits** that permanently sever those pathways — no fine-tuning, no gradients,
fully deterministic, with a compliance-grade audit report and quantified
commercial-performance retention.

## Live results (openai/clip-vit-base-patch32, laptop CPU)

| Benchmark | WEAT effect size | Association gap | Retention | Config |
|---|---|---|---|---|
| gender ↔ profession | 1.71 → 1.02 (**−40%**) | **−78%** | 99.0% | k=4 subspace |
| age ↔ competence | 1.63 → 0.34 (**−79%**) | −58% | 99.7% | k=4, calibrated |
| ethnicity ↔ valence | 0.74 → −0.10 (**−86%**) | −86% | 99.98% | calibrated |

Each surgery: ~20 projection edits, 3–15 seconds, gradient-free.

Two findings baked into the product from these runs:

- **Multi-directional erasure matters.** A single linear direction cut the
  gender WEAT by only 16%; erasing a 4-dimensional bias subspace cut it 40%
  at nearly identical retention (k=8 over-erases — the dial is exposed).
- **Full projection can overshoot.** On the ethnicity benchmark, hard erasure
  flipped the association past neutral (WEAT +0.74 → −0.98). *Calibrated
  surgery* sweeps the erasure strength on the isolated circuit and keeps the
  one minimizing residual |WEAT| (0.55 here), reaching −0.10. Overcorrection
  is itself a bias; the platform optimizes for neutrality, not annihilation.

## Product surface

**Free tier (audit)** — measures the bias and shows *exactly which components
cause it* (the teaser), without touching weights.
**Paid tiers (edit)** — apply the surgery, download the edited weights, get
the compliance report. Plans: `free` (audit-only, 10 jobs/mo), `pro` (edits,
100/mo), `enterprise` (unlimited). Quota violations return HTTP 402.

Deliverables per job:

- `report.json` — full machine-readable audit (circuit, every tensor touched,
  metrics before/after, calibration trials).
- `report.html` — standalone printable compliance report for legal/GRC.
- `model.zip` — the edited weights, drop-in `from_pretrained` compatible.
- optional **webhook** on completion for pipeline integration.

## How it works

1. **Contrastive activation capture** — the text tower runs over minimally
   contrastive prompt pairs; at the pooled (EOT) token we record the residual
   stream per layer plus every attention head's and MLP block's write.
2. **Bias subspace estimation** — difference-of-means direction (Marks &
   Tegmark 2023) at the most separable layer (9.5σ for CLIP gender),
   augmented with the top principal components of the paired activation
   differences for multi-dimensional bias (``num_directions``).
3. **Circuit isolation** — every component scored by the norm of its mean
   paired-difference write projected into the subspace; the smallest set
   covering a configurable share of total effect is the bias circuit
   (~10 of ~150 components for CLIP).
4. **Closed-form surgery** — each circuit component's output matrix gets the
   projection `W' = W − α·Vᵀ(VW)`: per-head `out_proj` column slices, MLP
   `fc2` (+ biases), and an optional hardening edit on `text_projection`
   (transported through `final_layer_norm`'s gain). At α=1 the component
   *cannot* write into the subspace for any input; calibration selects α to
   minimize residual |WEAT|. Same projection family as LEACE (Belrose et al.
   2023) and directional ablation (Arditi et al. 2024); rank structure as in
   ROME/MEMIT.
5. **Audit** — WEAT effect size + per-probe association gaps before/after,
   embedding- and geometry-retention on neutral prompts, full edit manifest.

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # 34 tests, fully offline

# Local surgery (recommended flags)
scalpel debias --model openai/clip-vit-base-patch32 \
               --bias gender_profession --num-directions 4 --calibrate --out out/

# Free-tier style audit (no edits)
scalpel debias --model openai/clip-vit-base-patch32 --bias ethnicity_valence --audit
```

## Running the platform

```bash
export SCALPEL_API_KEYS="acme:sk_live_yourkey"
export SCALPEL_TENANT_PLANS="acme:pro"       # optional; default: enterprise
scalpel serve --host 0.0.0.0 --port 8000
```

Web console at `/` (job submission, live status, metrics, report/weights
downloads, plan usage). OpenAPI docs at `/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/edit-jobs` | queue an audit or edit job (mode, options, webhook) |
| `GET /v1/edit-jobs` · `/{id}` | tenant-scoped job listing / status |
| `GET /v1/edit-jobs/{id}/report` | full JSON audit report |
| `GET /v1/edit-jobs/{id}/report.html` | standalone compliance report |
| `GET /v1/edit-jobs/{id}/artifact` | edited weights (zip) |
| `GET /v1/usage` | plan, quota, jobs this month |
| `GET /v1/models` · `/v1/biases` | catalogs |
| `GET /health` | unauthenticated liveness |

Custom bias benchmarks (your own contrastive pairs, probes, retention
prompts) can be submitted inline in the job payload.

```bash
cp .env.example .env && docker compose up   # containerized, persistent /data
```

## Repository layout

```
src/scalpel/
  models/            HF registry + architecture adapters (CLIP / LLM / encoder)
  biases/            benchmark catalog: gender_profession, age_competence,
                     ethnicity_valence + custom specs
  interpretability/  activation capture, bias subspace, circuit isolation
  editing/           low-rank projection edits, calibrated surgeon
  evaluation/        WEAT/association metrics, retention metrics
  pipelines/         end-to-end audit/edit pipeline
  reporting.py       standalone HTML compliance reports
  api/               FastAPI app, tenant auth, plans/quotas, durable jobs,
                     webhooks, web console
tests/               offline suite (tiny random CLIP), 34 tests
```

## Honest limitations & roadmap

- Association benchmarks are prompt-based; image-grounded audits (zero-shot
  parity on face datasets) are the next evaluation tier.
- Current surface is CLIP text towers; vision-tower and decoder-LLM editing
  (ROME-style additive) are registry extensions.
- Calibration optimizes the benchmark's own WEAT; a held-out probe split
  guards against overfitting the metric and ships next.
- Single-node execution; the worker pool swaps for a real queue at scale.
