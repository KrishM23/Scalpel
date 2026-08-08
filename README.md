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
pytest                                   # offline suite (tiny random CLIP)

# Local surgery (recommended flags)
scalpel debias --model openai/clip-vit-base-patch32 \
               --bias gender_profession --num-directions 4 --calibrate --out out/

# Free-tier style audit (no edits)
scalpel debias --model openai/clip-vit-base-patch32 --bias ethnicity_valence --audit
```

## Running the platform

```bash
cp .env.example .env                     # edit keys / signup settings
export SCALPEL_API_KEYS="acme:sk_live_yourkey"   # optional if public signup is on
export SCALPEL_TENANT_PLANS="acme:pro"
scalpel serve --host 0.0.0.0 --port 8000
```

| Path | Surface |
|---|---|
| `/` | Marketing site (product, developers, security, legal) |
| `/signup` · `/login` | Consumer workspace accounts (free audit plan) |
| `/app` | Ops console (jobs, alerts, live demo, reports) |
| `/docs` | OpenAPI |

| Endpoint | Purpose |
|---|---|
| `POST /v1/auth/signup` · `/login` | create / authenticate a workspace |
| `POST /v1/edit-jobs` | queue an audit or edit job (mode, options, webhook) |
| `GET /v1/edit-jobs` · `/{id}` | tenant-scoped job listing / status |
| `GET /v1/edit-jobs/{id}/report` | full JSON audit report |
| `GET /v1/edit-jobs/{id}/report.html` | standalone compliance report |
| `GET /v1/edit-jobs/{id}/artifact` | edited weights (zip) |
| `GET /v1/alerts` | active alerts (failures, unremediated / residual bias) |
| `GET /v1/usage` | plan, quota, jobs this month |
| `GET /v1/models` · `/v1/biases` | catalogs (any compatible HF id accepted) |
| `GET /health` · `/ready` | liveness / readiness (load balancers) |

Custom bias benchmarks (your own contrastive pairs, probes, retention
prompts) can be submitted inline in the job payload.

```bash
cp .env.example .env && docker compose up   # API + Postgres for durable signup
```

### Netlify (marketing site) + API host

Visitors use **one Netlify URL**. The site is static; `/v1/*`, `/app`, and `/r/*`
are proxied to your API so live surgery still runs for real.

1. Deploy `scalpel serve` (Railway, Fly, Render, GPU box, or `docker compose`)
   with `DATABASE_URL=postgresql://…`, `SCALPEL_PUBLIC_DEMO=1`, and
   `pip install "scalpel-ai[postgres]"`.
2. In Netlify → Site settings → Environment variables, set
   `SCALPEL_API_ORIGIN=https://your-api-host` (no trailing slash).
3. Connect the GitHub repo. Build command/publish dir come from `netlify.toml`
   (`python3 scripts/build_netlify.py` → `netlify-dist/`). The build is
   **stdlib-only** (no torch/fastapi install on Netlify).
4. Trigger a redeploy. Open the Netlify URL — `/` should load the landing page;
   live demo hits `/v1/public/demo-jobs` via the proxy.

If Netlify shows **Page not found**, the deploy likely failed or publish dir
is wrong — check Deploy log for `Wrote .../netlify-dist` and `index.html bytes=`.

```bash
SCALPEL_API_ORIGIN=https://api.yourdomain.com python3 scripts/build_netlify.py
# publish netlify-dist/ (or connect the repo; netlify.toml runs the build)
```

## Repository layout

```
src/scalpel/
  models/            HF registry + architecture adapters (CLIP / LLM / encoder)
  biases/            benchmark catalog: gender_profession, ad_gender_product,
                     ad_age_luxury, ad_ethnicity_brand, age_competence,
                     ethnicity_valence, religion_valence, disability_competence
                     + custom specs
  interpretability/  activation capture, bias subspace, circuit isolation
  editing/           low-rank projection edits, calibrated surgeon
  evaluation/        WEAT/association metrics, retention metrics
  pipelines/         end-to-end audit/edit pipeline
  reporting.py       standalone HTML compliance reports
  api/               FastAPI app, marketing site, signup, plans/quotas,
                     durable jobs, webhooks, ops console
tests/               offline suite (tiny random CLIP)
```

## Honest limitations & roadmap

- Association benchmarks are prompt-based; image-grounded audits (zero-shot
  parity on face datasets) are the next evaluation tier.
- Architecture adapters cover CLIP, causal LMs, and text encoders; vision-tower
  and ROME-style additive LLM edits are next registry extensions.
- Calibration optimizes the benchmark's own WEAT; a held-out probe split
  guards against overfitting the metric and ships next.
- Single-node execution; the worker pool swaps for a real queue at scale.
- Billing upgrades (free → Pro) are sales-assisted today; self-serve checkout
  is on the roadmap.
