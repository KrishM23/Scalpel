# Scalpel — Surgical Model Editing & Bias Mitigation Platform

Scalpel is a B2B enterprise API for automated model editing. It pulls foundation
models from Hugging Face, uses mechanistic interpretability to **isolate the
latent circuit that transports a bias concept**, and applies **closed-form,
rank-one weight edits** that permanently sever those pathways — no fine-tuning,
no gradients, fully deterministic, with a compliance-grade audit report and
quantified commercial-performance retention.

## Live results (openai/clip-vit-base-patch32, `gender_profession` benchmark)

| Metric | Before | After | Change |
|---|---|---|---|
| Mean absolute association gap (profession ↔ gender) | 0.0157 | 0.0029 | **−81.6%** |
| Association gap std across professions | 0.0172 | 0.0037 | −78.5% |
| WEAT effect size (Caliskan et al. 2017) | 1.71 | 1.43 | −16%¹ |
| Embedding cosine retention (neutral prompts) | — | 0.9995 | ~unchanged |
| Pairwise-geometry retention | — | 0.9997 | ~unchanged |

Total surgery: 20 rank-one edits across 10 components, 2.8 s on a laptop CPU.

¹ WEAT is a *scale-invariant* effect size: it stays elevated as long as any
residual association ordering survives, even when the association magnitude
(the thing that drives downstream ranking harm) has collapsed by ~80%. Both
numbers are reported so customers see the full picture.

## How it works

1. **Contrastive activation capture** — the text tower is run over minimally
   contrastive prompt pairs ("a photo of a man" / "a photo of a woman" × many
   templates). At the pooled (EOT) token we record the residual stream after
   every layer, each attention head's write (via the `out_proj` column
   decomposition), and each MLP block's write.
2. **Bias direction estimation** — the difference-of-means (diff-in-means)
   estimator over the paired activations yields a unit direction `v` per layer
   (Marks & Tegmark 2023); we select the layer with the largest standardized
   separation (9.5σ at layer 9 for CLIP ViT-B/32).
3. **Circuit isolation** — every component `c` is scored by its attributed
   write onto the bias direction, `|mean_i[(w_c(a_i) − w_c(b_i)) · v]|`.
   Because pairs are minimally contrastive, everything unrelated to the bias
   attribute cancels. The smallest component set covering a configurable share
   of total effect is the isolated **bias circuit**.
4. **Closed-form surgery** — each circuit component's output matrix gets the
   rank-one projection `W' = W − v(vᵀW)`, i.e. `W' = (I − vvᵀ)W`:
   - per attention head: only that head's column slice of `out_proj.weight`,
   - per MLP: `fc2.weight` (+ bias term),
   - optional hardening: `text_projection` is blinded to the direction in its
     input space (transported through `final_layer_norm`'s gain).
   After the edit the component **cannot write any signal along `v` for any
   input** — the pathway is erased permanently, while the orthogonal
   `d_model − 1` dimensional behavior of the component is untouched. This is
   the same closed-form projection family as LEACE (Belrose et al. 2023) and
   directional ablation (Arditi et al. 2024); ROME/MEMIT use the analogous
   rank-one structure for additive edits.
5. **Audit** — WEAT effect size and per-probe association gaps before/after,
   plus embedding- and geometry-retention on bias-neutral prompts. The full
   report (circuit, every tensor touched, metrics, config) is persisted as a
   compliance artifact next to the edited weights.

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                    # 18 tests, fully offline (tiny random CLIP)

# Run a surgery locally
scalpel debias --model openai/clip-vit-base-patch32 \
               --bias gender_profession --out out/
```

## Enterprise API

```bash
export SCALPEL_API_KEYS="acme:sk_live_yourkey"    # tenant:key pairs
scalpel serve --host 0.0.0.0 --port 8000           # OpenAPI docs at /docs
```

```bash
# Queue a surgery
curl -s -X POST localhost:8000/v1/edit-jobs \
  -H "X-API-Key: sk_live_yourkey" -H "Content-Type: application/json" \
  -d '{"model_id": "openai/clip-vit-base-patch32",
       "bias": "gender_profession",
       "options": {"max_components": 12, "harden_projection": true}}'

# Poll status / fetch the audit report
curl -s localhost:8000/v1/edit-jobs/<job_id>        -H "X-API-Key: ..."
curl -s localhost:8000/v1/edit-jobs/<job_id>/report -H "X-API-Key: ..."
```

Endpoints: `GET /v1/models`, `GET /v1/biases`, `POST /v1/edit-jobs`,
`GET /v1/edit-jobs`, `GET /v1/edit-jobs/{id}`, `GET /v1/edit-jobs/{id}/report`,
`GET /health`. Tenants may submit fully custom bias benchmarks (contrastive
pairs + probe sets + retention prompts) in the job payload. Jobs are durable
(SQLite), executed on a bounded worker pool, and tenant-isolated; API keys are
compared in constant time.

```bash
docker build -t scalpel . && docker run -p 8000:8000 -e SCALPEL_API_KEYS=... scalpel
```

## Repository layout

```
src/scalpel/
  models/            model registry + CLIP loading/encoding
  biases/            bias benchmark catalog (BiasSpec)
  interpretability/  activation capture, diff-in-means directions, circuit isolation
  editing/           rank-one projection edits + the surgeon
  evaluation/        WEAT/association bias metrics, retention metrics
  pipelines/         end-to-end debias pipeline + audit report
  api/               FastAPI app, auth, durable job store
  cli.py             scalpel debias | serve | biases
tests/               offline test suite (tiny random CLIP)
```

## Honest limitations & roadmap

- A single linear direction cannot capture non-linear or multi-dimensional
  bias structure (visible in the WEAT residual). Roadmap: multi-directional
  erasure (LEACE-style whitened projections), per-layer directions.
- Current product surface is CLIP text towers; vision-tower and decoder-LLM
  (ROME-style additive) editing are registry extensions, not rewrites.
- Bias benchmarks are prompt-based association tests; image-grounded audits
  (e.g. FairFace zero-shot parity) are the natural next evaluation tier.
- Single-node job execution; swap the worker pool for a real queue before
  multi-tenant scale.
