/**
 * Guided surgery walkthrough — demo-smooth visuals, live metrics when ready.
 * Step clicks only change the scene (never restart the job).
 */
(function () {
  function $(id) { return document.getElementById(id); }
  if (!$("liveDemo")) return;

  const apiBase = () => (window.SCALPEL_API_BASE || "").replace(/\/$/, "");
  const api = (path) => `${apiBase()}${path}`;
  const PHASES = ["measure", "locate", "cut", "prove"];

  const PREVIEW = {
    probes: [
      { name: "power drill", before: 0.018, after: 0.003 },
      { name: "makeup palette", before: -0.016, after: -0.002 },
      { name: "sports car", before: 0.014, after: 0.002 },
    ],
    weatBefore: 1.71,
    weatAfter: 1.02,
    gapPct: 78,
    retention: 99.0,
    edits: 20,
    hot: [5, 11, 14, 22, 27, 33, 38, 41],
  };

  const COPY = {
    measure: {
      title: "01 · Measure the bias",
      meta: "Contrastive prompts · WEAT effect size",
      caption: "Ask the model the same questions about different groups. The association gap shows how hard it leans.",
      status: "Measuring stereotyped associations…",
      log: ["> measure: 48 contrastive pairs", "> computing WEAT on the text tower…"],
    },
    locate: {
      title: "02 · Locate the circuit",
      meta: "Attribution across attention heads + MLPs",
      caption: "Not the whole network — only the components that write the bias into the residual stream.",
      status: "Isolating the bias circuit…",
      log: ["> locate: diff-in-means direction", "> ranking component writes onto the bias subspace…"],
    },
    cut: {
      title: "03 · Cut it out of the weights",
      meta: "Closed-form rank-k projections · W′ = W − α Vᵀ(VW)",
      caption: "A surgical edit removes that direction. No fine-tuning loop — deterministic projections on the hot components.",
      status: "Applying closed-form surgery…",
      log: ["> cut: project bias subspace out of selected weights", "> calibrate α so capability stays intact…"],
    },
    prove: {
      title: "04 · Prove it worked",
      meta: "Re-measure WEAT · retention on neutral prompts",
      caption: "Bias down, capability retained. Export the report, recipe, or edited weights.",
      status: "Verifying bias ↓ and retention…",
      log: ["> prove: re-run WEAT + retention probes", "> packing compliance artifacts…"],
    },
  };

  const LD = {
    cells: 48,
    hot: PREVIEW.hot,
    cellTimers: [],   // per-phase board animations — safe to clear on each scene
    storyTimers: [],  // autoplay advance — must NOT clear when entering a phase
    pollTimer: null,
    gen: 0,
    phase: "measure",
    autoplay: true,
    report: null,
    live: false,
    jobId: null,
    pollFails: 0,
    shareUrl: null,
    pdfUrl: null,
    recipeUrl: null,
    artifactUrl: null,
    modelId: null,
  };

  function ldClearCellTimers() {
    LD.cellTimers.forEach(clearTimeout);
    LD.cellTimers = [];
  }
  function ldClearStoryTimers() {
    LD.storyTimers.forEach(clearTimeout);
    LD.storyTimers = [];
  }
  function ldClearPoll() {
    if (LD.pollTimer) {
      clearTimeout(LD.pollTimer);
      LD.pollTimer = null;
    }
  }
  function ldClearAllTimers() {
    ldClearCellTimers();
    ldClearStoryTimers();
    ldClearPoll();
  }
  /** Board micro-animations for the current phase only. */
  function ldCellLater(ms, fn) {
    const t = setTimeout(fn, ms);
    LD.cellTimers.push(t);
    return t;
  }
  /** Autplay / story schedule — survives showPhase(). */
  function ldStoryLater(ms, fn) {
    const t = setTimeout(fn, ms);
    LD.storyTimers.push(t);
    return t;
  }

  function selectedBias() {
    return ($("ldBias") && $("ldBias").value) || "global_language_prestige";
  }
  function selectedModel() {
    const custom = ($("ldModelInput") && $("ldModelInput").value || "").trim();
    if (custom) return custom;
    return ($("ldModelSelect") && $("ldModelSelect").value) || "openai/clip-vit-base-patch32";
  }

  function metricsFromReport(report) {
    if (!report || !report.metrics) return null;
    const m = report.metrics;
    const before = m.bias_before || {};
    const after = m.bias_after || {};
    const red = m.bias_reduction || {};
    const gapsB = before.per_probe_gaps || {};
    const gapsA = after.per_probe_gaps || {};
    const keys = Object.keys(gapsB).slice(0, 3);
    const probes = (keys.length ? keys : PREVIEW.probes.map((p) => p.name)).map((key, i) => {
      if (keys.length) {
        const short = key.replace(/^an ad for a /i, "").replace(/^a photo of (a )?/i, "");
        return {
          name: short.length > 18 ? short.slice(0, 16) + "…" : short,
          before: gapsB[key] || 0,
          after: gapsA[key] != null ? gapsA[key] : 0,
        };
      }
      return PREVIEW.probes[i];
    });
    return {
      probes,
      weatBefore: red.weat_effect_size?.before ?? before.weat_effect_size ?? PREVIEW.weatBefore,
      weatAfter: red.weat_effect_size?.after ?? after.weat_effect_size ?? PREVIEW.weatAfter,
      gapPct: red.mean_abs_association_gap?.reduction_pct ?? PREVIEW.gapPct,
      retention: (m.retention?.embedding_cosine_retention ?? PREVIEW.retention / 100) * 100,
      edits: report.surgery?.num_edits ?? PREVIEW.edits,
      hot: PREVIEW.hot,
    };
  }

  function data() {
    return metricsFromReport(LD.report) || PREVIEW;
  }

  function ldBuildBoard() {
    const grid = $("ldGrid");
    if (!grid) return;
    grid.innerHTML = Array.from({ length: LD.cells }, (_, i) =>
      `<div class="ld-cell" data-i="${i}"></div>`).join("");
  }

  function setCells(mode) {
    const cells = [...($("ldGrid")?.children || [])];
    cells.forEach((c) => c.classList.remove("scan", "hot", "cut", "fade", "idle"));
    if (mode === "scan") {
      // one-shot highlight row, no infinite glide
      for (let i = 0; i < 12; i++) cells[i]?.classList.add("scan");
    } else if (mode === "hot") {
      LD.hot.forEach((idx) => cells[idx]?.classList.add("hot"));
    } else if (mode === "cut") {
      LD.hot.forEach((idx) => cells[idx]?.classList.add("cut"));
    } else if (mode === "done") {
      LD.hot.forEach((idx) => cells[idx]?.classList.add("cut", "fade"));
    }
  }

  function renderProbes(phase) {
    const d = data();
    const el = $("ldProbes");
    if (!el) return;
    el.innerHTML = d.probes.map((p, i) => {
      const useAfter = phase === "prove" || phase === "cut";
      const val = useAfter && phase === "prove" ? p.after : p.before;
      const pct = Math.min(100, Math.abs(val) / 0.02 * 100);
      const width = phase === "locate" || phase === "measure"
        ? Math.max(12, pct)
        : phase === "cut"
          ? Math.max(12, pct * 0.55)
          : Math.max(8, Math.abs(p.after) / 0.02 * 100 * 0.4);
      const neutral = phase === "prove" ? " neutral" : "";
      const shown = phase === "prove" ? p.after : p.before;
      return `<div class="ld-probe">
        <span>${p.name}</span>
        <div class="track"><div class="fill${neutral}" style="width:${width}%"></div></div>
        <span>${shown >= 0 ? "+" : ""}${Number(shown).toFixed(3)}</span>
      </div>`;
    }).join("");
  }

  function renderMetrics(phase) {
    const d = data();
    const weat = $("ldWeat");
    const weatSub = $("ldWeatSub");
    const gap = $("ldGap");
    const gapSub = $("ldGapSub");
    const ret = $("ldRet");
    const retSub = $("ldRetSub");
    if (!weat) return;

    weat.classList.remove("good");
    gap.classList.remove("good");
    ret.classList.remove("good");

    if (phase === "measure" || phase === "locate") {
      weat.textContent = Number(d.weatBefore).toFixed(2);
      weatSub.textContent = "baseline — stereotyped association";
      gap.textContent = "high";
      gapSub.textContent = "mean |Δ cosine| elevated";
      ret.textContent = "—";
      retSub.textContent = "measured after the cut";
    } else if (phase === "cut") {
      weat.textContent = Number(d.weatBefore).toFixed(2);
      weatSub.textContent = "editing weights now…";
      gap.textContent = "…";
      gapSub.textContent = "projecting bias subspace out";
      ret.textContent = "…";
      retSub.textContent = "calibrating retention";
    } else {
      weat.textContent = `${Number(d.weatBefore).toFixed(2)} → ${Number(d.weatAfter).toFixed(2)}`;
      weat.classList.add("good");
      weatSub.textContent = LD.live ? "live surgery result" : "preview · live numbers when job finishes";
      gap.textContent = `−${Number(d.gapPct).toFixed(0)}%`;
      gap.classList.add("good");
      gapSub.textContent = "association gap reduction";
      ret.textContent = `${Number(d.retention).toFixed(1)}%`;
      ret.classList.add("good");
      retSub.textContent = "neutral prompt fidelity";
    }
  }

  function setPhaseUI(phase) {
    document.querySelectorAll("#ldRail .ld-phase").forEach((el) => {
      const p = el.dataset.phase;
      const active = p === phase;
      const done = PHASES.indexOf(p) < PHASES.indexOf(phase);
      el.classList.toggle("active", active);
      el.classList.toggle("done", done);
      el.setAttribute("aria-selected", active ? "true" : "false");
    });
    const board = $("ldBoard");
    if (board) board.dataset.phase = phase;
    const liveDemo = $("liveDemo");
    if (liveDemo) liveDemo.dataset.phase = phase;
  }

  function showPhase(phase, { animate } = { animate: true }) {
    if (!PHASES.includes(phase)) return;
    ldClearCellTimers();
    LD.phase = phase;
    const copy = COPY[phase];
    const d = data();
    setPhaseUI(phase);

    const title = $("ldBoardTitle");
    const meta = $("ldBoardMeta");
    const caption = $("ldCaption");
    const status = $("ldStatus");
    const log = $("ldLog");
    const bar = $("ldBar");
    const badge = $("ldLiveBadge");

    if (title) title.textContent = copy.title;
    if (meta) {
      meta.textContent = phase === "prove"
        ? `${d.edits} edits · ${LD.live ? "live results" : "preview metrics"}`
        : copy.meta;
    }
    if (caption) caption.textContent = copy.caption;
    if (status) status.innerHTML = copy.status;
    if (log) {
      log.innerHTML = copy.log.map((l) => `<div class="line">${l}</div>`).join("");
    }
    if (bar) {
      bar.style.width = ({ measure: "25%", locate: "50%", cut: "75%", prove: "100%" })[phase];
    }
    if (badge) {
      badge.hidden = !LD.live;
      badge.textContent = LD.live ? "LIVE RESULTS" : "";
    }

    ldBuildBoard();
    if (phase === "measure") setCells("scan");
    else if (phase === "locate") {
      setCells("");
      if (animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        LD.hot.forEach((idx, n) => {
          ldCellLater(n * 90, () => $("ldGrid")?.children[idx]?.classList.add("hot"));
        });
      } else setCells("hot");
    } else if (phase === "cut") {
      setCells("hot");
      if (animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        LD.hot.forEach((idx, n) => {
          ldCellLater(120 + n * 80, () => {
            const cell = $("ldGrid")?.children[idx];
            if (!cell) return;
            cell.classList.remove("hot");
            cell.classList.add("cut");
            ldCellLater(200, () => cell.classList.add("fade"));
          });
        });
      } else setCells("done");
    } else {
      setCells("done");
    }

    renderProbes(phase);
    renderMetrics(phase);
    updateExports(phase === "prove");
  }

  function updateExports(show) {
    const share = $("ldShare");
    if (!share) return;
    if (!show || !LD.live) {
      share.hidden = true;
      return;
    }
    share.hidden = false;
    if ($("ldShareHtml") && LD.shareUrl) $("ldShareHtml").href = api(LD.shareUrl);
    if ($("ldSharePdf") && LD.pdfUrl) $("ldSharePdf").href = api(LD.pdfUrl);
    if ($("ldShareRecipe") && LD.recipeUrl) $("ldShareRecipe").href = api(LD.recipeUrl);
    const art = $("ldArtifact");
    if (art) {
      if (LD.artifactUrl) {
        art.hidden = false;
        art.href = api(LD.artifactUrl);
        art.textContent = "Download weights";
        art.onclick = null;
      } else {
        art.hidden = false;
        art.href = "#";
        art.textContent = "Export weights";
        art.onclick = (e) => {
          e.preventDefault();
          startJob(true, true);
        };
      }
    }
  }

  // Dwell time on each step before auto-advancing (ms).
  const DWELL = { measure: 2400, locate: 2600, cut: 2800, prove: 3200 };
  const CLICK_PAUSE_MS = 4500; // after a manual click, hold then resume autoplay

  function playAutoplay(gen, startAt) {
    LD.autoplay = true;
    ldClearStoryTimers();
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let i = Math.max(0, PHASES.indexOf(startAt || "measure"));
    if (i < 0) i = 0;

    function advance(idx) {
      if (gen !== LD.gen || !LD.autoplay) return;
      const phase = PHASES[idx];
      showPhase(phase, { animate: !reduce });
      const st = $("ldStatus");
      if (st) {
        if (LD.live && phase === "prove") {
          const d = data();
          st.innerHTML =
            `Surgery complete — <b>−${Number(d.gapPct).toFixed(0)}% gap</b>, ` +
            `<b>${Number(d.retention).toFixed(1)}% retention</b>` +
            ` <span class="ld-muted">· autoplaying</span>`;
        } else {
          st.innerHTML = `${COPY[phase].status} <span class="ld-muted">· autoplaying · click to jump</span>`;
        }
      }
      const wait = reduce ? 0 : DWELL[phase];
      const next = (idx + 1) % PHASES.length; // loop forever
      ldStoryLater(wait, () => advance(next));
    }
    advance(i);
  }

  function jumpLiveDemo(phase) {
    // Jump immediately, pause the loop briefly, then resume autoplay from here.
    if (!PHASES.includes(phase)) return;
    const gen = LD.gen;
    LD.autoplay = false;
    ldClearStoryTimers();
    ldClearCellTimers();
    showPhase(phase, { animate: true });
    const st = $("ldStatus");
    if (st) {
      st.innerHTML = LD.live
        ? `${COPY[phase].status} <span class="ld-muted">(paused · resumes soon)</span>`
        : `${COPY[phase].status} <span class="ld-muted">(paused · autoplay resumes)</span>`;
    }
    ldStoryLater(CLICK_PAUSE_MS, () => {
      if (gen !== LD.gen) return;
      // Resume from the next step so we don't re-show the one they just clicked.
      const next = PHASES[(PHASES.indexOf(phase) + 1) % PHASES.length];
      playAutoplay(gen, next);
    });
  }

  function applyLiveResult(body) {
    if (!body?.report) return;
    LD.report = body.report;
    LD.live = true;
    LD.shareUrl = body.share_url;
    LD.pdfUrl = body.pdf_url;
    LD.recipeUrl = body.recipe_url;
    LD.artifactUrl = body.artifact_url;
    LD.modelId = body.model_id;
    const label = $("ldModelLabel");
    if (label) label.textContent = `${body.model_id} · live`;
    const signup = $("ldSignup");
    if (signup) {
      signup.href = `/signup?${new URLSearchParams({
        model: body.model_id || selectedModel(),
        bias: selectedBias(),
      })}`;
    }
    // Refresh current scene with real numbers (don't yank the user mid-story).
    showPhase(LD.phase || "prove", { animate: false });
    if (LD.phase === "prove") {
      const d = data();
      const st = $("ldStatus");
      if (st) {
        st.innerHTML =
          `Live surgery ready — <b>−${Number(d.gapPct).toFixed(0)}% gap</b>, ` +
          `<b>${Number(d.retention).toFixed(1)}% retention</b>`;
      }
    }
  }

  async function pollJob(jobId) {
    try {
      const res = await fetch(api(`/v1/public/demo-jobs/${jobId}`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      LD.pollFails = 0;
      const body = await res.json();
      if (body.status === "queued" || body.status === "running") {
        LD.pollTimer = setTimeout(() => pollJob(jobId), 2000);
        return;
      }
      if (body.status === "succeeded" && body.report) {
        applyLiveResult(body);
        return;
      }
      if (body.status === "failed") {
        const label = $("ldModelLabel");
        if (label) label.textContent = `${body.model_id || "model"} · preview (job failed)`;
      }
    } catch (_) {
      LD.pollFails += 1;
      // 502 / proxy blips: quiet backoff, never interrupt the walkthrough UI.
      if (LD.pollFails < 6) {
        LD.pollTimer = setTimeout(() => pollJob(jobId), 3000 * LD.pollFails);
      } else {
        const label = $("ldModelLabel");
        if (label && !LD.live) label.textContent = `${selectedModel()} · preview only`;
      }
    }
  }

  async function startJob(force, exportWeights) {
    ldClearPoll();
    LD.pollFails = 0;
    const bias = selectedBias();
    const modelId = selectedModel();
    const label = $("ldModelLabel");
    if (label) label.textContent = `${modelId} · connecting…`;
    try {
      const res = await fetch(api("/v1/public/demo-jobs"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bias,
          model_id: modelId,
          force: !!force || !!exportWeights,
          export_weights: !!exportWeights,
        }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
      }
      LD.jobId = body.id;
      if (label) label.textContent = `${body.model_id} · job ${body.status}`;
      if (body.status === "succeeded" && body.report) {
        applyLiveResult(body);
        return;
      }
      pollJob(body.id);
    } catch (_) {
      // API/proxy down (common Netlify 502 if SCALPEL_API_ORIGIN unset) — keep demo smooth.
      if (label) label.textContent = `${modelId} · preview only`;
      LD.jobId = null;
    }
  }

  function restartLiveDemo() {
    ldClearAllTimers();
    LD.gen += 1;
    const gen = LD.gen;
    LD.report = null;
    LD.live = false;
    LD.jobId = null;
    LD.pollFails = 0;
    LD.shareUrl = LD.pdfUrl = LD.recipeUrl = LD.artifactUrl = null;
    LD.autoplay = true;
    const share = $("ldShare");
    if (share) share.hidden = true;
    playAutoplay(gen);
    startJob(true, false);
  }

  function boot() {
    ldBuildBoard();
    LD.gen += 1;
    const gen = LD.gen;
    playAutoplay(gen);
    startJob(false, false);
  }

  window.jumpLiveDemo = jumpLiveDemo;
  window.restartLiveDemo = restartLiveDemo;
  window.startLiveDemo = restartLiveDemo;

  async function loadCatalogs() {
    try {
      const [biasesRes, modelsRes] = await Promise.all([
        fetch(api("/v1/public/demo-biases")),
        fetch(api("/v1/public/demo-models")),
      ]);
      if (biasesRes.ok && $("ldBias")) {
        const items = await biasesRes.json();
        if (Array.isArray(items) && items.length) {
          const cur = $("ldBias").value;
          $("ldBias").innerHTML = items.map((b) =>
            `<option value="${b.name}">${b.name} — ${b.groups.join(" ↔ ")}</option>`
          ).join("");
          if ([...$("ldBias").options].some((o) => o.value === cur)) $("ldBias").value = cur;
          else if ([...$("ldBias").options].some((o) => o.value === "global_language_prestige")) {
            $("ldBias").value = "global_language_prestige";
          }
        }
      }
      if (modelsRes.ok && $("ldModelSelect")) {
        const dataM = await modelsRes.json();
        const featured = dataM.featured || [];
        const def = dataM.default_model_id || "openai/clip-vit-base-patch32";
        $("ldModelSelect").innerHTML = featured.map((m) =>
          `<option value="${m.model_id}">${m.model_id}</option>`
        ).join("");
        if (![...$("ldModelSelect").options].some((o) => o.value === def)) {
          $("ldModelSelect").insertAdjacentHTML("afterbegin", `<option value="${def}">${def}</option>`);
        }
        $("ldModelSelect").value = def;
      }
    } catch (_) { /* static fallbacks */ }
  }

  loadCatalogs().finally(() => {
    $("ldBias")?.addEventListener("change", () => restartLiveDemo());
    $("ldModelSelect")?.addEventListener("change", () => {
      if ($("ldModelInput")) $("ldModelInput").value = "";
      restartLiveDemo();
    });
    $("ldModelInput")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        restartLiveDemo();
      }
    });
    const root = $("liveDemo");
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          io.disconnect();
          boot();
        }
      }, { threshold: 0.2 });
      io.observe(root);
    } else {
      boot();
    }
  });
})();
