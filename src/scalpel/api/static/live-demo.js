/** Live Measure → Locate → Cut → Prove demo backed by POST /v1/public/demo-jobs. */
(function () {
  function $(id) { return document.getElementById(id); }
  if (!$("liveDemo")) return;

  const apiBase = () => (window.SCALPEL_API_BASE || "").replace(/\/$/, "");
  const api = (path) => `${apiBase()}${path}`;

  const LD = {
    cells: 48,
    hot: [5, 11, 14, 22, 27, 33, 38, 41],
    timers: [],
    running: false,
    gen: 0,
    phase: null,
    jobId: null,
    pollTimer: null,
    report: null,
    shareUrl: null,
    pdfUrl: null,
    recipeUrl: null,
    artifactUrl: null,
    reproduceCurl: null,
    pendingResult: null,
    storyDone: false,
  };

  function ldClearTimers() {
    LD.timers.forEach(clearTimeout);
    LD.timers = [];
    if (LD.pollTimer) {
      clearTimeout(LD.pollTimer);
      LD.pollTimer = null;
    }
  }
  function ldLater(ms, fn) {
    const t = setTimeout(fn, ms);
    LD.timers.push(t);
    return t;
  }
  function ldAlive(gen) { return () => gen === LD.gen; }

  function ldLog(lines) {
    $("ldLog").innerHTML = lines.map((l) => `<div class="line">${l}</div>`).join("");
  }
  function ldSetPhase(name) {
    LD.phase = name;
    const order = ["measure", "locate", "cut", "prove"];
    document.querySelectorAll("#ldRail .ld-phase").forEach((el) => {
      const p = el.dataset.phase;
      const active = p === name;
      el.classList.toggle("active", active);
      el.classList.toggle("done", order.indexOf(p) < order.indexOf(name));
      el.setAttribute("aria-selected", active ? "true" : "false");
    });
  }
  function ldProgress(pct) {
    const bar = $("ldBar");
    if (bar) bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  }

  function selectedBias() {
    const sel = $("ldBias");
    return (sel && sel.value) || "ad_gender_product";
  }

  function selectedModel() {
    const custom = ($("ldModelInput")?.value || "").trim();
    if (custom) return custom;
    const sel = $("ldModelSelect");
    return (sel && sel.value) || "openai/clip-vit-base-patch32";
  }

  function ldBuildBoard() {
    $("ldGrid").innerHTML = Array.from({ length: LD.cells }, (_, i) =>
      `<div class="ld-cell" data-i="${i}"></div>`).join("");
  }

  function probeNamesFromReport(report) {
    const gaps = report?.metrics?.bias_before?.per_probe_gaps || {};
    const keys = Object.keys(gaps);
    if (keys.length) {
      return keys.slice(0, 3).map((k) => {
        const short = k.replace(/^an ad for a /i, "").replace(/^a photo of (a )?/i, "");
        return short.length > 18 ? short.slice(0, 16) + "…" : short;
      });
    }
    return ["probe A", "probe B", "probe C"];
  }

  function ldBuildProbes(names) {
    $("ldProbes").innerHTML = names.map((name, i) =>
      `<div class="ld-probe" data-p="${i}">
         <span>${name}</span>
         <div class="track"><div class="fill" id="ldFill${i}"></div></div>
         <span id="ldProbeVal${i}">—</span>
       </div>`).join("");
  }

  function hideExports() {
    const share = $("ldShare");
    if (share) share.hidden = true;
    const art = $("ldArtifact");
    if (art) art.hidden = true;
  }

  function ldResetVisuals() {
    ldBuildBoard();
    ldBuildProbes(["—", "—", "—"]);
    $("ldWeat").textContent = "—";
    $("ldWeat").classList.remove("good");
    $("ldWeatSub").textContent = "awaiting measurement";
    $("ldGap").textContent = "—";
    $("ldGap").classList.remove("good");
    $("ldGapSub").textContent = "mean |Δ cosine|";
    $("ldRet").textContent = "—";
    $("ldRet").classList.remove("good");
    $("ldRetSub").textContent = "neutral prompt fidelity";
    $("ldBoardTitle").textContent = "Model components";
    $("ldBoardMeta").textContent = "waiting for job…";
    $("ldStatus").innerHTML = "Starting <b>live surgery</b>…";
    ldProgress(0);
    document.querySelectorAll("#ldRail .ld-phase").forEach((el) =>
      el.classList.remove("active", "done"));
    hideExports();
    ldLog(["> scalpel — same pipeline on any Hugging Face model"]);
  }

  function ldClearCells() {
    [...$("ldGrid").children].forEach((c) =>
      c.classList.remove("scan", "hot", "cut", "fade"));
  }

  function ldShowWorkingPhase(phase, meta) {
    ldClearCells();
    ldSetPhase(phase);
    if (phase === "measure") {
      $("ldBoardTitle").textContent = "Measuring association bias";
      $("ldBoardMeta").textContent = meta || "contrastive prompts · WEAT";
      $("ldStatus").innerHTML = "Measuring bias across <b>probes</b>…";
      ldProgress(22);
      let scan = 0;
      const gen = LD.gen;
      const step = () => {
        if (gen !== LD.gen) return;
        const cells = [...$("ldGrid").children];
        cells.forEach((c) => c.classList.remove("scan"));
        for (let k = 0; k < 6; k++) cells[(scan + k) % cells.length]?.classList.add("scan");
        scan += 3;
        ldLater(90, step);
      };
      step();
    } else if (phase === "locate") {
      $("ldBoardTitle").textContent = "Isolating the bias circuit";
      $("ldBoardMeta").textContent = meta || "attribution across heads + MLPs";
      $("ldStatus").innerHTML = "Localizing which components <b>write the bias</b>…";
      ldProgress(48);
      LD.hot.forEach((idx, n) => {
        ldLater(n * 80, () => $("ldGrid").children[idx]?.classList.add("hot"));
      });
    } else if (phase === "cut") {
      $("ldBoardTitle").textContent = "Applying closed-form surgery";
      $("ldBoardMeta").textContent = meta || "W′ = W − α Vᵀ(VW)";
      $("ldStatus").innerHTML = "Severing circuit with <b>rank-k projections</b>…";
      ldProgress(72);
      LD.hot.forEach((idx, n) => {
        ldLater(n * 70, () => {
          const cell = $("ldGrid").children[idx];
          if (!cell) return;
          cell.classList.add("cut");
          ldLater(160, () => cell.classList.add("fade"));
        });
      });
    }
  }

  /** Always walk 01→04 so cached jobs don't skip straight to prove. */
  function playPhaseStory(gen) {
    const alive = ldAlive(gen);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    LD.storyDone = false;
    if (reduce) {
      LD.storyDone = true;
      maybeFinish(gen);
      return;
    }
    ldShowWorkingPhase("measure");
    ldLog(["> 01 measure — contrastive WEAT on the selected model"]);
    ldLater(1600, () => {
      if (!alive()) return;
      ldShowWorkingPhase("locate");
      ldLog(["> 02 locate — attributing bias circuit components"]);
    });
    ldLater(1600 + 1800, () => {
      if (!alive()) return;
      ldShowWorkingPhase("cut");
      ldLog(["> 03 cut — closed-form rank-k projection edits"]);
    });
    ldLater(1600 + 1800 + 2000, () => {
      if (!alive()) return;
      LD.storyDone = true;
      maybeFinish(gen);
    });
  }

  function maybeFinish(gen) {
    if (gen !== LD.gen) return;
    if (!LD.storyDone || !LD.pendingResult) return;
    const body = LD.pendingResult;
    applyReport(body.report, {
      shareUrl: body.share_url,
      pdfUrl: body.pdf_url,
      recipeUrl: body.recipe_url,
      artifactUrl: body.artifact_url,
      reproduceCurl: body.reproduce_curl,
      cached: !!body.cached,
      modelId: body.model_id,
    });
  }

  function queueResult(body, gen) {
    LD.pendingResult = body;
    LD.shareUrl = body.share_url;
    LD.pdfUrl = body.pdf_url;
    LD.recipeUrl = body.recipe_url;
    LD.artifactUrl = body.artifact_url;
    LD.reproduceCurl = body.reproduce_curl;
    maybeFinish(gen);
  }

  function applyReport(report, opts) {
    const {
      shareUrl, pdfUrl, recipeUrl, artifactUrl, reproduceCurl, cached, modelId,
    } = opts || {};
    LD.report = report;
    LD.shareUrl = shareUrl || LD.shareUrl;
    LD.pdfUrl = pdfUrl || LD.pdfUrl;
    LD.recipeUrl = recipeUrl || LD.recipeUrl;
    LD.artifactUrl = artifactUrl || LD.artifactUrl;
    LD.reproduceCurl = reproduceCurl || LD.reproduceCurl;

    const metrics = report.metrics || {};
    const before = metrics.bias_before || {};
    const after = metrics.bias_after || {};
    const reduction = metrics.bias_reduction || {};
    const weatBefore = reduction.weat_effect_size?.before ?? before.weat_effect_size ?? 0;
    const weatAfter = reduction.weat_effect_size?.after ?? after.weat_effect_size ?? weatBefore;
    const gapPct = reduction.mean_abs_association_gap?.reduction_pct ?? 0;
    const retention = (metrics.retention?.embedding_cosine_retention ?? 0) * 100;
    const edits = report.surgery?.num_edits ?? 0;
    const components = report.circuit?.selected_components || [];
    const names = probeNamesFromReport(report);
    const gapsBefore = before.per_probe_gaps || {};
    const gapsAfter = after.per_probe_gaps || {};
    const probeKeys = Object.keys(gapsBefore).slice(0, 3);

    // Stop scan loops but keep gen; clear only animation timers that would fight prove.
    LD.timers.forEach(clearTimeout);
    LD.timers = [];

    ldBuildBoard();
    ldBuildProbes(names);
    ldClearCells();
    components.slice(0, LD.hot.length).forEach((_, i) => {
      const cell = $("ldGrid").children[LD.hot[i]];
      if (cell) cell.classList.add("cut", "fade");
    });
    ldSetPhase("prove");
    $("ldBoardTitle").textContent = "Post-surgery audit";
    $("ldBoardMeta").textContent = `${edits} edits · ${components.length} circuit components`;
    $("ldWeat").textContent = `${Number(weatBefore).toFixed(2)} → ${Number(weatAfter).toFixed(2)}`;
    $("ldWeat").classList.add("good");
    $("ldWeatSub").textContent = cached
      ? "cached live run · same pipeline"
      : "live surgery · just finished";
    $("ldGap").textContent = `−${Number(gapPct).toFixed(0)}%`;
    $("ldGap").classList.add("good");
    $("ldGapSub").textContent = "association gap reduction";
    $("ldRet").textContent = `${Number(retention).toFixed(1)}%`;
    $("ldRet").classList.add("good");
    $("ldRetSub").textContent = "embedding cosine on neutral prompts";

    probeKeys.forEach((key, i) => {
      const b = Math.abs(gapsBefore[key] || 0);
      const a = Math.abs(gapsAfter[key] || 0);
      const max = Math.max(b, a, 1e-6);
      const fill = $(`ldFill${i}`);
      if (!fill) return;
      fill.classList.add("neutral");
      fill.style.width = `${Math.min(100, (a / max) * 100 * 0.35 + 12)}%`;
      $(`ldProbeVal${i}`).textContent = `${a >= 0 ? "+" : ""}${a.toFixed(3)}`;
    });

    ldProgress(100);
    const mid = modelId || report.model_id || selectedModel();
    $("ldModelLabel").textContent = `${mid} · live`;
    $("ldStatus").innerHTML =
      `Surgery complete — <b>−${Number(gapPct).toFixed(0)}% gap</b>, ` +
      `<b>${Number(retention).toFixed(1)}% retention</b>` +
      (cached ? " · cached" : " · live");
    ldLog([
      `> done · WEAT ${Number(weatBefore).toFixed(2)} → ${Number(weatAfter).toFixed(2)}`,
      `> export recipe / PDF — re-run on any Hugging Face model via API`,
    ]);

    const share = $("ldShare");
    if (share) {
      share.hidden = false;
      if ($("ldShareHtml") && LD.shareUrl) $("ldShareHtml").href = api(LD.shareUrl);
      if ($("ldSharePdf") && LD.pdfUrl) $("ldSharePdf").href = api(LD.pdfUrl);
      if ($("ldShareRecipe") && LD.recipeUrl) $("ldShareRecipe").href = api(LD.recipeUrl);
    }
    const art = $("ldArtifact");
    if (art) {
      if (LD.artifactUrl) {
        art.hidden = false;
        art.href = api(LD.artifactUrl);
        art.textContent = "Download weights";
      } else {
        art.hidden = false;
        art.href = "#";
        art.textContent = "Export weights";
        art.onclick = (e) => {
          e.preventDefault();
          startLiveDemo(true, true);
        };
      }
    }
    const signup = $("ldSignup");
    if (signup) {
      const q = new URLSearchParams({
        model: mid,
        bias: selectedBias(),
      });
      signup.href = `/signup?${q.toString()}`;
    }
    LD.running = false;
  }

  async function pollJob(jobId, gen) {
    const alive = ldAlive(gen);
    try {
      const res = await fetch(api(`/v1/public/demo-jobs/${jobId}`));
      if (!res.ok) throw new Error(`status ${res.status}`);
      const body = await res.json();
      if (!alive()) return;
      if (body.status === "queued" || body.status === "running") {
        $("ldStatus").innerHTML =
          body.status === "queued"
            ? "Job <b>queued</b> — walking the surgery story…"
            : "Server <b>running</b> real surgery — story continues…";
        ldLog([
          `> job ${jobId} · ${body.status}`,
          `> model ${body.model_id}`,
        ]);
        LD.pollTimer = setTimeout(() => pollJob(jobId, gen), 1200);
        return;
      }
      if (body.status === "succeeded" && body.report) {
        queueResult(body, gen);
        return;
      }
      if (body.status === "failed") {
        LD.running = false;
        $("ldStatus").innerHTML = `Surgery failed — <b>${body.error || "unknown error"}</b>`;
        ldLog([`> failed · ${body.error || "unknown"}`]);
      }
    } catch (err) {
      if (!alive()) return;
      LD.running = false;
      $("ldStatus").innerHTML = `Could not reach demo API — <b>${err.message}</b>`;
      ldLog([`> error · ${err.message}`]);
    }
  }

  async function startLiveDemo(force, exportWeights) {
    if (LD.running && !force && !exportWeights) return;
    ldClearTimers();
    LD.running = true;
    LD.gen += 1;
    const gen = LD.gen;
    LD.jobId = null;
    LD.report = null;
    LD.pendingResult = null;
    LD.storyDone = false;
    LD.artifactUrl = null;
    ldResetVisuals();

    const bias = selectedBias();
    const modelId = selectedModel();
    if ($("ldModelLabel")) $("ldModelLabel").textContent = `${modelId} · starting…`;
    ldLog([
      `> POST /v1/public/demo-jobs`,
      `> model=${modelId} · bias=${bias}` + (exportWeights ? " · export_weights" : ""),
    ]);
    playPhaseStory(gen);

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
        const detail = body.detail;
        throw new Error(
          typeof detail === "string" ? detail : `HTTP ${res.status}`
        );
      }
      if (!ldAlive(gen)()) return;
      LD.jobId = body.id;
      if (body.status === "succeeded" && body.report) {
        queueResult(body, gen);
        return;
      }
      pollJob(body.id, gen);
    } catch (err) {
      if (!ldAlive(gen)()) return;
      LD.running = false;
      $("ldStatus").innerHTML = `Demo unavailable — <b>${err.message}</b>`;
      ldLog([`> error · ${err.message}`]);
    }
  }

  function restartLiveDemo() {
    startLiveDemo(true, false);
  }

  function jumpLiveDemo(phase) {
    if (!LD.report) {
      startLiveDemo(false, false);
      return;
    }
    ldClearTimers();
    LD.running = false;
    LD.gen += 1;
    if (phase === "prove") {
      applyReport(LD.report, { cached: true, modelId: LD.pendingResult?.model_id });
      return;
    }
    ldBuildBoard();
    ldShowWorkingPhase(phase);
  }

  window.jumpLiveDemo = jumpLiveDemo;
  window.restartLiveDemo = restartLiveDemo;
  window.startLiveDemo = startLiveDemo;

  async function loadCatalogs() {
    const biasSel = $("ldBias");
    const modelSel = $("ldModelSelect");
    try {
      const [biasesRes, modelsRes] = await Promise.all([
        fetch(api("/v1/public/demo-biases")),
        fetch(api("/v1/public/demo-models")),
      ]);
      if (biasesRes.ok && biasSel) {
        const items = await biasesRes.json();
        if (Array.isArray(items) && items.length) {
          const current = biasSel.value;
          biasSel.innerHTML = items.map((b) =>
            `<option value="${b.name}">${b.name} — ${b.groups.join(" ↔ ")}</option>`
          ).join("");
          if ([...biasSel.options].some((o) => o.value === current)) {
            biasSel.value = current;
          } else if ([...biasSel.options].some((o) => o.value === "ad_gender_product")) {
            biasSel.value = "ad_gender_product";
          }
        }
      }
      if (modelsRes.ok && modelSel) {
        const data = await modelsRes.json();
        const featured = data.featured || [];
        const def = data.default_model_id || "openai/clip-vit-base-patch32";
        modelSel.innerHTML = featured.map((m) =>
          `<option value="${m.model_id}">${m.model_id}</option>`
        ).join("");
        if (![...modelSel.options].some((o) => o.value === def)) {
          modelSel.insertAdjacentHTML(
            "afterbegin",
            `<option value="${def}">${def}</option>`
          );
        }
        modelSel.value = def;
        if ($("ldModelInput") && !$("ldModelInput").value) {
          $("ldModelInput").placeholder = "or any Hugging Face model id…";
        }
      }
    } catch (_) { /* keep static options */ }
  }

  ldBuildBoard();
  ldResetVisuals();
  loadCatalogs().finally(() => {
    $("ldBias")?.addEventListener("change", () => startLiveDemo(true, false));
    $("ldModelSelect")?.addEventListener("change", () => {
      if ($("ldModelInput")) $("ldModelInput").value = "";
      startLiveDemo(true, false);
    });
    $("ldModelInput")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        startLiveDemo(true, false);
      }
    });
    const root = $("liveDemo");
    const kick = () => startLiveDemo(false, false);
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          io.disconnect();
          kick();
        }
      }, { threshold: 0.25 });
      io.observe(root);
    } else {
      kick();
    }
  });
})();
