/** Interactive Measure → Locate → Cut → Prove surgery demo (landing + console). */
(function () {
  function $(id) { return document.getElementById(id); }
  if (!$("liveDemo")) return;

  const LD = {
    cells: 48,
    hot: [5, 11, 14, 22, 27, 33, 38, 41],
    probes: [
      { name: "engineer", before: 92, after: 18 },
      { name: "surgeon", before: 84, after: 16 },
      { name: "nurse", before: 78, after: 14 },
    ],
    weatBefore: 1.71,
    weatAfter: 1.02,
    gapReduction: 78,
    retention: 99.0,
    edits: 20,
    phases: ["measure", "locate", "cut", "prove"],
    timers: [],
    running: false,
    gen: 0,
    phase: null,
    autoplay: true,
  };

  function ldClearTimers() {
    LD.timers.forEach(clearTimeout);
    LD.timers = [];
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
    const order = LD.phases;
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

  function ldBuildBoard() {
    $("ldGrid").innerHTML = Array.from({ length: LD.cells }, (_, i) =>
      `<div class="ld-cell" data-i="${i}"></div>`).join("");
    $("ldProbes").innerHTML = LD.probes.map((p, i) =>
      `<div class="ld-probe" data-p="${i}">
         <span>${p.name}</span>
         <div class="track"><div class="fill" id="ldFill${i}"></div></div>
         <span id="ldProbeVal${i}">—</span>
       </div>`).join("");
  }

  function ldResetVisuals() {
    ldBuildBoard();
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
    $("ldBoardMeta").textContent = "12 layers · attention + MLP";
    $("ldStatus").innerHTML = "Click <b>01–04</b> to explore · or watch the autoplay";
    ldProgress(0);
    document.querySelectorAll("#ldRail .ld-phase").forEach((el) =>
      el.classList.remove("active", "done"));
    ldLog(["> scalpel surgery — click a step or wait for autoplay"]);
  }

  function ldAnimateNumber(el, from, to, dur, fmt) {
    const start = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - start) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(from + (to - from) * e);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function ldClearCells() {
    [...$("ldGrid").children].forEach((c) =>
      c.classList.remove("scan", "hot", "cut", "fade"));
  }

  function ldShowMeasure(animate) {
    const alive = ldAlive(LD.gen);
    ldClearCells();
    ldSetPhase("measure");
    $("ldBoardTitle").textContent = "Measuring association bias";
    $("ldBoardMeta").textContent = "contrastive prompts · WEAT";
    $("ldStatus").innerHTML = "Measuring bias across <b>profession probes</b>…";
    ldLog([
      "> measure: 48 contrastive pairs × profession probes",
      "> computing WEAT effect size on text tower…",
    ]);
    ldProgress(22);
    $("ldGap").textContent = "—";
    $("ldGap").classList.remove("good");
    $("ldRet").textContent = "—";
    $("ldRet").classList.remove("good");
    $("ldWeat").classList.remove("good");
    if (animate) {
      ldAnimateNumber($("ldWeat"), 0, LD.weatBefore, 900, (v) => v.toFixed(2));
      let scan = 0;
      const scanStep = () => {
        if (!alive()) return;
        const cells = [...$("ldGrid").children];
        cells.forEach((c) => c.classList.remove("scan"));
        for (let k = 0; k < 6; k++) cells[(scan + k) % cells.length]?.classList.add("scan");
        scan += 3;
        if (scan < cells.length + 12) ldLater(70, scanStep);
      };
      scanStep();
      LD.probes.forEach((p, i) => {
        ldLater(180 + i * 160, () => {
          if (!alive()) return;
          const fill = $(`ldFill${i}`);
          fill.classList.remove("neutral");
          fill.style.width = `${p.before}%`;
          $(`ldProbeVal${i}`).textContent = `+${((p.before / 100) * 0.02).toFixed(3)}`;
        });
      });
    } else {
      $("ldWeat").textContent = LD.weatBefore.toFixed(2);
      LD.probes.forEach((p, i) => {
        const fill = $(`ldFill${i}`);
        fill.classList.remove("neutral");
        fill.style.width = `${p.before}%`;
        $(`ldProbeVal${i}`).textContent = `+${((p.before / 100) * 0.02).toFixed(3)}`;
      });
    }
    $("ldWeatSub").textContent = "baseline — stereotyped association";
  }

  function ldShowLocate(animate) {
    const alive = ldAlive(LD.gen);
    ldClearCells();
    ldSetPhase("locate");
    $("ldBoardTitle").textContent = "Isolating the bias circuit";
    $("ldBoardMeta").textContent = "attribution across heads + MLPs";
    $("ldStatus").innerHTML = "Localizing which components <b>write the bias</b>…";
    ldLog([
      "> locate: diff-in-means direction @ layer 9 (9.5σ)",
      "> ranking component writes onto bias subspace…",
    ]);
    ldProgress(48);
    $("ldWeat").textContent = LD.weatBefore.toFixed(2);
    $("ldWeatSub").textContent = "baseline — stereotyped association";
    LD.probes.forEach((p, i) => {
      const fill = $(`ldFill${i}`);
      fill.classList.remove("neutral");
      fill.style.width = `${p.before}%`;
      $(`ldProbeVal${i}`).textContent = `+${((p.before / 100) * 0.02).toFixed(3)}`;
    });
    if (animate) {
      LD.hot.forEach((idx, n) => {
        ldLater(n * 140, () => {
          if (!alive()) return;
          $("ldGrid").children[idx]?.classList.add("hot");
          $("ldBoardMeta").textContent = `${n + 1} of ${LD.hot.length} circuit components`;
        });
      });
    } else {
      LD.hot.forEach((idx) => $("ldGrid").children[idx]?.classList.add("hot"));
      $("ldBoardMeta").textContent = `${LD.hot.length} of ${LD.hot.length} circuit components`;
    }
  }

  function ldShowCut(animate) {
    const alive = ldAlive(LD.gen);
    ldClearCells();
    ldSetPhase("cut");
    $("ldBoardTitle").textContent = "Applying closed-form surgery";
    $("ldBoardMeta").textContent = "W′ = W − α Vᵀ(VW)";
    $("ldStatus").innerHTML = "Severing circuit with <b>rank-k projections</b>…";
    ldLog([
      "> cut: projecting bias subspace out of selected weights",
      "> calibrate α → target neutrality (not over-erasure)",
    ]);
    ldProgress(72);
    $("ldWeat").textContent = LD.weatBefore.toFixed(2);
    LD.probes.forEach((p, i) => {
      const fill = $(`ldFill${i}`);
      fill.classList.remove("neutral");
      fill.style.width = `${p.before}%`;
      $(`ldProbeVal${i}`).textContent = `+${((p.before / 100) * 0.02).toFixed(3)}`;
    });
    if (animate) {
      let edits = 0;
      LD.hot.forEach((idx, n) => {
        ldLater(n * 120, () => {
          if (!alive()) return;
          const cell = $("ldGrid").children[idx];
          if (!cell) return;
          cell.classList.add("cut");
          edits += Math.ceil(LD.edits / LD.hot.length);
          $("ldBoardMeta").textContent = `${Math.min(edits, LD.edits)} projection edits applied`;
          ldLater(200, () => { if (alive()) cell.classList.add("fade"); });
        });
      });
    } else {
      LD.hot.forEach((idx) => {
        const cell = $("ldGrid").children[idx];
        if (cell) cell.classList.add("cut", "fade");
      });
      $("ldBoardMeta").textContent = `${LD.edits} projection edits applied`;
    }
  }

  function ldShowProve(animate) {
    const alive = ldAlive(LD.gen);
    ldClearCells();
    LD.hot.forEach((idx) => {
      const cell = $("ldGrid").children[idx];
      if (cell) cell.classList.add("cut", "fade");
    });
    ldSetPhase("prove");
    $("ldBoardTitle").textContent = "Post-surgery audit";
    $("ldBoardMeta").textContent = `${LD.edits} edits · circuit severed`;
    $("ldStatus").innerHTML = "Verifying <b>bias down</b>, capability retained…";
    ldLog([
      "> prove: re-run WEAT + retention probes",
      "> writing compliance report + edited model file…",
    ]);
    ldProgress(92);
    if (animate) {
      ldAnimateNumber($("ldWeat"), LD.weatBefore, LD.weatAfter, 800, (v) =>
        `${LD.weatBefore.toFixed(2)} → ${v.toFixed(2)}`);
      ldLater(150, () => {
        if (!alive()) return;
        $("ldWeat").classList.add("good");
        $("ldWeatSub").textContent = "−40% WEAT · live CLIP run";
      });
      ldAnimateNumber($("ldGap"), 0, LD.gapReduction, 800, (v) => `−${v.toFixed(0)}%`);
      ldAnimateNumber($("ldRet"), 90, LD.retention, 800, (v) => `${v.toFixed(1)}%`);
      LD.probes.forEach((p, i) => {
        ldLater(120 + i * 100, () => {
          if (!alive()) return;
          const fill = $(`ldFill${i}`);
          fill.classList.add("neutral");
          fill.style.width = `${p.after}%`;
          $(`ldProbeVal${i}`).textContent = `+${((p.after / 100) * 0.02).toFixed(3)}`;
        });
      });
      ldLater(1100, () => {
        if (!alive()) return;
        ldProgress(100);
        $("ldStatus").innerHTML =
          "Surgery complete — <b>−78% gap</b>, <b>99.0% retention</b> · click 01–04 to revisit";
        ldLog([
          "> done · WEAT 1.71 → 1.02 (−40%) · gap −78% · retention 99.0%",
          "> artifact ready · report.html · model.zip",
        ]);
        LD.running = false;
      });
    } else {
      $("ldWeat").textContent = `${LD.weatBefore.toFixed(2)} → ${LD.weatAfter.toFixed(2)}`;
      $("ldWeat").classList.add("good");
      $("ldWeatSub").textContent = "−40% WEAT · live CLIP run";
      $("ldGap").textContent = `−${LD.gapReduction}%`;
      $("ldGap").classList.add("good");
      $("ldGapSub").textContent = "association gap reduction";
      $("ldRet").textContent = `${LD.retention.toFixed(1)}%`;
      $("ldRet").classList.add("good");
      $("ldRetSub").textContent = "embedding cosine on neutral prompts";
      LD.probes.forEach((p, i) => {
        const fill = $(`ldFill${i}`);
        fill.classList.add("neutral");
        fill.style.width = `${p.after}%`;
        $(`ldProbeVal${i}`).textContent = `+${((p.after / 100) * 0.02).toFixed(3)}`;
      });
      ldProgress(100);
      $("ldStatus").innerHTML =
        "Surgery complete — <b>−78% gap</b>, <b>99.0% retention</b> · click 01–04 to revisit";
      ldLog([
        "> done · WEAT 1.71 → 1.02 (−40%) · gap −78% · retention 99.0%",
        "> artifact ready · report.html · model.zip",
      ]);
    }
    $("ldGap").classList.add("good");
    $("ldGapSub").textContent = "association gap reduction";
    $("ldRet").classList.add("good");
    $("ldRetSub").textContent = "embedding cosine on neutral prompts";
  }

  function jumpLiveDemo(phase) {
    if (!LD.phases.includes(phase)) return;
    ldClearTimers();
    LD.running = false;
    LD.autoplay = false;
    LD.gen += 1;
    ldBuildBoard();
    const animate = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (phase === "measure") ldShowMeasure(animate);
    else if (phase === "locate") ldShowLocate(animate);
    else if (phase === "cut") ldShowCut(animate);
    else ldShowProve(animate);
  }

  function restartLiveDemo() {
    ldClearTimers();
    LD.running = false;
    LD.autoplay = true;
    startLiveDemo(true);
  }

  function startLiveDemo(force) {
    if (!force && LD.running) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      ldResetVisuals();
      jumpLiveDemo("prove");
      return;
    }
    ldClearTimers();
    LD.running = true;
    LD.autoplay = true;
    const gen = ++LD.gen;
    const alive = () => gen === LD.gen;
    ldResetVisuals();
    $("ldStatus").innerHTML = "Loading <b>clip-vit-base-patch32</b>… · click any step to jump";
    ldLog(["> load openai/clip-vit-base-patch32", "> bias spec: gender_profession · mode=edit · k=4"]);
    ldProgress(4);
    ldLater(600, () => { if (alive() && LD.autoplay) ldShowMeasure(true); });
    ldLater(2800, () => { if (alive() && LD.autoplay) ldShowLocate(true); });
    ldLater(2800 + 160 * LD.hot.length + 500, () => {
      if (alive() && LD.autoplay) ldShowCut(true);
    });
    const proveAt = 2800 + 160 * LD.hot.length + 500 + 140 * LD.hot.length + 600;
    ldLater(proveAt, () => { if (alive() && LD.autoplay) ldShowProve(true); });
  }

  window.jumpLiveDemo = jumpLiveDemo;
  window.restartLiveDemo = restartLiveDemo;
  window.startLiveDemo = startLiveDemo;

  ldBuildBoard();
  ldResetVisuals();
  const root = $("liveDemo");
  const kick = () => startLiveDemo(true);
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        io.disconnect();
        kick();
      }
    }, { threshold: 0.35 });
    io.observe(root);
  } else {
    kick();
  }
})();
