/* CAI5735 Final Project - Group 6
   Interactive explorer: every panel is linked to the same filter, model, threshold and selected patient.
   Uses the helpers and data from app.js (D, S, MODELS, confusion, curves, mk, reg, refresh, ...). */
"use strict";
(function () {
  const X = { rows: null, binIdx: {}, spec: {}, cell: null, patient: 17, tour: -1, lastRoc: [], flagged: 0 };
  const SEL = {};                         // selected bin indexes for each variable
  const n = () => D.y_test.length;
  const pc = v => (isFinite(v) ? pct(v, 0) : "-");
  const num = v => (isFinite(v) ? f3(v) : "-");
  const cellOf = (p, y, t) => (p >= t ? (y ? "tp" : "fp") : (y ? "fn" : "tn"));
  const CELL_NAME = { tp: "true positive", fp: "false positive", fn: "false negative", tn: "true negative" };

  /* ---- binning for the filter histograms: at most 10 bars per variable ---- */
  function binSpec(f) {
    const vals = D.data[f], u = [...new Set(vals)].sort((a, b) => a - b);
    if (f === "diabetes_med") return { labels: ["No", "Yes"], fn: v => v };
    if (u.length <= 10) return { labels: u.map(String), fn: v => u.indexOf(v) };
    const lo = u[0], hi = quant(vals, 0.99), w = Math.max(1, Math.ceil((hi - lo + 1) / 10)), nb = 10;
    return { labels: Array.from({ length: nb }, (_, i) => (i === nb - 1 ? `${lo + i * w}+` : `${lo + i * w}-${lo + i * w + w - 1}`)), fn: v => Math.min(nb - 1, Math.floor((v - lo) / w)) };
  }
  function prep() {
    if (X.rows) return;
    X.rows = D.test_idx.map(k => FEATS.map(f => D.data[f][k]));
    FEATS.forEach((f, j) => { X.spec[f] = binSpec(f); SEL[f] = new Set(); X.binIdx[f] = X.rows.map(r => X.spec[f].fn(r[j])); });
    const R = rng(11); X.jit = D.test_idx.map(() => R());
  }
  const passes = i => FEATS.every(f => !SEL[f].size || SEL[f].has(X.binIdx[f][i]));
  const filtersOn = () => FEATS.some(f => SEL[f].size);

  /* ---- one calculation shared by every panel ---- */
  function compute() {
    const p = D.probs[S.model], y = D.y_test, idx = [];
    for (let i = 0; i < n(); i++) if (passes(i)) idx.push(i);
    const sp = idx.map(i => p[i]), sy = idx.map(i => y[i]);
    const c = confusion(sp, sy, S.thr);
    const both = sum(sy) > 0 && sum(sy) < sy.length;
    const cv = both && idx.length >= 15 ? curves(sp, sy) : null;
    return { p, y, idx, sp, sy, c, cv };
  }

  /* ---- html helpers for panels ---- */
  function panel(id, title, extra = "") {
    return h(`<div class="card ex-panel" id="${id}"><div class="ch"><h4>${title}</h4><div class="tg">${extra}</div></div><div class="ex-body"></div><p class="how"></p><p class="says"></p></div>`);
  }
  const body = el => $(".ex-body", el);
  const how = (el, t) => { $(".how", el).innerHTML = "<b>How to read this.</b> " + t; };
  const says = (el, t) => { const s = $(".says", el); s.innerHTML = t ? "<b>What it says.</b> " + t : ""; s.style.display = t ? "" : "none"; };

  /* ---- PAGE ---- */
  PAGES.explorer = () => {
    prep();
    const p = h("<div></div>");
    p.append(titlebar("Interactive explorer", "One dashboard where every chart is linked. Filter patients, move the threshold, or pick a patient, and everything updates together."));
    p.append(h(`<div class="ex-intro"><span><b>Try this:</b> click bars to filter patients, click the ROC curve or the risk chart to move the threshold, click a dot to pick a patient.</span> <button class="btn pri" id="exTourBtn">Guided tour</button> <button class="btn" id="exReset">Reset everything</button></div>`));
    p.append(h(`<div class="ex-tour" id="exTour" style="display:none"><div id="exTourText"></div><div class="ex-tour-btns"><button class="btn" id="exPrev">Back</button><button class="btn pri" id="exNext">Next</button><button class="btn" id="exSkip">Skip tour</button></div></div>`));
    p.append(settingsBar());

    const metrics = h(`<div id="exMetrics"></div>`);
    const filt = panel("ex-filters", "1. Filter the patients", `<button class="btn" id="exClear">Clear filters</button>`);
    const cm = panel("ex-cm", "2. Confusion matrix");
    const roc = panel("ex-roc", "3. ROC curve (click to set the threshold)");
    const risk = panel("ex-risk", "4. Predicted risk (click to set the threshold)");
    const strip = panel("ex-strip", "5. Every test patient (click a dot to pick one)", `<button class="btn" id="exRand">Random patient</button>`);
    const pt = panel("ex-patient", "6. The selected patient");
    $(".ex-body", roc).innerHTML = `<div class="cbox" style="height:300px"><canvas></canvas></div>`;
    $(".ex-body", risk).innerHTML = `<div class="cbox" style="height:300px"><canvas></canvas></div>`;
    $(".ex-body", strip).innerHTML = `<div class="cbox" style="height:300px"><canvas></canvas></div>`;
    const rowA = h(`<div class="ex-grid2"></div>`), rc = h(`<div></div>`); rc.append(cm, roc); rowA.append(filt, rc);
    const rowB = h(`<div class="ex-grid2"></div>`); rowB.append(risk, strip);
    p.append(metrics, rowA, rowB, pt);

    /* clear / reset / random / tour buttons */
    const clearAll = () => { FEATS.forEach(f => SEL[f].clear()); X.cell = null; };
    $("#exClear", filt).onclick = () => { clearAll(); refresh(); };
    $("#exReset", p).onclick = () => { clearAll(); S.model = "Random forest"; S.thr = 0.40; X.patient = 17; syncSettings(); };
    $("#exRand", strip).onclick = () => { pickRandom(); };
    const tours = [
      ["ex-filters", "Step 1 of 6. Click bars in any histogram to keep only those patients. For example, choose the highest diagnoses bars. Every other chart now describes only the patients you kept."],
      ["ex-cm", "Step 2 of 6. The confusion matrix and the numbers at the top use only the filtered patients. Click one of the four boxes to focus on those patients in the dot chart (step 5)."],
      ["ex-roc", "Step 3 of 6. The ROC curve is drawn for the filtered patients. Click a point on the curve to choose a threshold. The dot, the matrix and every other tab follow."],
      ["ex-risk", "Step 4 of 6. This chart shows the predicted risk for readmitted and not readmitted patients. Click it to move the threshold, and watch how the two groups overlap."],
      ["ex-strip", "Step 5 of 6. Each dot is one real test patient. Dots right of the dashed line get flagged. Click a dot to select that patient, or press Random patient."],
      ["ex-patient", "Step 6 of 6. See the selected patient's values, what each model predicted, and how the logistic regression built its score. Use the button to send them to the Try a patient tab."]
    ];
    const tourShow = k => {
      X.tour = k; $$(".tour-hl", p).forEach(e => e.classList.remove("tour-hl"));
      const bar = $("#exTour", p);
      if (k < 0 || k >= tours.length) { bar.style.display = "none"; return; }
      bar.style.display = ""; $("#exTourText", p).textContent = tours[k][1];
      const tgt = $("#" + tours[k][0], p); tgt.classList.add("tour-hl"); tgt.scrollIntoView({ behavior: "smooth", block: "center" });
      $("#exPrev", p).disabled = k === 0; $("#exNext", p).textContent = k === tours.length - 1 ? "Finish" : "Next";
    };
    $("#exTourBtn", p).onclick = () => tourShow(0);
    $("#exPrev", p).onclick = () => tourShow(X.tour - 1);
    $("#exNext", p).onclick = () => tourShow(X.tour + 1);
    $("#exSkip", p).onclick = () => tourShow(-1);

    function pickRandom() { const R = Math.random(), pool = []; for (let i = 0; i < n(); i++) if (passes(i)) pool.push(i); if (!pool.length) return; X.patient = pool[Math.floor(R * pool.length)]; refresh(); }
    function stepPatient(d) { const pool = []; for (let i = 0; i < n(); i++) if (passes(i)) pool.push(i); if (!pool.length) return; let k = pool.indexOf(X.patient); k = k < 0 ? 0 : (k + d + pool.length) % pool.length; X.patient = pool[k]; refresh(); }

    /* ---- the one redraw that updates every panel ---- */
    const wrap = h(`<div></div>`);
    wrap.redraw = () => {
      const C = compute(), c = C.c, tot = n();
      const rate = C.idx.length ? mean(C.sy) : NaN, all = mean(D.y_test), auc = C.cv ? C.cv.auc : NaN, aucAll = ROC[S.model].auc;
      if (!passes(X.patient)) { const first = C.idx[0]; if (first !== undefined) X.patient = first; }

      /* metrics bar */
      metrics.innerHTML = kpis([
        ["Patients shown", `${C.idx.length}`, `of ${tot} test patients`], ["Readmitted", pc(rate), `overall ${pc(all)}`], ["Accuracy", pc(c.acc), `threshold ${S.thr.toFixed(2)}`],
        ["Sensitivity", pc(c.sens), "readmissions caught"], ["Specificity", pc(c.spec), "non-readmissions cleared"], ["AUROC", num(auc), `all patients ${f3(aucAll)}`]]).outerHTML;

      /* 1. filters */
      const selP = X.patient;
      const f1 = FEATS.map((f, j) => {
        const sp = X.spec[f], nb = sp.labels.length, tot_ = Array(nb).fill(0), cur = Array(nb).fill(0);
        for (let i = 0; i < tot; i++) { const b = X.binIdx[f][i]; tot_[b]++; if (passes(i)) cur[b]++; }
        const mx = Math.max(...tot_), pb = X.binIdx[f][selP];
        const bars = sp.labels.map((l, b) => `<div class="fb ${SEL[f].has(b) ? "on" : ""}" data-f="${f}" data-b="${b}" title="${axT(f)}: ${l} (${cur[b]} of ${tot_[b]} patients shown)"><div class="fcol"><i class="fall" style="height:${tot_[b] / mx * 100}%"></i><i class="fnow" style="height:${cur[b] / mx * 100}%"></i></div><span>${l}</span>${b === pb ? '<em title="selected patient">▲</em>' : "<em></em>"}</div>`).join("");
        return `<div class="fvar"><div class="ftitle">${axT(f)}</div><div class="fbars" style="grid-template-columns:repeat(${nb},1fr)">${bars}</div></div>`;
      }).join("");
      body(filt).innerHTML = `<div class="flegend"><span class="lg all"></span>all test patients <span class="lg now"></span>patients shown</div>${f1}`;
      $$(".fb", filt).forEach(b => b.onclick = () => { const f = b.dataset.f, k = +b.dataset.b; if (SEL[f].has(k)) SEL[f].delete(k); else SEL[f].add(k); refresh(); });
      how(filt, "Each small chart counts the test patients in each range of one variable (bar heights are counts of patients). Click bars to keep only those ranges. Light bars are all 750 test patients, dark bars are the patients still shown. The triangle marks the bar your selected patient is in.");
      says(filt, filtersOn() ? `You kept ${C.idx.length} of ${tot} patients (${pct(C.idx.length / tot, 0)}). Their readmission rate is ${pc(rate)}, compared with ${pc(all)} for everyone. ${C.idx.length < 30 ? "This is a small group, so the numbers below are noisy." : "Compare the matrix and curves with the unfiltered view to see if the model works better or worse for this group."}` : "No filters yet, so all 750 test patients are shown. Click a few bars, such as the highest number of diagnoses, and watch every other panel change.");

      /* 2. confusion matrix */
      const P_ = c.tp + c.fn, N_ = c.tn + c.fp, mine = cellOf(C.p[selP], C.y[selP], S.thr);
      const cell = (k, val, lab, cls) => `<div class="cell ${cls} ${X.cell === k ? "pick" : ""} ${mine === k ? "me" : ""}" data-c="${k}" title="Click to focus on these patients">${val}<small>${lab}</small></div>`;
      body(cm).innerHTML = `<div class="cm"><div></div><div>Predicted<br>not readmitted</div><div>Predicted<br>readmitted</div>
        <div class="lab">Actual<br>not readm.</div>${cell("tn", c.tn, "TRUE NEGATIVE", "good")}${cell("fp", c.fp, "FALSE POSITIVE", "bad")}
        <div class="lab">Actual<br>readmitted</div>${cell("fn", c.fn, "FALSE NEGATIVE", "bad")}${cell("tp", c.tp, "TRUE POSITIVE", "good")}</div>`;
      $$(".cell", cm).forEach(b => b.onclick = () => { X.cell = X.cell === b.dataset.c ? null : b.dataset.c; refresh(); });
      how(cm, "Each box counts test patients (the number of patients) for the selected model at the current threshold. Teal boxes are correct and coral boxes are mistakes. Click a box to highlight those patients in chart 5. The outlined box is where the selected patient falls.");
      says(cm, C.idx.length ? `At a threshold of ${S.thr.toFixed(2)}, the ${S.model.toLowerCase()} caught ${c.tp} of ${P_} readmitted patients (${pc(c.sens)}) and cleared ${c.tn} of ${N_} patients who did not return (${pc(c.spec)}). It missed ${c.fn} readmissions and raised ${c.fp} false alarms.` : "No patients match the filters.");

      /* 3. ROC */
      X.lastRoc = C.cv ? C.cv.roc : [];
      const rcv = $("canvas", roc);
      if (C.cv) {
        const full = ROC[S.model].roc, ds = [{ label: `Filtered patients (AUROC ${f3(C.cv.auc)})`, data: C.cv.roc, showLine: true, pointRadius: 0, borderColor: MC(MODELS.indexOf(S.model)), borderWidth: 3 },
          { label: `All patients (AUROC ${f3(aucAll)})`, data: full, showLine: true, pointRadius: 0, borderColor: css("--muted"), borderWidth: 1.2, borderDash: [4, 3] },
          { label: "Coin flip", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderColor: css("--line"), borderWidth: 1 },
          { label: "Current threshold", data: [{ x: 1 - c.spec, y: c.sens }], pointRadius: 8, backgroundColor: css("--ink"), borderColor: css("--paper"), borderWidth: 2 }];
        mk(rcv, { type: "scatter", data: { datasets: ds }, options: {
          onClick: (e, _, ch) => { const x = ch.scales.x.getValueForPixel(e.x), y = ch.scales.y.getValueForPixel(e.y); let best = null, bd = 1e9; X.lastRoc.forEach(q => { const d = (q.x - x) ** 2 + (q.y - y) ** 2; if (d < bd) { bd = d; best = q; } }); if (best && best.t <= 1) setThr(best.t); },
          onHover: (e, a, ch) => { ch.canvas.style.cursor = "pointer"; },
          scales: { x: { min: 0, max: 1, title: { display: true, text: "False positive rate (share of non-readmitted flagged, 0 to 1)" } }, y: { min: 0, max: 1, title: { display: true, text: "Sensitivity (share of readmissions caught, 0 to 1)" } } },
          plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
      } else { const old = Chart.getChart(rcv); if (old) old.destroy(); }
      how(roc, "The curve shows sensitivity against the false positive rate (both 0 to 1) for every possible threshold, using only the filtered patients. The dashed gray line is the curve for all patients. Closer to the top left is better. Click the curve to choose that threshold.");
      says(roc, C.cv ? `For these ${C.idx.length} patients the AUROC is ${f3(C.cv.auc)} (all patients: ${f3(aucAll)}). ${Math.abs(C.cv.auc - aucAll) < 0.03 ? "The model performs about the same for this group as for everyone." : C.cv.auc > aucAll ? "The model does somewhat better for this group." : "The model does worse for this group."} The curve stays close to the diagonal, so the model is weak either way.` : "The curve needs at least 15 patients with both outcomes. Loosen the filters.");

      /* 4. risk histogram */
      const kcv = $("canvas", risk), pr = D.probs[S.model], lo = Math.min(...pr), hi = Math.max(...pr) + 1e-9, nb = 24, w = (hi - lo) / nb;
      const cnt = g => { const a = Array(nb).fill(0); C.idx.forEach(i => { if (C.y[i] === g) a[Math.min(nb - 1, Math.floor((C.p[i] - lo) / w))]++; }); return a; };
      const mark = (ch, v, col, dash, lab) => { const ca = ch.chartArea, x = ca.left + (v - lo) / (hi - lo) * (ca.right - ca.left); if (x < ca.left || x > ca.right) return; const g = ch.ctx; g.save(); g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dash); g.beginPath(); g.moveTo(x, ca.top); g.lineTo(x, ca.bottom); g.stroke(); g.setLineDash([]); g.fillStyle = col; g.font = "11px sans-serif"; g.fillText(lab, Math.min(x + 4, ca.right - 60), ca.top + 12); g.restore(); };
      mk(kcv, { type: "bar", data: { labels: Array.from({ length: nb }, (_, i) => (lo + i * w).toFixed(2)), datasets: [{ label: "Not readmitted", data: cnt(0), backgroundColor: css("--teal") + "aa", borderRadius: 2 }, { label: "Readmitted", data: cnt(1), backgroundColor: css("--coral") + "aa", borderRadius: 2 }] }, options: {
        onClick: (e, _, ch) => { const ca = ch.chartArea; if (e.x < ca.left || e.x > ca.right) return; setThr(lo + (e.x - ca.left) / (ca.right - ca.left) * (hi - lo)); },
        onHover: (e, a, ch) => { ch.canvas.style.cursor = "pointer"; },
        scales: { x: { title: { display: true, text: "Predicted probability of readmission (0 to 1)" }, grid: { display: false }, ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: "Number of patients (count)" } } } },
        plugins: [{ id: "ex-marks", afterDraw(ch) { mark(ch, S.thr, css("--ink"), [5, 4], "threshold"); mark(ch, C.p[selP], css("--sun"), [], "selected"); } }] });
      how(risk, "Each bar counts the filtered patients (number of patients) whose predicted probability of readmission falls in that range, with teal for patients who were not readmitted and coral for those who were. The dashed line is the threshold and the yellow line is the selected patient. Patients to the right of the dashed line are flagged.");
      const m0 = mean(C.idx.filter(i => !C.y[i]).map(i => C.p[i])), m1 = mean(C.idx.filter(i => C.y[i]).map(i => C.p[i]));
      says(risk, C.idx.length ? `The average predicted risk is ${pc(m0)} for patients who did not return and ${pc(m1)} for those who did, a gap of only ${isFinite(m0 + m1) ? ((m1 - m0) * 100).toFixed(1) : "-"} points. The two colors overlap heavily, so moving the threshold only trades missed readmissions for false alarms.` : "No patients match the filters.");

      /* 5. every test patient */
      const sc = $("canvas", strip), jig = i => C.y[i] + (X.jit[i] - 0.5) * 0.6;
      const inFocus = i => passes(i) && (!X.cell || cellOf(C.p[i], C.y[i], S.thr) === X.cell);
      const pts = (g, f) => { const o = []; for (let i = 0; i < n(); i++) if (C.y[i] === g && f(i)) o.push({ x: C.p[i], y: jig(i), i }); return o; };
      const flagLeft = C.idx.filter(i => C.y[i] === 1 && C.p[i] < S.thr).length;
      mk(sc, { type: "scatter", data: { datasets: [
        { label: "Not readmitted", data: pts(0, inFocus), backgroundColor: css("--teal") + "bb", pointRadius: 3.5 },
        { label: "Readmitted", data: pts(1, inFocus), backgroundColor: css("--coral") + "bb", pointRadius: 3.5 },
        { label: "Hidden by filters", data: [...pts(0, i => !inFocus(i)), ...pts(1, i => !inFocus(i))], backgroundColor: css("--line"), pointRadius: 2 },
        { label: "Selected patient", data: [{ x: C.p[selP], y: jig(selP), i: selP }], backgroundColor: css("--sun"), borderColor: css("--ink"), borderWidth: 2, pointRadius: 9 }] }, options: {
        onClick: (e, _, ch) => { const els = ch.getElementsAtEventForMode(e, "nearest", { intersect: true }, true).filter(el => el.datasetIndex < 2 || el.datasetIndex === 3); if (els.length) { X.patient = ch.data.datasets[els[0].datasetIndex].data[els[0].index].i; setTimeout(refresh, 0); } },
        onHover: (e, els, ch) => { ch.canvas.style.cursor = els.length ? "pointer" : "default"; },
        scales: { x: { title: { display: true, text: "Predicted probability of readmission (0 to 1)" } }, y: { min: -0.5, max: 1.5, title: { display: true, text: "Actual outcome (0 = not readmitted, 1 = readmitted; jittered)" }, ticks: { callback: v => (v === 0 ? "Not readmitted" : v === 1 ? "Readmitted" : ""), stepSize: 0.5 } } },
        plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } }, tooltip: { callbacks: { label: c => ` Patient #${c.raw.i + 1}: predicted ${pct(c.raw.x, 0)}` } } } },
        plugins: [{ id: "ex-thr", beforeDatasetsDraw(ch) { const ca = ch.chartArea, x = ch.scales.x.getPixelForValue(S.thr); if (x > ca.right) return; const g = ch.ctx; g.save(); g.fillStyle = css("--coral") + "14"; g.fillRect(Math.max(x, ca.left), ca.top, ca.right - Math.max(x, ca.left), ca.bottom - ca.top); g.strokeStyle = css("--ink"); g.setLineDash([5, 4]); g.beginPath(); g.moveTo(x, ca.top); g.lineTo(x, ca.bottom); g.stroke(); g.restore(); } }] });
      how(strip, "Each dot is one test patient. Left to right is the model's predicted probability of readmission (0 to 1). The top row is patients who really were readmitted and the bottom row is patients who were not (dots are spread out vertically only so they do not overlap). The shaded area to the right of the dashed line is flagged. Gray dots are hidden by your filters. Click a dot to select that patient.");
      says(strip, `${C.idx.length} patients are shown. ${flagLeft} readmitted patients sit left of the dashed line, so they are missed. ${X.cell ? `You are focusing on the ${CELL_NAME[X.cell]} patients (${C.idx.filter(i => cellOf(C.p[i], C.y[i], S.thr) === X.cell).length}).` : "Coral and teal dots are mixed in every part of the chart, which is why the model cannot separate the groups."}`);

      /* 6. selected patient */
      const rowv = X.rows[selP], act = C.y[selP], lrm = D.lr;
      const zs = rowv.map((v, i) => (v - lrm.mean[i]) / lrm.scale[i]), contrib = zs.map((z, i) => z * lrm.coef[i]), lrProb = predict(MODELS[0], rowv);
      const maxc = Math.max(...contrib.map(Math.abs), 0.01);
      const preds = MODELS.map((m, i) => { const q = D.probs[m][selP], yes = q >= S.thr, ok = yes === !!act; return `<div class="pcard ${yes ? "yes" : ""}"><span class="tag">${m}</span><b>${(q * 100).toFixed(0)}%</b><div class="bar"><i style="width:${q * 100}%"></i><u style="left:${S.thr * 100}%"></u></div><span>${yes ? "Flagged as readmitted" : "Not flagged"}</span><br><small style="color:${ok ? "var(--teal)" : "var(--coral)"}"><b>${ok ? "Correct" : "Wrong"}</b> (${cellOf(q, act, S.thr).toUpperCase()})</small></div>`; }).join("");
      const vals = FEATS.map((f, i) => { const v = D.data[f], pctl = f === "diabetes_med" ? null : Math.round(v.filter(a => a <= rowv[i]).length / v.length * 100); return `<tr><td>${axT(f)}</td><td class="n">${f === "diabetes_med" ? (rowv[i] ? "Yes" : "No") : rowv[i]}</td><td>${pctl === null ? "" : `<div class="bar"><i style="width:${pctl}%"></i></div>`}</td><td class="n">${pctl === null ? "" : pctl + "th"}</td><td class="n">${contrib[i] >= 0 ? "+" : ""}${contrib[i].toFixed(2)}</td><td><div class="cbar"><i style="width:${Math.abs(contrib[i]) / maxc * 50}%;${contrib[i] >= 0 ? "left:50%;background:var(--coral)" : "right:50%;background:var(--teal)"}"></i></div></td></tr>`; }).join("");
      const k = D.test_idx[selP];
      body(pt).innerHTML = `<div class="pthead"><b>Test patient #${selP + 1}</b> <span class="tag" style="background:${act ? "var(--coral)" : "var(--teal)"};color:#fff">Actual: ${act ? "readmitted" : "not readmitted"}</span> <span class="tag">${D.cats.age[k]} years · ${D.cats.gender[k]} · ${D.cats.admission_type[k]} admission</span>
        <span class="ptbtns"><button class="btn" id="exPrevP">← Previous</button> <button class="btn" id="exNextP">Next →</button> <button class="btn" id="exRandP">Random</button> <button class="btn pri" id="exToTry">Open in "Try a patient"</button></span></div>
        <div class="pred">${preds}</div>
        <div class="scroll"><table style="margin-top:12px"><tr><th>Variable</th><th class="n">Value</th><th>Percentile among all stays</th><th class="n"></th><th class="n">Logistic regression push</th><th>(left = away from readmission, right = toward)</th></tr>${vals}
        <tr><td colspan="4"><b>Intercept + all pushes = score</b></td><td class="n"><b>${(lrm.intercept + sum(contrib) >= 0 ? "+" : "") + (lrm.intercept + sum(contrib)).toFixed(2)}</b></td><td>log-odds, which becomes <b>${(lrProb * 100).toFixed(0)}%</b> chance of readmission</td></tr></table></div>`;
      $("#exPrevP", pt).onclick = () => stepPatient(-1); $("#exNextP", pt).onclick = () => stepPatient(1); $("#exRandP", pt).onclick = pickRandom;
      $("#exToTry", pt).onclick = () => { const v = rowv.slice(); location.hash = "#try"; setTimeout(() => { if (TRY_SET) TRY_SET(v); }, 80); };
      how(pt, "The three boxes show each model's predicted chance of readmission for this patient (0 to 100%), with a black mark at the threshold. In the table, the percentile says how this patient compares with all 3,000 stays, and the push column shows how many log-odds each variable added or removed in the logistic regression (coral pushes toward readmitted, teal pushes away). Age, gender and admission type are shown for context only and are not used in the models.");
      const top = contrib.map((v, i) => ({ v, i })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v))[0];
      says(pt, `Patient #${selP + 1} ${act ? "was" : "was not"} readmitted. The ${S.model.toLowerCase()} gave them ${(C.p[selP] * 100).toFixed(0)}%, so at a threshold of ${S.thr.toFixed(2)} it ${C.p[selP] >= S.thr ? "flagged" : "did not flag"} them, which is a ${CELL_NAME[mine]}. The biggest push in the logistic regression came from ${LBL[FEATS[top.i]].toLowerCase()} (${top.v >= 0 ? "toward" : "away from"} readmission), but every push is small.`);
    };
    reg(wrap); p.append(wrap);
    wrap.style.display = "none";
    return p;
  };
})();
