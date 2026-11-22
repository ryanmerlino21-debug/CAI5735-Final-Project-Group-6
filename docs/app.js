/* CAI5735 Final Project - Group 6 (Ryan Merlino, April Walter, Nikolas Platon)
   Everything on this page is computed in the browser from docs/data.json, which is written by scripts/export_results.py */
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const FEATS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses", "diabetes_med"];
const LBL = { time_in_hospital: "Days in hospital", num_lab_procedures: "Lab tests", num_procedures: "Procedures", num_medications: "Medications", number_diagnoses: "Diagnoses", diabetes_med: "Diabetes medication", readmitted: "Readmitted" };
const UNITS = { time_in_hospital: "Days in hospital (days)", num_lab_procedures: "Lab tests (count per stay)", num_procedures: "Procedures (count per stay)", num_medications: "Medications (count per stay)", number_diagnoses: "Diagnoses (count per stay)", diabetes_med: "Diabetes medication (0 = no, 1 = yes)" };
const axT = f => UNITS[f] || LBL[f];
const METRIC_T = { auc: "AUROC (0.5 = coin flip, 1 = perfect)", acc: "Accuracy (share of predictions correct, 0 to 1)", sens: "Sensitivity (share of readmissions caught, 0 to 1)", spec: "Specificity (share of non-readmissions cleared, 0 to 1)", prec: "Precision (share of flagged who returned, 0 to 1)", f1: "F1 score (0 to 1)", j: "Youden's J (sensitivity + specificity - 1)", flag: "Share of patients flagged (0 to 1)" };
let CUR_AXES = null, TRY_SET = null;
const clampT = t => Math.min(0.7, Math.max(0.1, +(+t).toFixed(2)));
function setThr(t) { S.thr = clampT(t); setTimeout(syncSettings, 0); } // deferred so a chart is not destroyed inside its own click handler
const MODELS = ["Logistic regression", "Decision tree", "Random forest"];
const MC = i => [css("--teal"), css("--coral"), css("--violet")][i];
const S = { tab: "overview", model: "Random forest", thr: 0.40 };
let D, ROC = {}, PR = {}, REG = {}, BUILT = {}, BOOT = null;

/* ---------------- small stats helpers ---------------- */
const sum = a => a.reduce((s, v) => s + v, 0), mean = a => sum(a) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(sum(a.map(v => (v - m) ** 2)) / (a.length - 1)); };
const quant = (a, q) => { const s = [...a].sort((x, y) => x - y), p = (s.length - 1) * q, i = Math.floor(p); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (p - i); };
const pearson = (a, b) => { const ma = mean(a), mb = mean(b); let n = 0, da = 0, db = 0; for (let i = 0; i < a.length; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; } return n / Math.sqrt(da * db); };
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const f3 = v => v.toFixed(3);
function confusion(p, y, t) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < y.length; i++) { const q = p[i] >= t ? 1 : 0; if (q && y[i]) tp++; else if (!q && !y[i]) tn++; else if (q) fp++; else fn++; }
  const sens = tp / (tp + fn), spec = tn / (tn + fp), prec = tp + fp ? tp / (tp + fp) : NaN;
  return { tp, tn, fp, fn, n: y.length, acc: (tp + tn) / y.length, sens, spec, prec, f1: prec ? 2 * prec * sens / (prec + sens) : 0, j: sens + spec - 1, flag: (tp + fp) / y.length };
}
function curves(p, y) {
  const idx = p.map((_, i) => i).sort((a, b) => p[b] - p[a]), P = sum(y), N = y.length - P;
  let tp = 0, fp = 0, auc = 0, prev = { x: 0, y: 0 }; const roc = [{ x: 0, y: 0, t: 1 }], pr = []; let ap = 0, prevR = 0;
  for (let k = 0; k < idx.length; k++) {
    if (y[idx[k]]) tp++; else fp++;
    if (k === idx.length - 1 || p[idx[k + 1]] !== p[idx[k]]) {
      const pt = { x: fp / N, y: tp / P, t: p[idx[k]] }; auc += (pt.x - prev.x) * (pt.y + prev.y) / 2; roc.push(pt); prev = pt;
      const prec = tp / (tp + fp); pr.push({ x: tp / P, y: prec }); ap += (tp / P - prevR) * prec; prevR = tp / P;
    }
  }
  return { roc, pr, auc, ap };
}
function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/* ---------------- model inference ---------------- */
const evalTree = (t, x) => { let n = 0; while (t.l[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n]; return t.p[n]; };
function predict(name, x) {
  if (name === MODELS[0]) { const m = D.lr; let z = m.intercept; for (let i = 0; i < x.length; i++) z += m.coef[i] * (x[i] - m.mean[i]) / m.scale[i]; return 1 / (1 + Math.exp(-z)); }
  if (name === MODELS[1]) return evalTree(D.tree, x);
  return D.forest.reduce((s, t) => s + evalTree(t, x), 0) / D.forest.length;
}

/* ---------------- DOM + chart helpers ---------------- */
function h(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; }
function mk(canvas, cfg) {
  const old = Chart.getChart(canvas); if (old) old.destroy();
  cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: false }, cfg.options || {});
  const A = CUR_AXES;
  if (A) {
    if (["doughnut", "pie", "radar"].includes(cfg.type)) { cfg.options.plugins = cfg.options.plugins || {}; if (A.t && !(cfg.options.plugins.title && cfg.options.plugins.title.display)) cfg.options.plugins.title = { display: true, text: A.t, font: { size: 12, weight: "600" } }; }
    else { cfg.options.scales = cfg.options.scales || {}; ["x", "y", "y2"].forEach(k => { if (A[k] === undefined) return; const sc = (cfg.options.scales[k] = cfg.options.scales[k] || {}); if (!sc.title || !sc.title.display) sc.title = { display: true, text: A[k] }; }); }
  }
  return new Chart(canvas, cfg);
}
function themeCharts() {
  Chart.defaults.color = css("--muted"); Chart.defaults.borderColor = css("--line");
  Chart.defaults.font.family = '"Avenir Next",Avenir,"Segoe UI",system-ui,sans-serif';
}
function reg(el) { (REG[S.tab] = REG[S.tab] || []).push(el); return el; }
function refresh() { (REG[S.tab] || []).forEach(e => e.redraw && e.redraw()); }

/* a chart (or html) card with toggles above it and a "how to read this" note below */
function card({ title, toggles = [], height = 300, kind = "canvas", caption = "", draw, wide }) {
  const el = h(`<div class="card"><div class="ch"><h4>${title}</h4><div class="tg"></div></div><div class="body"></div><p class="how"></p><p class="says"></p></div>`);
  const st = {}, tg = $(".tg", el), body = $(".body", el);
  toggles.forEach(t => {
    st[t.key] = t.value;
    if (t.type === "select") {
      const lab = h(`<label>${t.label}<select>${t.options.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></label>`);
      const sel = $("select", lab); sel.value = t.value;
      sel.onchange = () => {
        st[t.key] = sel.value;
        if (t.sync) (REG[S.tab] || []).forEach(o => { if (o !== el && o.syncSel && o.syncSel[t.sync]) { o.syncSel[t.sync](sel.value); } });
        el.redraw();
      };
      if (t.sync) { el.syncSel = el.syncSel || {}; el.syncSel[t.sync] = v => { sel.value = v; st[t.key] = v; el.redraw(); }; }
      tg.append(lab);
    } else if (t.type === "range") {
      const lab = h(`<label>${t.label} <input type="range" min="${t.min}" max="${t.max}" step="${t.step}" value="${t.value}"><b class="mono">${t.value}</b></label>`);
      const r = $("input", lab); r.oninput = () => { st[t.key] = +r.value; $("b", lab).textContent = r.value; el.redraw(); }; tg.append(lab);
    } else if (t.type === "button") {
      const b = h(`<button class="btn pri">${t.label}</button>`); b.onclick = () => t.onclick(st, el); tg.append(b);
    } else {
      const seg = h(`<span class="seg">${t.options.map(([v, l]) => `<button type="button" data-v="${v}" class="${v === t.value ? "on" : ""}">${l}</button>`).join("")}</span>`);
      if (t.label) tg.append(h(`<label>${t.label}</label>`));
      seg.onclick = e => { const b = e.target.closest("button"); if (!b) return; st[t.key] = b.dataset.v; $$("button", seg).forEach(x => x.classList.toggle("on", x === b)); el.redraw(); };
      tg.append(seg);
    }
  });
  if (kind === "canvas") body.innerHTML = `<div class="cbox" style="height:${height}px"><canvas></canvas></div>`;
  else if (kind === "both") body.innerHTML = `<div class="cbox" style="height:${height}px"><canvas></canvas></div><div class="alt" style="display:none"></div>`;
  el.st = st;
  el.redraw = () => {
    CUR_AXES = AXES[title] ? AXES[title](st) : null;
    draw(st, kind === "html" ? body : $("canvas", el), el);
    CUR_AXES = null;
    $(".how", el).innerHTML = caption ? "<b>How to read this.</b> " + (typeof caption === "function" ? caption(st, el) : caption) : "";
    if (!caption) $(".how", el).remove();
    const sf = SAYS[title], sp = $(".says", el);
    if (sf) { let t = ""; try { t = sf(st, el); } catch (e) { t = ""; } sp.innerHTML = t ? "<b>What it says.</b> " + t : ""; if (!t) sp.style.display = "none"; else sp.style.display = ""; } else sp.remove();
  };
  return reg(el);
}
/* for cards that can show either a canvas chart or an html view */
function showCanvas(el, on) { $(".cbox", el).style.display = on ? "" : "none"; $(".alt", el).style.display = on ? "none" : ""; if (!on) { const c = Chart.getChart($("canvas", el)); if (c) c.destroy(); } return $(".alt", el); }
function waffle(parts) {
  const tot = sum(parts.map(p => p.n)); let cells = parts.map(p => ({ ...p, k: Math.floor(p.n / tot * 100), r: (p.n / tot * 100) % 1 }));
  let left = 100 - sum(cells.map(c => c.k)); [...cells].sort((a, b) => b.r - a.r).slice(0, left).forEach(c => c.k++);
  return `<div class="waffle">${cells.map(c => `<i style="background:${c.color}" title="${c.label}"></i>`.repeat(c.k)).join("")}</div><div class="wleg">${cells.map(c => `<span style="--c:${c.color}">${c.label}: <b>${c.n.toLocaleString()}</b> (${pct(c.n / tot, 0)})</span>`).join("")}</div>`;
}
function spark(vals, color) {
  const lo = Math.min(...vals), hi = Math.max(...vals), nb = 12, w = (hi - lo + 1e-9) / nb, c = Array(nb).fill(0);
  vals.forEach(v => c[Math.min(nb - 1, Math.floor((v - lo) / w))]++); const m = Math.max(...c);
  return `<svg width="84" height="26" viewBox="0 0 84 26">${c.map((v, i) => `<rect x="${i * 7}" y="${26 - v / m * 24}" width="6" height="${v / m * 24}" rx="1.5" fill="${color}"/>`).join("")}</svg>`;
}
function boxSVG(rows, lo, hi, axisLabel) {
  const W = 640, L = 130, R = 20, rh = 56, H = rows.length * rh + 40, X = v => L + (v - lo) / (hi - lo || 1) * (W - L - R);
  const ticks = Array.from({ length: 6 }, (_, i) => lo + (hi - lo) * i / 5);
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:760px;display:block;margin:auto">${ticks.map(t => `<line x1="${X(t)}" x2="${X(t)}" y1="6" y2="${H - 30}" stroke="${css("--line")}"/><text x="${X(t)}" y="${H - 14}" text-anchor="middle">${Math.round(t * 10) / 10}</text>`).join("")}
  ${rows.map((r, i) => { const y = 20 + i * rh, q = r.q; return `<text x="${L - 10}" y="${y + 6}" text-anchor="end" style="fill:${css("--ink")};font-weight:700;font-size:12px">${r.label}</text>
    <line x1="${X(q[0])}" x2="${X(q[4])}" y1="${y}" y2="${y}" stroke="${r.color}" stroke-width="3"/><line x1="${X(q[0])}" x2="${X(q[0])}" y1="${y - 8}" y2="${y + 8}" stroke="${r.color}" stroke-width="3"/><line x1="${X(q[4])}" x2="${X(q[4])}" y1="${y - 8}" y2="${y + 8}" stroke="${r.color}" stroke-width="3"/>
    <rect x="${X(q[1])}" y="${y - 14}" width="${Math.max(2, X(q[3]) - X(q[1]))}" height="28" rx="6" fill="${r.color}" fill-opacity=".35" stroke="${r.color}" stroke-width="2"/><line x1="${X(q[2])}" x2="${X(q[2])}" y1="${y - 14}" y2="${y + 14}" stroke="${css("--ink")}" stroke-width="3"/>
    <circle cx="${X(r.mean)}" cy="${y}" r="4.5" fill="${css("--sun")}" stroke="${css("--ink")}"/>`; }).join("")}
  <text x="${(L + W - R) / 2}" y="${H - 1}" text-anchor="middle">${axisLabel}</text></svg>`;
}
function dyn(fn, cls = "") { const el = h(`<div class="${cls}"></div>`); el.redraw = () => { el.innerHTML = fn(); }; return reg(el); }
function section(title, ...kids) { const s = h(`<section class="block"><h3>${title}</h3></section>`); kids.forEach(k => k && s.append(typeof k === "string" ? h(`<div>${k}</div>`) : k)); return s; }
const para = t => h(`<p>${t}</p>`);
const lede = t => h(`<p class="lede">${t}</p>`);
function titlebar(t, sub) { return h(`<div class="titlebar"><h2>${t}</h2><p>${sub}</p><span class="tip">💡 Use the buttons above each chart to switch how it is drawn.</span></div>`); }
function kpis(items) { return h(`<div class="kpis k${items.length}">${items.map(([l, v, s]) => `<div class="kpi"><small>${l}</small><b>${v}</b><span>${s || ""}</span></div>`).join("")}</div>`); }

/* ---------------- shared global controls (model + threshold) ---------------- */
function settingsBar() {
  const el = h(`<div class="settings">
    <label>Model <select class="gm">${MODELS.map(m => `<option>${m}</option>`).join("")}</select></label>
    <label>Threshold <input class="gt" type="range" min="0.10" max="0.70" step="0.01"><output class="gv"></output></label>
    <span><button class="btn" data-t="0.5">Default 0.50</button> <button class="btn" data-t="0.4">Our choice 0.40</button> <button class="btn" data-t="best">Best balance</button></span>
  </div>`);
  const upd = () => { $(".gm", el).value = S.model; $(".gt", el).value = S.thr; $(".gv", el).textContent = S.thr.toFixed(2); };
  $(".gm", el).onchange = e => { S.model = e.target.value; syncSettings(); };
  $(".gt", el).oninput = e => { S.thr = +e.target.value; syncSettings(); };
  el.onclick = e => { const b = e.target.closest("button[data-t]"); if (!b) return; S.thr = b.dataset.t === "best" ? bestJ(S.model) : +b.dataset.t; syncSettings(); };
  el.upd = upd; upd(); return el;
}
function bestJ(m) { let b = 0.5, bj = -9; for (let t = 0.1; t <= 0.7001; t += 0.01) { const j = confusion(D.probs[m], D.y_test, t).j; if (j > bj) { bj = j; b = +t.toFixed(2); } } return b; }
function syncSettings() { $$(".settings").forEach(s => s.upd()); refresh(); }

/* ---------------- binning for the explore charts ---------------- */
function makeBins(v, mode) {
  const vals = D.data[v], u = [...new Set(vals)].sort((a, b) => a - b);
  if (v === "diabetes_med") return { labels: ["No", "Yes"], idx: vals.map(x => x) };
  if (mode === "coarse") {
    const ups = [...new Set([1, 2, 3, 4, 5].map(i => Math.round(quant(vals, i / 5))))].sort((a, b) => a - b);
    const idx = vals.map(x => ups.findIndex(u2 => x <= u2));
    const labels = ups.map((u2, j) => (j === 0 ? `≤${u2}` : `${ups[j - 1] + 1}-${u2}`));
    return { labels, idx };
  }
  if (u.length <= 16) return { labels: u.map(String), idx: vals.map(x => u.indexOf(x)) };
  const lo = u[0], hi = u[u.length - 1], w = Math.ceil((hi - lo + 1) / 20), nb = Math.floor((hi - lo) / w) + 1;
  return { labels: Array.from({ length: nb }, (_, i) => (w === 1 ? String(lo + i) : `${lo + i * w}-${lo + i * w + w - 1}`)), idx: vals.map(x => Math.floor((x - lo) / w)) };
}
const VAR_OPTS = FEATS.map(f => [f, LBL[f]]);


/* axis titles with units, applied by mk() whenever a chart does not set its own */
const AXES = {
  "Model comparison": st => st.ch === "radar" ? { t: "Score on each metric (0 to 1)" } : st.ch === "bar" ? { x: METRIC_T[st.metric], y: "Model" } : st.ch === "dots" ? { x: `${METRIC_T[st.metric]}, axis zoomed in`, y: "Model" } : null,
  "Outcome groups": st => st.ch === "bar" ? { x: "Outcome group", y: "Number of stays (count)" } : { t: "Number of stays in each group (count)" },
  "Split sizes and readmission rate": () => ({ y: "Data set", x: undefined }),
  "Distribution": () => ({ y: "Share of group (% of stays)" }),
  "Readmission rate from lowest to highest values": st => st.t === "radar" ? { t: "Readmission rate (% of stays) in each group" } : { x: "Group of the variable (lowest to highest, equal-size groups)", y: "Readmission rate (% of stays)" },
  "Context variables": st => ({ t: "Number of stays in each group (count)", x: { age: "Age group (years)", gender: "Gender", admission_type: "Admission type" }[st.v], y: st.t === "rate" ? "Readmission rate (% of stays)" : st.t === "n" ? "Number of stays (count)" : "Share of group (%)" }),
  "Readmitted vs not readmitted": st => st.v === "d" ? { y: "Variable", x: "Effect size (standard deviations, readmitted minus not)" } : st.v === "g" ? { x: "Variable", y: "Group average as an index (overall average = 100)" } : { t: "Group average as an index (overall average = 100)" },
  "Variable importance": st => st.t === "radar" ? { t: "Share of importance (%) for each variable" } : st.t === "all" ? { x: "Variable", y: "Share of importance (%)" } : { y: "Variable" },
  "Bootstrap check of the AUROC": () => ({ y: "Model" }),
  "Metrics at every threshold": st => ({ y: st.t === "stack" ? "Share of test patients (%)" : st.m === "ss" ? "Sensitivity or specificity (0 to 1)" : METRIC_T[st.m] }),
  "Why did the logistic regression say that?": st => st.v === "r" ? { t: "Percentile of this patient among all stays (0 to 100)" } : { y: "Variable" },
  "What if? Change one variable and watch the prediction": () => ({ y: "Predicted chance of readmission (%)" }),
  "Default threshold vs our threshold": st => st.t === "radar" ? { t: "Score on each metric (0 to 1)" } : { x: "Model", y: METRIC_T[st.m] },
  "Readmission rate by value": () => ({}),
  "Scatter plot": () => ({})
};
/* ---------------- "What it says": a plain language reading of each chart, with live numbers ---------------- */
let PATIENT = null;
const dWord = d => { d = Math.abs(d); return d < 0.1 ? "almost no difference" : d < 0.2 ? "a very small difference" : d < 0.5 ? "a small difference" : "a clear difference"; };
const yv = () => D.readmitted;
const cohen = f => { const v = D.data[f], y = yv(), a = v.filter((_, i) => y[i]), b = v.filter((_, i) => !y[i]); return (mean(a) - mean(b)) / Math.sqrt((sd(a) ** 2 + sd(b) ** 2) / 2); };
const binRates = (v, mode) => { const { labels, idx } = makeBins(v, mode), n = Array(labels.length).fill(0), r = Array(labels.length).fill(0); idx.forEach((b, i) => { n[b]++; r[b] += yv()[i]; }); return { labels, n, rate: r.map((x, i) => n[i] ? x / n[i] * 100 : null) }; };
const SAYS = {
  "Model comparison": st => {
    const t = st.at === "half" ? 0.5 : S.thr, get = (m, k) => k === "auc" ? ROC[m].auc : confusion(D.probs[m], D.y_test, t)[k], k = st.ch === "bar" || st.ch === "dots" ? st.metric : "auc";
    const vals = MODELS.map(m => get(m, k)), hi = Math.max(...vals), lo = Math.min(...vals), best = MODELS[vals.indexOf(hi)], nm = { auc: "AUROC", acc: "accuracy", sens: "sensitivity", spec: "specificity", prec: "precision", f1: "F1 score" }[k];
    let msg = `On ${nm}${k === "auc" ? "" : ` (threshold ${t.toFixed(2)})`} the ${best.toLowerCase()} scores highest at ${f3(hi)}, and the lowest model scores ${f3(lo)}, a gap of only ${f3(hi - lo)}. `;
    if (k === "auc") msg += "All three are about 0.59, only a little above the 0.50 of a coin flip, so none of them is a strong predictor.";
    else if (k === "sens") msg += hi < 0.5 ? "Every model catches fewer than half of the patients who really come back." : "At this threshold the models catch a good share of readmissions, but at the cost of more false alarms.";
    else if (k === "spec") msg += hi > 0.75 ? "High specificity here mostly comes from the models predicting not readmitted for almost everyone." : "Lower specificity means more patients who did not come back get flagged.";
    else msg += "The models are close enough that the choice between them is a judgment call.";
    return msg;
  },
  "Outcome groups": st => { const L = D.readmitted_levels, n = yv().length, r = L["After 30 days"] + L["Within 30 days"]; return `${r.toLocaleString()} of ${n.toLocaleString()} stays (${pct(r / n, 0)}) were followed by a return to the hospital, and ${L["Not readmitted"].toLocaleString()} (${pct(L["Not readmitted"] / n, 0)}) were not. ${st.lv === "three" ? `Only ${L["Within 30 days"]} stays (${pct(L["Within 30 days"] / n, 0)}) were readmitted within 30 days, which is why we combined the two readmitted groups.` : "Because the groups are not wildly uneven, accuracy is a fair measure, but we have to beat 60% to do better than always guessing not readmitted."}`; },
  "Data preview": st => `The file has ${yv().length.toLocaleString()} rows like these and ${pct(mean(yv()), 0)} of them are readmitted. ${st.f === "all" ? "Try filtering to Readmitted and then Not readmitted: the rows look very alike, which is an early sign that the variables will not separate the two groups easily." : "Compare these rows with the other filter. It is hard to see any pattern by eye, which previews what the models find."}`,
  "Split sizes and readmission rate": st => { const pt = sum(D.y_test), ptr = sum(yv()) - pt; return `The training set has ${D.n_train.toLocaleString()} stays (${pct(ptr / D.n_train, 1)} readmitted) and the test set has ${D.n_test.toLocaleString()} stays (${pct(pt / D.n_test, 1)} readmitted). The rates are almost identical, so the test set is a fair stand-in for new patients.`; },
  "Distribution": st => {
    const v = D.data[st.v], y = yv(), a = v.filter((_, i) => !y[i]), b = v.filter((_, i) => y[i]), d = cohen(st.v);
    if (st.v === "diabetes_med") return `${pct(mean(a), 0)} of patients who were not readmitted and ${pct(mean(b), 0)} of those who were readmitted were prescribed a diabetes medication, ${dWord(d)}.`;
    return `The typical (median) ${LBL[st.v].toLowerCase()} is ${quant(a, .5)} for not readmitted and ${quant(b, .5)} for readmitted, and the averages are ${mean(a).toFixed(1)} and ${mean(b).toFixed(1)}. That is ${dWord(d)} (effect size ${d.toFixed(2)}). The two groups overlap almost completely, so this variable alone cannot tell them apart.`;
  },
  "Readmission rate by value": st => {
    const { labels, n, rate } = binRates(st.v, st.d), ok = rate.map((r, i) => n[i] >= 30 ? i : -1).filter(i => i >= 0);
    if (!ok.length) return "";
    const hiI = ok.reduce((a, b) => rate[b] > rate[a] ? b : a), loI = ok.reduce((a, b) => rate[b] < rate[a] ? b : a), r = pearson(D.data[st.v], yv());
    return `Among groups with at least 30 stays, the readmission rate is highest at ${labels[hiI]} (${rate[hiI].toFixed(0)}%) and lowest at ${labels[loI]} (${rate[loI].toFixed(0)}%), compared with ${pct(mean(yv()), 0)} overall. The overall trend is ${Math.abs(r) < 0.05 ? "essentially flat" : r > 0 ? "a slight rise" : "a slight fall"} (correlation ${r.toFixed(2)}), so ${LBL[st.v].toLowerCase()} gives only a weak hint about readmission.`;
  },
  "Readmission rate from lowest to highest values": st => {
    const k = +st.k, rows = FEATS.slice(0, 5).map(f => { const idx = quantBins(f, k), r = Array(k).fill(0), n = Array(k).fill(0); idx.forEach((b, i) => { n[b]++; r[b] += yv()[i]; }); const last = n.map((v, i) => v ? i : -1).filter(i => i >= 0), a = r[last[0]] / n[last[0]] * 100, z = r[last[last.length - 1]] / n[last[last.length - 1]] * 100; return { f, a, z, d: z - a }; });
    const big = rows.reduce((a, b) => Math.abs(b.d) > Math.abs(a.d) ? b : a);
    return `The biggest change from the lowest group to the highest group is for ${LBL[big.f].toLowerCase()}: the rate goes from ${big.a.toFixed(0)}% to ${big.z.toFixed(0)}% (${big.d >= 0 ? "+" : ""}${big.d.toFixed(0)} points). Even that is modest, and the other variables move less, so no variable gives a steep slope.`;
  },
  "Readmission rate by two variables": st => {
    const k = +st.k, X = quantInfo(st.x, k), Y = quantInfo(st.y, k), cells = {};
    yv().forEach((v, i) => { const key = Y.idx[i] + "," + X.idx[i]; (cells[key] = cells[key] || { n: 0, r: 0, j: Y.idx[i], i: X.idx[i] }); cells[key].n++; cells[key].r += v; });
    const list = Object.values(cells).filter(c => c.n >= 20).map(c => ({ ...c, q: c.r / c.n * 100 })); if (!list.length) return "";
    const hi = list.reduce((a, b) => b.q > a.q ? b : a), lo = list.reduce((a, b) => b.q < a.q ? b : a);
    return `Among cells with at least 20 stays, the highest rate is ${hi.q.toFixed(0)}% (${LBL[st.x].toLowerCase()} ${X.labels[hi.i]}, ${LBL[st.y].toLowerCase()} ${Y.labels[hi.j]}, n = ${hi.n}) and the lowest is ${lo.q.toFixed(0)}% (${LBL[st.x].toLowerCase()} ${X.labels[lo.i]}, ${LBL[st.y].toLowerCase()} ${Y.labels[lo.j]}, n = ${lo.n}). Combining two variables spreads the rates out a little more than one variable does, but small cells are noisy, so we should not read too much into any single cell.`;
  },
  "Context variables": st => {
    const vals = D.cats[st.v], groups = [...new Set(vals)], rows = groups.map(g => ({ g, n: vals.filter(v => v === g).length, r: mean(yv().filter((_, i) => vals[i] === g)) * 100 })), hi = rows.reduce((a, b) => b.r > a.r ? b : a), lo = rows.reduce((a, b) => b.r < a.r ? b : a), big = rows.reduce((a, b) => b.n > a.n ? b : a);
    return `The largest group is ${big.g} (${pct(big.n / vals.length, 0)} of stays). The readmission rate is highest for ${hi.g} (${hi.r.toFixed(0)}%) and lowest for ${lo.g} (${lo.r.toFixed(0)}%), a gap of ${(hi.r - lo.r).toFixed(1)} points. We did not use this variable in the models, but it suggests the groups come back at similar rates.`;
  },
  "Correlation heatmap": () => {
    const rows = FEATS.map(f => ({ f, r: pearson(D.data[f], yv()) })), top = rows.reduce((a, b) => Math.abs(b.r) > Math.abs(a.r) ? b : a);
    let pair = { r: 0 }; for (let i = 0; i < 5; i++) for (let j = i + 1; j < 6; j++) { const r = pearson(D.data[FEATS[i]], D.data[FEATS[j]]); if (Math.abs(r) > Math.abs(pair.r)) pair = { r, a: FEATS[i], b: FEATS[j] }; }
    return `The strongest link to readmission is ${LBL[top.f].toLowerCase()} at r = ${top.r.toFixed(2)}, which is very weak (0 is no link and 1 is perfect). The strongest link between two predictors is ${LBL[pair.a].toLowerCase()} and ${LBL[pair.b].toLowerCase()} at r = ${pair.r.toFixed(2)}. So the variables carry little information about readmission on their own.`;
  },
  "Scatter plot": st => {
    if (st.x === st.y) return `X and Y are the same variable, so the dots fall on a straight line. Pick two different variables to compare them.`;
    const r = pearson(D.data[st.x], D.data[st.y]);
    return `${LBL[st.x]} and ${LBL[st.y].toLowerCase()} have a correlation of ${r.toFixed(2)}, so ${Math.abs(r) > 0.4 ? "they clearly rise together" : Math.abs(r) > 0.15 ? "they rise together a little" : "they are mostly unrelated"}. ${st.c === "on" ? "The coral (readmitted) and teal (not readmitted) dots are mixed together everywhere, so no region of this plot is mostly one color." : "Turn on the color option to check whether readmitted patients cluster anywhere."}`;
  },
  "Readmitted vs not readmitted": st => {
    const rows = FEATS.map(f => ({ f, d: cohen(f) })), big = rows.reduce((a, b) => Math.abs(b.d) > Math.abs(a.d) ? b : a);
    return `The largest difference is for ${LBL[big.f].toLowerCase()}, with an effect size of ${big.d.toFixed(2)} (${dWord(big.d)}). ${rows.every(r => Math.abs(r.d) < 0.3) ? "Every variable is below 0.3, which is small, so readmitted and not readmitted patients look alike on all six." : ""}`;
  },
  "Confusion matrix": () => {
    const c = confusion(D.probs[S.model], D.y_test, S.thr);
    return `At a threshold of ${S.thr.toFixed(2)}, the ${S.model.toLowerCase()} correctly flagged ${c.tp} of the ${c.tp + c.fn} patients who came back (sensitivity ${pct(c.sens, 0)}) and correctly cleared ${c.tn} of the ${c.tn + c.fp} who did not (specificity ${pct(c.spec, 0)}). It missed ${c.fn} readmissions and raised ${c.fp} false alarms. Overall ${pct(c.acc, 0)} of its predictions were right.`;
  },
  "Performance curve": st => {
    const m = S.model, i = D.probs[m].map((_, k) => k).sort((a, b) => D.probs[m][b] - D.probs[m][a]), n = i.length, P = sum(D.y_test), base = P / n;
    const top = f => i.slice(0, Math.round(n * f)).reduce((s, k) => s + D.y_test[k], 0);
    if (st.c === "roc") return `The AUROC values are ${MODELS.map(x => `${x} ${f3(ROC[x].auc)}`).join(", ")}. A coin flip scores 0.50 and a perfect model scores 1.00, so all three models are barely above chance. The curves lie close to the diagonal, which means that whenever the model catches more readmissions it also raises about as many extra false alarms.`;
    if (st.c === "pr") return `Flagging everyone would give a precision of ${pct(base, 0)} because that is the share of readmitted patients. Our models stay only a little above that line (average precision ${MODELS.map(x => f3(PR[x].ap)).join(", ")}), so the patients they flag are only slightly more likely to be readmitted than a random patient.`;
    if (st.c === "gain") return `If staff called the top 20% of patients by predicted risk from the ${m.toLowerCase()}, they would reach ${pct(top(.2) / P, 0)} of all readmissions (random calling would reach 20%). Calling the top 40% would reach ${pct(top(.4) / P, 0)} (random would reach 40%). The model helps a little, but not by much.`;
    if (st.c === "lift") return `For the ${m.toLowerCase()}, the top 20% of patients by predicted risk are readmitted ${((top(.2) / (n * .2)) / base).toFixed(2)} times as often as the average patient (a lift of 1.00 would mean no help). The lift fades toward 1 as we call more patients.`;
    const pr = D.probs[m]; return `The ${m} predicts probabilities only between ${pct(Math.min(...pr), 0)} and ${pct(Math.max(...pr), 0)}, while real patients are either readmitted or not. A model that was confident and right would spread from near 0 to near 1. The narrow range shows the model has little to go on.`;
  },
  "Predicted probabilities for the selected model": () => {
    const pr = D.probs[S.model], y = D.y_test, a = mean(pr.filter((_, i) => !y[i])), b = mean(pr.filter((_, i) => y[i]));
    return `The ${S.model} gives patients who were not readmitted an average predicted risk of ${pct(a, 1)} and patients who were readmitted ${pct(b, 1)}, a difference of only ${((b - a) * 100).toFixed(1)} points. The two shapes overlap almost completely, so no threshold can cleanly split them.`;
  },
  "Comparison table": () => {
    const row = k => MODELS.reduce((a, b) => (k === "auc" ? ROC[b].auc > ROC[a].auc : confusion(D.probs[b], D.y_test, S.thr)[k] > confusion(D.probs[a], D.y_test, S.thr)[k]) ? b : a);
    return `At a threshold of ${S.thr.toFixed(2)}: the best accuracy is ${row("acc")}, the best sensitivity is ${row("sens")}, the best specificity is ${row("spec")} and the best AUROC is ${row("auc")}. No model wins everything, which shows the models trade one strength for another instead of one being clearly better.`;
  },
  "Variable importance": st => {
    const imp = m => { const a = m === MODELS[0] ? D.lr.coef.map(Math.abs) : D.importance[m], t = sum(a); return a.map(v => v / t * 100); };
    const tops = MODELS.map(m => { const a = imp(m), j = a.indexOf(Math.max(...a)); return { m, f: LBL[FEATS[j]], v: a[j] }; });
    return st.t === "all" ? `All three models rely most on ${tops.every(t => t.f === tops[0].f) ? tops[0].f.toLowerCase() : "different variables"} (${tops.map(t => `${t.m}: ${t.f.toLowerCase()} ${t.v.toFixed(0)}%`).join("; ")}). The rest of the weight is spread across the other variables, so no single variable drives the predictions.` : `The ${S.model} leans most on ${tops[MODELS.indexOf(S.model)].f.toLowerCase()} (${tops[MODELS.indexOf(S.model)].v.toFixed(0)}% of its importance). Importance only shows what the model used, not that the variable causes readmission.`;
  },
  "Inside the models": st => {
    const T = D.tree, f = T.f[0];
    if (st.v === "lr") { const o = FEATS.map((x, i) => ({ x, or: Math.exp(D.lr.coef[i] / D.lr.scale[i]) })), big = o.slice(0, 5).reduce((a, b) => Math.abs(Math.log(b.or)) > Math.abs(Math.log(a.or)) ? b : a); return `The largest per unit effect among the counts is ${LBL[big.x].toLowerCase()}, with an odds ratio of ${big.or.toFixed(2)}: each extra unit changes the odds of readmission by about ${Math.abs((big.or - 1) * 100).toFixed(0)}%. The prescription variable has an odds ratio of ${o[5].or.toFixed(2)}.`; }
    return `The tree's first question is "${LBL[FEATS[f]].toLowerCase()} ≤ ${Math.floor(T.t[0])}?", which splits ${T.n[0].toLocaleString()} training stays into ${T.n[T.l[0]]} and ${T.n[T.r[0]]}. Its leaves range from ${pct(Math.min(...T.p.filter((_, i) => T.l[i] === -1)), 0)} to ${pct(Math.max(...T.p.filter((_, i) => T.l[i] === -1)), 0)} readmitted, so even the tree's best splits only move the odds a little.`;
  },
  "Bootstrap check of the AUROC": () => BOOT ? `The random forest beat logistic regression in ${pct(BOOT.rfBeatsLr, 0)} of 300 resamples. The middle 95% ranges are ${MODELS.map((m, i) => `${m} ${f3(BOOT.ci[i][0])} to ${f3(BOOT.ci[i][1])}`).join("; ")}. They overlap heavily, so we cannot say one model is truly better than another from this test set.` : "Press the button to see how much the AUROC would change by chance.",
  "Metrics at every threshold": st => {
    const m = S.model, c = t => confusion(D.probs[m], D.y_test, t), a = c(0.3), b = c(0.5), j = bestJ(m), cj = c(j);
    return `For the ${m.toLowerCase()}, moving the threshold from 0.30 to 0.50 drops sensitivity from ${pct(a.sens, 0)} to ${pct(b.sens, 0)} and raises specificity from ${pct(a.spec, 0)} to ${pct(b.spec, 0)}. Flagged patients go from ${pct(a.flag, 0)} to ${pct(b.flag, 0)} of everyone. The best balance (Youden's J = ${cj.j.toFixed(3)}) is at a threshold of ${j.toFixed(2)}, with sensitivity ${pct(cj.sens, 0)} and specificity ${pct(cj.spec, 0)}.`;
  },
  "Expected cost per 100 patients": st => {
    const m = S.model, P = sum(D.y_test), n = D.y_test.length, cost = t => { const c = confusion(D.probs[m], D.y_test, t); return (c.fn * st.r + c.fp) / c.n * 100; };
    let bt = .1, bc = 1e9; for (let t = .1; t <= .7001; t += .01) if (cost(t) < bc) { bc = cost(t); bt = t; }
    const none = P * st.r / n * 100, every = (n - P) / n * 100;
    return `When a missed readmission costs ${st.r} false alarms, the cheapest threshold for the ${m.toLowerCase()} is ${bt.toFixed(2)}, at a cost of ${bc.toFixed(0)} per 100 patients. Flagging nobody would cost ${none.toFixed(0)} and flagging everybody would cost ${every.toFixed(0)}. ${bc < Math.min(none, every) - 1 ? "So the model does save some cost compared with the two simple options." : "So the model saves little or nothing compared with a simple rule like flagging everybody."}`;
  },
  "Selected model at common thresholds": () => {
    const a = confusion(D.probs[S.model], D.y_test, .4), b = confusion(D.probs[S.model], D.y_test, .5);
    return `At 0.40 the ${S.model.toLowerCase()} catches ${pct(a.sens, 0)} of readmissions and flags ${pct(a.flag, 0)} of patients. At 0.50 it catches ${pct(b.sens, 0)} and flags ${pct(b.flag, 0)}. That is why we chose 0.40: it finds far more of the patients who come back, at the cost of more false alarms.`;
  },
  "Why did the logistic regression say that?": st => {
    const x = PATIENT(), z = x.map((v, i) => (v - D.lr.mean[i]) / D.lr.scale[i]), c = z.map((v, i) => v * D.lr.coef[i]), j = c.map(Math.abs).indexOf(Math.max(...c.map(Math.abs))), p = predict(MODELS[0], x);
    return `For this patient the logistic regression predicts a ${pct(p, 0)} chance of readmission. The biggest push comes from ${LBL[FEATS[j]].toLowerCase()} (${c[j] >= 0 ? "toward" : "away from"} readmission by ${Math.abs(c[j]).toFixed(2)} on the log-odds scale). All of the pushes are small, which is why the prediction stays near the average of 40%.`;
  },
  "Where this patient sits in the data": () => {
    const x = PATIENT(), hi = FEATS.slice(0, 5).filter((f, i) => D.data[f].filter(a => a <= x[i]).length / D.data[f].length > .75).map(f => LBL[f].toLowerCase()), lo = FEATS.slice(0, 5).filter((f, i) => D.data[f].filter(a => a <= x[i]).length / D.data[f].length < .25).map(f => LBL[f].toLowerCase());
    return `${hi.length ? `This patient is in the top quarter for ${hi.join(", ")}. ` : ""}${lo.length ? `They are in the bottom quarter for ${lo.join(", ")}. ` : ""}${!hi.length && !lo.length ? "This patient is close to the middle of the data on every variable, so they look like a typical stay. " : ""}Patients at the extremes get the models' most extreme predictions.`;
  },
  "What if? Change one variable and watch the prediction": st => {
    const j = FEATS.indexOf(st.v), [lo, hi] = CFG[st.v], b = PATIENT(), at = (m, v) => { const x = b.slice(); x[j] = v; return predict(m, x) * 100; };
    return `Moving ${LBL[st.v].toLowerCase()} from ${lo} to ${hi} changes the predicted chance of readmission from ${at(MODELS[0], lo).toFixed(0)}% to ${at(MODELS[0], hi).toFixed(0)}% for logistic regression and from ${at(MODELS[2], lo).toFixed(0)}% to ${at(MODELS[2], hi).toFixed(0)}% for the random forest. ${Math.abs(at(MODELS[2], hi) - at(MODELS[2], lo)) < 25 ? "Even a huge change in this one variable moves the prediction only modestly." : "This variable has a noticeable effect on the prediction."}`;
  },
  "Default threshold vs our threshold": st => {
    const a = confusion(D.probs[S.model], D.y_test, .5), b = confusion(D.probs[S.model], D.y_test, .4);
    return `For the ${S.model.toLowerCase()}, lowering the threshold from 0.50 to 0.40 raises sensitivity from ${pct(a.sens, 0)} to ${pct(b.sens, 0)} but lowers specificity from ${pct(a.spec, 0)} to ${pct(b.spec, 0)} and accuracy from ${pct(a.acc, 0)} to ${pct(b.acc, 0)}. We accept the lower accuracy because catching readmissions matters more than avoiding extra follow up calls.`;
  }
};

/* ================= PAGES ================= */
const PAGES = {};

/* ---------- 1. overview ---------- */
PAGES.overview = () => {
  const p = h("<div></div>");
  p.append(titlebar("Research question", "The question we asked, the problem it addresses, why it matters, and what we found."));
  p.append(h(`<div class="callout"><h4>Start here: what this app is for and where to begin</h4>
    <div>We asked whether a few numbers recorded during a hospital stay can predict if a patient with diabetes comes back to the hospital. This app walks through how we answered that.</div>
    <ol>
      <li><a href="#dataset" data-go>Dataset</a>: where the data came from and what each column means.</li>
      <li><a href="#explorer" data-go>Interactive explorer</a>: one linked dashboard. Filter patients, move the threshold and pick a patient, and every chart updates together.</li>
      <li><a href="#explore" data-go>Explore</a>: charts to look at each variable and how it relates to readmission.</li>
      <li><a href="#models" data-go>Models</a>: the three classification models, with confusion matrices, ROC curves and more.</li>
      <li><a href="#threshold" data-go>Threshold</a>: how the cutoff changes what the model catches and misses.</li>
      <li><a href="#try" data-go>Try a patient</a>: enter a made up patient and see each model's prediction.</li>
      <li><a href="#results" data-go>Results</a>: our answer to the research question.</li>
    </ol></div>`));
  p.append(section("How we did it", h(`<div class="steps">
    <div class="step"><div class="ic">📥</div><b>1. Load the data</b><span>3,000 hospital stays from the UCI Diabetes 130-US Hospitals data.</span></div>
    <div class="step"><div class="ic">🎯</div><b>2. Pick variables</b><span>Six predictors from the stay. The response is readmitted or not.</span></div>
    <div class="step"><div class="ic">✂️</div><b>3. Split</b><span>75% to train the models and 25% to test them.</span></div>
    <div class="step"><div class="ic">🤖</div><b>4. Train 3 models</b><span>Logistic regression, decision tree and random forest.</span></div>
    <div class="step"><div class="ic">📏</div><b>5. Evaluate</b><span>Confusion matrix, accuracy, sensitivity, specificity, ROC and AUROC.</span></div>
    <div class="step"><div class="ic">🎚️</div><b>6. Choose</b><span>Pick the best model and where to set the threshold.</span></div></div>`)));
  p.append(section("Research question",
    h(`<div class="card" style="border-left:6px solid var(--teal);font-size:1.1rem">How accurately can a diabetes patient's length of stay, number of lab tests, procedures, medications and diagnoses, plus whether they were prescribed a diabetes medication, predict whether they will be readmitted to the hospital?</div>`)));
  p.append(section("What problem are we trying to solve?",
    para("When a patient with diabetes leaves the hospital and then has to come back, it is hard on the patient and it is costly for the hospital. Hospitals in the US are even penalized when too many patients are readmitted. If we could flag high risk patients before they go home, staff could focus follow up calls and medication reviews on those patients first."),
    para("We chose this question because it is a real problem in healthcare and because the answer is not obvious. Which means that better care could go to the people who need it most, but only if the prediction is good enough to trust. In this project we test how far a few simple numbers from the stay can take us.")));
  p.append(section("The numbers at a glance", dyn(() => {
    const best = MODELS.reduce((a, b) => ROC[a].auc > ROC[b].auc ? a : b), c = confusion(D.probs[best], D.y_test, 0.5), c4 = confusion(D.probs[best], D.y_test, 0.4);
    const rate = sum(D.readmitted) / D.readmitted.length;
    return kpis([["Hospital stays", D.readmitted.length.toLocaleString(), "130 US hospitals"], ["Readmitted", pct(rate, 0), "of all stays"], ["Best AUROC", f3(ROC[best].auc), best], ["Always guess no", pct(1 - rate, 0), "baseline accuracy"], ["Catch rate at 0.40", pct(c4.sens, 0), "sensitivity, " + best]]).outerHTML;
  })));
  p.append(section("How do the three models compare?", cmpCard()));
  p.append(section("Our main findings", dyn(() => {
    const top = D.importance["Random forest"].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])[0][1];
    return `<div class="card"><ol>
      <li><b>The models are only slightly better than guessing.</b> All three have an AUROC of about 0.59, where 0.5 is a coin flip and 1.0 is perfect.</li>
      <li><b>The random forest is our pick, but it barely wins.</b> The three models are within test-set noise of each other (see the bootstrap on the Models tab).</li>
      <li><b>${LBL[FEATS[top]]} mattered most</b> to the random forest, but no single variable separates the two groups well.</li>
      <li><b>A threshold of 0.40 works better than 0.50</b> because we care more about catching patients who come back.</li></ol></div>`;
  })));
  return p;
};

/* a reusable model comparison bar chart */
function cmpCard() {
  return card({
    title: "Model comparison", height: 300, kind: "both",
    toggles: [
      { key: "ch", label: "Show as", value: "bar", options: [["bar", "Bars"], ["radar", "Radar (all metrics)"], ["dots", "Dots"], ["tab", "Table"]] },
      { key: "metric", label: "Metric", type: "select", value: "auc", options: [["auc", "AUROC"], ["acc", "Accuracy"], ["sens", "Sensitivity"], ["spec", "Specificity"], ["prec", "Precision"], ["f1", "F1 score"]] },
      { key: "at", label: "Threshold", value: "cur", options: [["cur", "Current"], ["half", "0.50"]] }],
    draw(st, cv, el) {
      const t = st.at === "half" ? 0.5 : S.thr; st._t = t;
      const mt = m => confusion(D.probs[m], D.y_test, t), keys = ["auc", "acc", "sens", "spec", "prec", "f1"], names = ["AUROC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1"];
      const get = (m, k) => k === "auc" ? ROC[m].auc : mt(m)[k];
      if (st.ch === "tab") { const alt = showCanvas(el, false); alt.innerHTML = `<div class="scroll"><table><tr><th>Model</th>${names.map(n => `<th class="n">${n}</th>`).join("")}</tr>${MODELS.map(m => `<tr><td>${m}</td>${keys.map(k => `<td class="n">${f3(get(m, k))}</td>`).join("")}</tr>`).join("")}</table></div>`; return; }
      showCanvas(el, true);
      if (st.ch === "radar") return mk(cv, { type: "radar", data: { labels: names, datasets: MODELS.map((m, i) => ({ label: m, data: keys.map(k => get(m, k)), borderColor: MC(i), backgroundColor: MC(i) + "22", pointRadius: 3 })) }, options: { scales: { r: { min: 0, max: 1, ticks: { backdropColor: "transparent" } } } } });
      const vals = MODELS.map(m => get(m, st.metric));
      if (st.ch === "dots") {
        /* one row per model on a shared, zoomed-in axis, with each value written next to its dot and a dashed "no skill" line */
        const ref = st.metric === "auc" ? [0.5, "coin flip 0.50"] : st.metric === "acc" ? [1 - mean(D.y_test), `always guess no ${f3(1 - mean(D.y_test))}`] : null;
        const pts = ref ? [...vals, ref[0]] : vals;
        const lo = Math.max(0, Math.floor((Math.min(...pts) - .02) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...pts) + .02) * 20) / 20);
        const labels = { id: "dotLabels", afterDatasetsDraw(ch) {
          const { ctx, chartArea: a, scales: { x } } = ch; ctx.save(); ctx.textBaseline = "middle"; ctx.textAlign = "left";
          if (ref) { const rx = x.getPixelForValue(ref[0]); ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(rx, a.top); ctx.lineTo(rx, a.bottom); ctx.stroke(); ctx.setLineDash([]); ctx.font = `11px ${Chart.defaults.font.family}`; ctx.fillStyle = css("--muted"); ctx.fillText(ref[1], rx + 5, a.top + 8); }
          ctx.font = `700 12px ${Chart.defaults.font.family}`; ctx.fillStyle = css("--ink");
          ch.getDatasetMeta(0).data.forEach((p, i) => ctx.fillText(f3(vals[i]), p.x + 16, p.y));
          ctx.restore();
        } };
        return mk(cv, { type: "line", data: { labels: MODELS, datasets: [{ data: vals, showLine: false, pointRadius: 10, pointHoverRadius: 12, pointBackgroundColor: MODELS.map((_, i) => MC(i)), pointBorderColor: css("--paper"), pointBorderWidth: 2 }] },
          options: { indexAxis: "y", layout: { padding: { right: 48 } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => f3(c.parsed.x) } } }, scales: { x: { min: lo, max: hi, ticks: { stepSize: .05 } }, y: { offset: true, grid: { display: false } } } }, plugins: [labels] });
      }
      mk(cv, { type: "bar", data: { labels: MODELS, datasets: [{ data: vals, backgroundColor: MODELS.map((_, i) => MC(i)), borderRadius: 8 }] },
        options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => f3(c.parsed.x) } } }, scales: { x: { min: 0, max: 1 } } } });
    },
    caption: st => st.ch === "radar" ? "Each corner is one metric and each colored shape is a model. A bigger shape is better. The shapes nearly overlap, so no model is clearly ahead." : st.ch === "tab" ? "All six measures for each model in one place." : `${st.ch === "dots" ? "Dots further right are better. The axis is zoomed in to where the scores are, so small gaps are easy to see. Read the numbers next to the dots for the real size of a gap." : "Longer bars are better."} ${st.metric === "auc" ? "AUROC does not depend on the threshold." : `This uses a threshold of ${st._t.toFixed(2)}.`} The models are very close, which is part of our finding.`
  });
}

/* ---------- 2. dataset ---------- */
PAGES.dataset = () => {
  const p = h("<div></div>");
  p.append(titlebar("Dataset", "Where the data came from, what is in it, and how we set up the outcome."));
  p.append(section("Source",
    para(`We used the <i>Diabetes 130-US Hospitals for Years 1999-2008</i> dataset from the UCI Machine Learning Repository (Strack et al., 2014, <i>Impact of HbA1c measurement on hospital readmission rates</i>, BioMed Research International, <a href="https://doi.org/10.24432/C5230J">doi.org/10.24432/C5230J</a>). Our copy is a simplified sample of ${D.readmitted.length.toLocaleString()} hospital stays from 130 US hospitals between 1999 and 2008. It was provided for the course. Each row is one stay, and there are no missing values.`)));
  p.append(section("Data at a glance", kpis([["Rows", D.readmitted.length.toLocaleString(), "hospital stays"], ["Columns", "10", "7 used for analysis"], ["Missing values", "0", "nothing to fill in"], ["Predictors", "6", "from the stay"], ["Readmitted", pct(mean(D.readmitted), 0), "of stays"], ["Years", "1999-2008", "130 US hospitals"]])));
  p.append(section("Columns we used", dyn(() => {
    const desc = { time_in_hospital: "Days in the hospital", num_lab_procedures: "Lab tests during the stay", num_procedures: "Procedures other than lab tests", num_medications: "Different medications given", number_diagnoses: "Diagnoses recorded for the stay", diabetes_med: "Prescribed a diabetes medication (1 = yes)" };
    const rows = FEATS.map(f => { const v = D.data[f]; return `<tr><td><code>${f}</code></td><td>Predictor</td><td>${desc[f]}</td><td>${spark(v, css("--violet"))}</td><td class="n">${Math.min(...v)} / ${quant(v, .5)} / ${Math.max(...v)}</td><td class="n">${mean(v).toFixed(2)}</td></tr>`; }).join("");
    return `<div class="card scroll"><table><tr><th>Column</th><th>Role</th><th>Meaning</th><th>Shape</th><th class="n">Min / median / max</th><th class="n">Mean</th></tr>
      <tr><td><code>readmitted</code></td><td><b>Response</b></td><td>Did the patient come back? (we use yes or no)</td><td>${spark(D.readmitted, css("--coral"))}</td><td class="n">0 / 0 / 1</td><td class="n">${mean(D.readmitted).toFixed(2)}</td></tr>${rows}
      <tr><td>age, gender, admission_type</td><td>Not used</td><td>Left out of the models to focus on what happens during the stay. We still show them on the Explore tab for context.</td><td></td><td></td><td></td></tr></table></div>`;
  })));
  p.append(section("Setting up the outcome",
    para("The original <code>readmitted</code> column has three groups: not readmitted, after 30 days, and within 30 days. Our question is about whether a patient comes back at all, so we combined the last two groups into one group called readmitted. This also gives us a fairly balanced outcome, so plain accuracy still means something."),
    card({
      title: "Outcome groups", height: 280, kind: "both",
      toggles: [{ key: "lv", label: "Groups", value: "two", options: [["three", "Original 3"], ["two", "Our 2"]] }, { key: "ch", label: "Show as", value: "doughnut", options: [["doughnut", "Donut"], ["bar", "Bars"], ["pie", "Pie"], ["waffle", "100 patients"]] }],
      draw(st, cv, el) {
        const L = D.readmitted_levels, lab = st.lv === "three" ? ["Not readmitted", "After 30 days", "Within 30 days"] : ["Not readmitted", "Readmitted"];
        const vals = st.lv === "three" ? lab.map(k => L[k]) : [L["Not readmitted"], L["After 30 days"] + L["Within 30 days"]];
        const cols = st.lv === "three" ? [css("--teal"), css("--violet"), css("--coral")] : [css("--teal"), css("--coral")];
        if (st.ch === "waffle") { showCanvas(el, false).innerHTML = waffle(lab.map((l, i) => ({ label: l, n: vals[i], color: cols[i] }))); return; }
        showCanvas(el, true);
        mk(cv, { type: st.ch, data: { labels: lab, datasets: [{ data: vals, backgroundColor: cols, borderWidth: 0, borderRadius: st.ch === "bar" ? 8 : 0 }] },
          options: { plugins: { legend: { display: st.ch !== "bar", position: "right" }, tooltip: { callbacks: { label: c => ` ${c.label}: ${(c.parsed.y ?? c.parsed).toLocaleString()} stays (${pct((c.parsed.y ?? c.parsed) / sum(vals), 1)})` } } }, scales: st.ch === "bar" ? { y: { beginAtZero: true } } : {} } });
      },
      caption: "Each slice, bar or square is a share of the hospital stays. In the 100 patients view, every square is one out of 100 patients. In our two group version about 40 of every 100 stays were readmitted."
    })));
  p.append(section("Browse the data",
    lede("Here are real rows from the file. Use the buttons to filter and shuffle."),
    card({
      title: "Data preview", kind: "html",
      toggles: [{ key: "f", label: "Show", value: "all", options: [["all", "All stays"], ["1", "Readmitted"], ["0", "Not readmitted"]] }, { key: "n", label: "Rows", value: "8", options: [["8", "8"], ["20", "20"]] }, { key: "go", type: "button", label: "🔀 Shuffle", onclick: (st, el) => { st.seed = Math.floor(Math.random() * 1e9); el.redraw(); } }],
      draw(st, body) {
        const R = rng(st.seed || 3), pool = D.readmitted.map((_, i) => i).filter(i => st.f === "all" || D.readmitted[i] === +st.f);
        const pick = []; while (pick.length < Math.min(+st.n, pool.length)) { const i = pool[Math.floor(R() * pool.length)]; if (!pick.includes(i)) pick.push(i); }
        body.innerHTML = `<div class="scroll"><table class="dtab"><tr><th>Age</th><th>Gender</th><th>Admission</th>${FEATS.map(f => `<th class="n">${LBL[f]}</th>`).join("")}<th>Readmitted?</th></tr>${pick.map(i => `<tr><td>${D.cats.age[i]}</td><td>${D.cats.gender[i]}</td><td>${D.cats.admission_type[i]}</td>${FEATS.map(f => `<td class="n">${f === "diabetes_med" ? (D.data[f][i] ? "Yes" : "No") : D.data[f][i]}</td>`).join("")}<td><span class="tag" style="background:${D.readmitted[i] ? css("--coral") : css("--teal")};color:#fff">${D.readmitted[i] ? "Yes" : "No"}</span></td></tr>`).join("")}</table></div>`;
      },
      caption: "Each row is one hospital stay. The first three columns are not used in our models. The last column is what we are trying to predict."
    })));
  p.append(section("Train and test split",
    para(`We split the stays into ${D.n_train.toLocaleString()} for training (75%) and ${D.n_test.toLocaleString()} for testing (25%). The split is stratified, so the readmission rate is about the same in both. Every result in this app comes from the test stays, which the models never saw while training.`),
    card({
      title: "Split sizes and readmission rate", height: 200,
      toggles: [{ key: "v", label: "Show", value: "n", options: [["n", "Number of stays"], ["r", "Readmission rate"]] }],
      draw(st, cv) {
        const nt = D.n_test, ntr = D.n_train, pt = sum(D.y_test), ptr = sum(D.readmitted) - pt;
        const data = st.v === "n" ? [[ntr - ptr, ptr], [nt - pt, pt]] : [[(ntr - ptr) / ntr * 100, ptr / ntr * 100], [(nt - pt) / nt * 100, pt / nt * 100]];
        mk(cv, { type: "bar", data: { labels: ["Training", "Testing"], datasets: [{ label: "Not readmitted", data: data.map(d => d[0]), backgroundColor: css("--teal") }, { label: "Readmitted", data: data.map(d => d[1]), backgroundColor: css("--coral") }] },
          options: { indexAxis: "y", scales: { x: { stacked: true, title: { display: true, text: st.v === "n" ? "Number of stays (count)" : "Share of the data set (% of stays)" } }, y: { stacked: true } } } });
      },
      caption: "Both bars have about the same coral share. That is what stratifying means, and it keeps the test fair."
    })));
  return p;
};

/* ---------- 3. explore ---------- */
function quantInfo(v, k) {
  const vals = D.data[v], ups = [...new Set(Array.from({ length: k }, (_, i) => Math.round(quant(vals, (i + 1) / k))))].sort((a, b) => a - b);
  return { idx: vals.map(x => ups.findIndex(u => x <= u)), labels: ups.map((u, j) => j === 0 ? `≤${u}` : ups[j - 1] + 1 === u ? `${u}` : `${ups[j - 1] + 1}-${u}`) };
}
const quantBins = (v, k) => quantInfo(v, k).idx;
PAGES.explore = () => {
  const p = h("<div></div>");
  p.append(titlebar("Explore the data", "Look at every variable before modeling. Pick a variable, then switch between chart types to see it from different angles."));
  p.append(section("How each variable is distributed",
    lede("If a variable predicted readmission well, the readmitted and not readmitted shapes would look different."),
    card({
      title: "Distribution", height: 330, kind: "both",
      toggles: [{ key: "v", label: "Variable", type: "select", value: "number_diagnoses", sync: "var", options: VAR_OPTS },
        { key: "view", label: "Show as", value: "bar", options: [["bar", "Bars"], ["line", "Smoothed line"], ["cdf", "Cumulative"], ["box", "Box plot"]] },
        { key: "scale", label: "Scale", value: "pct", options: [["pct", "% of group"], ["n", "Count"]] },
        { key: "grp", label: "Groups", value: "split", options: [["split", "Split"], ["all", "All stays"]] }],
      draw(st, cv, el) {
        const y = D.readmitted, v = D.data[st.v], n0 = y.filter(a => !a).length, n1 = y.length - n0;
        st._i = st.v === "diabetes_med" ? `Share prescribed a diabetes medication: ${pct(mean(v.filter((_, i) => !y[i])), 0)} for not readmitted, ${pct(mean(v.filter((_, i) => y[i])), 0)} for readmitted.` : `Average for not readmitted: ${mean(v.filter((_, i) => !y[i])).toFixed(2)}. Average for readmitted: ${mean(v.filter((_, i) => y[i])).toFixed(2)}.`;
        if (st.view === "box") {
          const grp = st.grp === "all" ? [["All stays", v, css("--violet")]] : [["Not readmitted", v.filter((_, i) => !y[i]), css("--teal")], ["Readmitted", v.filter((_, i) => y[i]), css("--coral")]];
          showCanvas(el, false).innerHTML = boxSVG(grp.map(([label, a, color]) => ({ label, color, mean: mean(a), q: [quant(a, .05), quant(a, .25), quant(a, .5), quant(a, .75), quant(a, .95)] })), Math.min(...v), Math.max(...v), axT(st.v));
          return;
        }
        showCanvas(el, true);
        const { labels, idx } = makeBins(st.v, "fine"), nb = labels.length, line = st.view !== "bar";
        const cnt = f => { const c = Array(nb).fill(0); idx.forEach((b, i) => { if (f(i)) c[b]++; }); return c; };
        const smooth = a => st.v === "diabetes_med" ? a : a.map((_, i) => mean(a.slice(Math.max(0, i - 1), i + 2)));
        const mkds = (label, c, n, col) => {
          let d = c.map(x => x / n * 100); if (st.view === "cdf") { let t = 0; d = d.map(x => (t += x)); } else { d = st.scale === "pct" ? d : c; if (line) d = smooth(d); }
          return line ? { label, data: d, borderColor: col, backgroundColor: col + "33", fill: true, pointRadius: 0, tension: .35 } : { label, data: d, backgroundColor: col + "cc", borderRadius: 3 };
        };
        const ds = st.grp === "all" ? [mkds("All stays", cnt(() => true), y.length, css("--violet"))] : [mkds("Not readmitted", cnt(i => !y[i]), n0, css("--teal")), mkds("Readmitted", cnt(i => y[i]), n1, css("--coral"))];
        mk(cv, { type: line ? "line" : "bar", data: { labels, datasets: ds }, options: { scales: { x: { title: { display: true, text: axT(st.v) }, grid: { display: false } }, y: { beginAtZero: true, max: st.view === "cdf" ? 100 : undefined, title: { display: true, text: st.view === "cdf" ? "Cumulative share of group (%)" : st.scale === "pct" ? "Share of group (% of stays)" : "Number of stays (count)" } } } } });
      },
      caption: st => `${st.view === "box" ? "The box covers the middle half of stays, the thick black line is the median, the whiskers reach the 5th and 95th percentiles, and the yellow dot is the average." : st.view === "cdf" ? "Each line shows the share of stays at or below a value. If one group were different, its line would sit clearly above or below the other." : "Each bar is the share of a group with that value. When the coral and teal shapes rise and fall together, that variable does not separate the groups."} ${st._i}`
    })));
  p.append(section("How the readmission rate changes across values",
    lede("This flips the view. For each value or range of a variable, it shows how often those patients were readmitted."),
    card({
      title: "Readmission rate by value", height: 330, kind: "both",
      toggles: [{ key: "v", label: "Variable", type: "select", value: "number_diagnoses", sync: "var", options: VAR_OPTS },
        { key: "t", label: "Show as", value: "bar", options: [["bar", "Bars"], ["line", "Line"], ["area", "Area"], ["strip", "Heat strip"], ["combo", "Count + rate"], ["stack", "Stacked share"]] },
        { key: "d", label: "Detail", value: "fine", options: [["fine", "Each value"], ["coarse", "5 groups"]] }],
      draw(st, cv, el) {
        const { labels, idx } = makeBins(st.v, st.d), y = D.readmitted, n = Array(labels.length).fill(0), r = Array(labels.length).fill(0);
        idx.forEach((b, i) => { n[b]++; r[b] += y[i]; }); const rate = r.map((v, i) => n[i] ? v / n[i] * 100 : null), all = mean(y) * 100;
        st._lo = Math.min(...rate.filter(v => v !== null)); st._hi = Math.max(...rate.filter(v => v !== null));
        if (st.t === "strip") {
          const lo = Math.min(30, st._lo), hi = Math.max(50, st._hi);
          showCanvas(el, false).innerHTML = `<div class="strip" style="grid-template-columns:repeat(${labels.length},minmax(46px,1fr))">${labels.map((l, i) => { const q = rate[i] === null ? .5 : (rate[i] - lo) / (hi - lo), c = rate[i] >= all ? css("--coral") : css("--teal"), a = Math.round(25 + 75 * Math.abs(rate[i] - all) / (hi - lo) * 2); return `<div style="background:color-mix(in srgb,${c} ${Math.min(100, a)}%,${css("--soft")})" title="n = ${n[i]}">${rate[i] === null ? "-" : rate[i].toFixed(0) + "%"}<small>${l}</small></div>`; }).join("")}</div><p class="note" style="color:var(--muted);font-size:.8rem">Horizontal axis: ${axT(st.v)}. Each cell shows the readmission rate (% of stays) for that value or group. Coral cells are above the overall rate (${all.toFixed(0)}%), teal cells are below it.</p>`;
          return;
        }
        showCanvas(el, true);
        if (st.t === "combo") return mk(cv, { type: "bar", data: { labels, datasets: [{ type: "bar", label: "Stays (count)", data: n, backgroundColor: css("--violet") + "66", borderRadius: 4, yAxisID: "y" }, { type: "line", label: "Readmission rate", data: rate, borderColor: css("--coral"), backgroundColor: css("--coral"), pointRadius: 4, tension: .25, yAxisID: "y2" }] }, options: { scales: { x: { title: { display: true, text: axT(st.v) }, grid: { display: false } }, y: { beginAtZero: true, title: { display: true, text: "Number of stays (count)" } }, y2: { position: "right", min: 0, max: 100, grid: { drawOnChartArea: false }, title: { display: true, text: "Readmission rate (% of stays)" } } } } });
        if (st.t === "stack") return mk(cv, { type: "bar", data: { labels, datasets: [{ label: "Not readmitted", data: rate.map(v => v === null ? null : 100 - v), backgroundColor: css("--teal") }, { label: "Readmitted", data: rate, backgroundColor: css("--coral") }] }, options: { scales: { x: { stacked: true, title: { display: true, text: axT(st.v) }, grid: { display: false } }, y: { stacked: true, max: 100, title: { display: true, text: "Share of stays (%)" } } } } });
        const ds = [{ type: st.t === "bar" ? "bar" : "line", label: "Readmission rate", data: rate, backgroundColor: st.t === "bar" ? css("--coral") + "cc" : css("--coral") + "33", fill: st.t === "area", borderColor: css("--coral"), tension: .25, pointRadius: 4, borderRadius: 5 },
          { type: "line", label: `Overall (${all.toFixed(0)}%)`, data: labels.map(() => all), borderColor: css("--ink"), borderDash: [6, 5], pointRadius: 0, borderWidth: 1.5 }];
        mk(cv, { type: "bar", data: { labels, datasets: ds }, options: { scales: { y: { min: 0, title: { display: true, text: "Readmission rate (% of stays)" } }, x: { title: { display: true, text: axT(st.v) }, grid: { display: false } } },
          plugins: { tooltip: { callbacks: { label: c => c.datasetIndex ? c.dataset.label : (() => { const k = c.dataIndex, q = r[k] / n[k], e = 1.96 * Math.sqrt(q * (1 - q) / n[k]) * 100; return ` ${c.parsed.y.toFixed(1)}% (n = ${n[k]}, ±${e.toFixed(1)})`; })() } } } } });
      },
      caption: st => `The dashed line is the overall rate (in the heat strip, coral cells are above it and teal cells are below it). Bars far from it only matter if they also have many stays, so hover to see n and an approximate margin of error. Here the rate runs from ${st._lo.toFixed(0)}% to ${st._hi.toFixed(0)}%.`
    })));
  p.append(section("All variables side by side",
    lede("Split each variable into groups from lowest to highest and compare the readmission rate. A variable that predicts well would show a clear slope."),
    card({
      title: "Readmission rate from lowest to highest values", height: 340,
      toggles: [{ key: "t", label: "Show as", value: "line", options: [["line", "Lines"], ["bar", "Bars"], ["radar", "Radar"]] }, { key: "k", label: "Groups", value: "4", options: [["3", "3"], ["4", "Quartiles"], ["5", "5"]] }],
      draw(st, cv) {
        const k = +st.k, cols = [css("--teal"), css("--coral"), css("--violet"), css("--sun"), "#3b82f6"];
        const names = Array.from({ length: k }, (_, i) => i === 0 ? "Lowest" : i === k - 1 ? "Highest" : `Group ${i + 1}`);
        const ds = FEATS.slice(0, 5).map((f, j) => { const idx = quantBins(f, k), r = Array(k).fill(0), n = Array(k).fill(0); idx.forEach((b, i) => { n[b]++; r[b] += D.readmitted[i]; }); return { label: LBL[f], data: names.map((_, i) => n[i] ? r[i] / n[i] * 100 : null), borderColor: cols[j], backgroundColor: st.t === "bar" ? cols[j] + "cc" : cols[j] + "22", pointRadius: 4, tension: .2, borderRadius: 4 }; });
        if (st.t === "radar") return mk(cv, { type: "radar", data: { labels: names, datasets: ds }, options: { scales: { r: { min: 20, max: 60, ticks: { backdropColor: "transparent" } } } } });
        mk(cv, { type: st.t, data: { labels: names, datasets: ds }, options: { scales: { y: { min: 20, max: 60, title: { display: true, text: "Readmission rate (% of stays)" } } } } });
      },
      caption: "Each line is one variable. If readmission rose steadily with a variable, its line would climb from left to right. Most of these lines wander around the overall 40% without a clear trend."
    })));
  p.append(section("Two variables at once",
    lede("This splits the stays by two variables at the same time and shows the readmission rate in every cell."),
    card({
      title: "Readmission rate by two variables", height: 400, kind: "both",
      toggles: [{ key: "x", label: "Across", type: "select", value: "number_diagnoses", options: VAR_OPTS.slice(0, 5) }, { key: "y", label: "Down", type: "select", value: "time_in_hospital", options: VAR_OPTS.slice(0, 5) },
        { key: "k", label: "Groups", value: "4", options: [["3", "3"], ["4", "4"], ["5", "5"]] }, { key: "t", label: "Show as", value: "heat", options: [["heat", "Heatmap"], ["bubble", "Bubbles"]] }],
      draw(st, cv, el) {
        const k = +st.k, X = quantInfo(st.x, k), Y = quantInfo(st.y, k), nx = X.labels.length, ny = Y.labels.length, n = Array.from({ length: ny }, () => Array(nx).fill(0)), r = Array.from({ length: ny }, () => Array(nx).fill(0));
        D.readmitted.forEach((v, i) => { n[Y.idx[i]][X.idx[i]]++; r[Y.idx[i]][X.idx[i]] += v; }); const all = mean(D.readmitted) * 100;
        if (st.t === "heat") {
          showCanvas(el, false).innerHTML = `<div class="scroll"><table class="heat"><tr><th>${axT(st.y)} ↓ / ${axT(st.x)} →</th>${X.labels.map(l => `<th>${l}</th>`).join("")}</tr>${Y.labels.map((yl, j) => `<tr><th style="text-align:right">${yl}</th>${X.labels.map((_, i) => { const q = n[j][i] ? r[j][i] / n[j][i] * 100 : null, c = q >= all ? css("--coral") : css("--teal"), a = q === null ? 0 : Math.min(95, 15 + Math.abs(q - all) * 4); return `<td style="background:color-mix(in srgb,${c} ${Math.round(a)}%,transparent)" title="n = ${n[j][i]}">${q === null ? "-" : q.toFixed(0) + "%"}<br><small style="opacity:.65">n=${n[j][i]}</small></td>`; }).join("")}</tr>`).join("")}</table></div>`;
          return;
        }
        showCanvas(el, true); const pts = [];
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (n[j][i]) pts.push({ x: i, y: j, r: Math.max(4, Math.sqrt(n[j][i]) * 1.1), q: r[j][i] / n[j][i] * 100, n: n[j][i] });
        mk(cv, { type: "bubble", data: { datasets: [{ data: pts, backgroundColor: pts.map(p => (p.q >= all ? css("--coral") : css("--teal")) + "bb") }] }, options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ${c.raw.q.toFixed(0)}% readmitted (n = ${c.raw.n})` } } }, scales: { x: { min: -.6, max: nx - .4, title: { display: true, text: axT(st.x) }, ticks: { stepSize: 1, callback: v => X.labels[v] ?? "" } }, y: { min: -.6, max: ny - .4, title: { display: true, text: axT(st.y) }, ticks: { stepSize: 1, callback: v => Y.labels[v] ?? "" } } } } });
      },
      caption: st => st.t === "heat" ? "Each cell is a group of stays with both values. Coral cells have a readmission rate above the overall rate and teal cells are below it. Darker means farther from the overall rate. Cells with a small n can be misleading." : "Each bubble is a group of stays. Bigger bubbles have more stays. Coral bubbles have a readmission rate above the overall rate and teal bubbles are below it."
    })));
  p.append(section("Age, gender and admission type (not used in our models)",
    lede("We left these out on purpose, but they give useful context about who is in the data."),
    card({
      title: "Context variables", height: 300,
      toggles: [{ key: "v", label: "Variable", type: "select", value: "age", options: [["age", "Age group"], ["gender", "Gender"], ["admission_type", "Admission type"]] },
        { key: "t", label: "Show as", value: "rate", options: [["rate", "Readmission rate"], ["n", "Number of stays"], ["stack", "Stacked share"], ["donut", "Donut"]] }],
      draw(st, cv) {
        const vals = D.cats[st.v], y = D.readmitted, order = st.v === "age" ? ["0-49", "50-59", "60-69", "70-79", "80+"] : [...new Set(vals)].sort((a, b) => vals.filter(x => x === b).length - vals.filter(x => x === a).length);
        const n = order.map(k => vals.filter(v => v === k).length), r = order.map(k => mean(y.filter((_, i) => vals[i] === k)) * 100), pal = [css("--teal"), css("--coral"), css("--violet"), css("--sun"), "#3b82f6"];
        if (st.t === "donut") return mk(cv, { type: "doughnut", data: { labels: order, datasets: [{ data: n, backgroundColor: pal, borderWidth: 0 }] }, options: { plugins: { legend: { position: "right" } } } });
        if (st.t === "stack") return mk(cv, { type: "bar", data: { labels: order, datasets: [{ label: "Not readmitted", data: r.map(v => 100 - v), backgroundColor: css("--teal") }, { label: "Readmitted", data: r, backgroundColor: css("--coral") }] }, options: { scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: "Share of group (%)" } } } } });
        mk(cv, { type: "bar", data: { labels: order, datasets: [{ label: st.t === "rate" ? "% readmitted" : "Stays", data: st.t === "rate" ? r : n, backgroundColor: pal.slice(0, order.length).map(c => c + "dd"), borderRadius: 8 }] },
          options: { plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: c => st.t === "rate" ? `n = ${n[c.dataIndex]}` : `${r[c.dataIndex].toFixed(1)}% readmitted` } } }, scales: { y: { beginAtZero: true } } } });
      },
      caption: "Each bar or slice is one group. Switch between how common a group is and how often its patients came back. The stacked view shows the teal and coral split inside each group. Differences here are small too."
    })));
  p.append(section("How the variables relate to each other",
    card({
      title: "Correlation heatmap", kind: "html",
      toggles: [{ key: "rows", label: "Show", value: "all", options: [["all", "All pairs"], ["out", "Only vs readmitted"]] }, { key: "num", label: "Numbers", value: "on", options: [["on", "On"], ["off", "Off"]] }],
      draw(st, body) {
        const names = [...FEATS, "readmitted"], cols = names.map(n => n === "readmitted" ? D.readmitted : D.data[n]);
        const cell = (r, self) => {
          if (self) return `<td class="self" title="A variable compared with itself is always 1">${st.num === "on" ? "1.00" : "&nbsp;&nbsp;"}</td>`;
          const t = tint(r >= 0 ? "--coral" : "--teal", Math.min(.9, Math.abs(r) * 1.8), "--soft");
          return `<td style="background:${t.fill};color:${t.text}" title="r = ${r.toFixed(3)}">${st.num === "on" ? r.toFixed(2) : "&nbsp;&nbsp;"}</td>`;
        };
        let html;
        if (st.rows === "out") html = `<table class="heat corr"><tr><th></th><th>Correlation with readmitted (r, -1 to 1)</th></tr>${FEATS.map((f, i) => `<tr><th style="text-align:right">${LBL[f]}</th>${cell(pearson(cols[i], D.readmitted))}</tr>`).join("")}</table>`;
        else html = `<div class="scroll"><table class="heat corr"><tr><th>Correlation (r, -1 to 1)</th>${names.map(n => `<th>${LBL[n]}</th>`).join("")}</tr>${names.map((a, i) => `<tr><th style="text-align:right">${LBL[a]}</th>${names.map((b, j) => cell(i === j ? 1 : pearson(cols[i], cols[j]), i === j)).join("")}</tr>`).join("")}</table></div>`;
        body.innerHTML = html;
      },
      caption: "Coral cells mean the two variables rise together, teal cells mean one rises as the other falls, and the stronger the color, the stronger the link. Cells with little color are close to zero. The plain diagonal is each variable compared with itself, which is always 1. The Readmitted row shows what we care about, and all of those are close to zero. Medications and diagnoses move together, which can make a model's job harder."
    })));
  p.append(section("Looking at two variables together",
    card({
      title: "Scatter plot", height: 350,
      toggles: [{ key: "x", label: "X", type: "select", value: "time_in_hospital", options: VAR_OPTS.slice(0, 5) }, { key: "y", label: "Y", type: "select", value: "num_medications", options: VAR_OPTS.slice(0, 5) },
        { key: "n", label: "Points", value: "600", options: [["300", "300"], ["600", "600"], ["1500", "1500"]] }, { key: "c", label: "Color by outcome", value: "on", options: [["on", "On"], ["off", "Off"]] }],
      draw(st, cv) {
        const R = rng(7), N = +st.n, mkp = f => { const o = []; for (let i = 0; i < N; i++) if (f(i)) o.push({ x: D.data[st.x][i] + (R() - .5) * .6, y: D.data[st.y][i] + (R() - .5) * .6 }); return o; };
        const y = D.readmitted, ds = st.c === "on" ? [{ label: "Not readmitted", data: mkp(i => !y[i]), backgroundColor: css("--teal") + "cc", borderColor: css("--teal") }, { label: "Readmitted", data: mkp(i => y[i]), backgroundColor: css("--coral") + "cc", borderColor: css("--coral") }] : [{ label: "Stays", data: mkp(() => true), backgroundColor: css("--violet") + "cc", borderColor: css("--violet") }];
        mk(cv, { type: "scatter", data: { datasets: ds.map(d => ({ ...d, pointRadius: 3.5, borderWidth: 1 })) }, options: { scales: { x: { title: { display: true, text: axT(st.x) } }, y: { title: { display: true, text: axT(st.y) } } } } });
      },
      caption: "Each dot is one stay (jittered a little so dots do not stack). If coral and teal dots formed separate clouds, two variables could predict readmission. Here they are mixed together almost completely."
    })));
  p.append(section("Which variables differ the most between groups?",
    card({
      title: "Readmitted vs not readmitted", height: 320,
      toggles: [{ key: "v", label: "Show as", value: "d", options: [["d", "Effect size"], ["g", "Group averages"], ["r", "Radar profile"]] }],
      draw(st, cv) {
        const y = D.readmitted, rows = FEATS.map(f => { const v = D.data[f], a = v.filter((_, i) => y[i]), b = v.filter((_, i) => !y[i]); return { f, d: (mean(a) - mean(b)) / Math.sqrt((sd(a) ** 2 + sd(b) ** 2) / 2), ia: mean(a) / mean(v) * 100, ib: mean(b) / mean(v) * 100 }; });
        if (st.v === "d") { rows.sort((a, b) => Math.abs(b.d) - Math.abs(a.d)); st._t = LBL[rows[0].f];
          mk(cv, { type: "bar", data: { labels: rows.map(r => LBL[r.f]), datasets: [{ data: rows.map(r => r.d), backgroundColor: rows.map(r => r.d >= 0 ? css("--coral") : css("--teal")), borderRadius: 6 }] }, options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "Effect size (standard deviations, readmitted minus not)" } } } } }); }
        else if (st.v === "g") mk(cv, { type: "bar", data: { labels: rows.map(r => LBL[r.f]), datasets: [{ label: "Not readmitted", data: rows.map(r => r.ib), backgroundColor: css("--teal"), borderRadius: 4 }, { label: "Readmitted", data: rows.map(r => r.ia), backgroundColor: css("--coral"), borderRadius: 4 }] }, options: { scales: { y: { min: 80, title: { display: true, text: "Group average as an index (overall average = 100)" } } } } });
        else mk(cv, { type: "radar", data: { labels: rows.map(r => LBL[r.f]), datasets: [{ label: "Not readmitted", data: rows.map(r => r.ib), borderColor: css("--teal"), backgroundColor: css("--teal") + "33" }, { label: "Readmitted", data: rows.map(r => r.ia), borderColor: css("--coral"), backgroundColor: css("--coral") + "33" }] }, options: { scales: { r: { min: 80, max: 120, ticks: { backdropColor: "transparent" } } } } });
      },
      caption: st => st.v === "d" ? `This is the group difference measured in standard deviations (Cohen's d). Values near 0 mean no difference. Even the biggest one (${st._t}) is small, and by convention 0.2 is a "small" effect.` : "Each point or bar is a group's average compared with the overall average (100). The two groups are almost the same on every variable."
    })));
  return p;
};

/* Color helpers: mix a theme color into a background and pick dark or white text, whichever contrasts more. */
const hexRGB = hex => { const v = parseInt(hex.replace("#", ""), 16); return [v >> 16 & 255, v >> 8 & 255, v & 255]; };
const lumRGB = c => { const [r, g, b] = c.map(v => { v /= 255; return v <= .03928 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4; }); return .2126 * r + .7152 * g + .0722 * b; };
const contrastRGB = (a, b) => (Math.max(lumRGB(a), lumRGB(b)) + .05) / (Math.min(lumRGB(a), lumRGB(b)) + .05);
const TXT_DARK = "#14213d", TXT_LIGHT = "#ffffff";
function tint(colorVar, k, baseVar) {
  const c = hexRGB(css(colorVar)), b = hexRGB(css(baseVar)), fill = c.map((v, i) => Math.round(v * k + b[i] * (1 - k)));
  return { fill: `rgb(${fill})`, text: contrastRGB(fill, hexRGB(TXT_DARK)) >= contrastRGB(fill, hexRGB(TXT_LIGHT)) ? TXT_DARK : TXT_LIGHT };
}

/* The tree is drawn sideways: the first question is on the left, "yes" branches go up and "no" branches go down,
   so its 5 levels fit across the card instead of its 13 leaves. Text color is picked per box for contrast. */
function treeSVG() {
  const T = D.tree, BW = 136, BH = 32, ROW = 38, PAD = 4; let leaf = 0, maxD = 0; const pos = {};
  const walk = (n, d) => { maxD = Math.max(maxD, d); if (T.l[n] === -1) { pos[n] = { y: PAD + leaf++ * ROW + ROW / 2, d }; return; } walk(T.l[n], d + 1); walk(T.r[n], d + 1); pos[n] = { y: (pos[T.l[n]].y + pos[T.r[n]].y) / 2, d }; };
  walk(0, 0);
  const wd = 820, step = (wd - BW) / Math.max(1, maxD), ht = leaf * ROW + PAD * 2, X = n => pos[n].d * step;
  const box = p => tint(p >= .4 ? "--coral" : "--teal", Math.min(85, 20 + Math.abs(p - .4) * 400) / 100, "--paper");
  let edges = "", nodes = "";
  T.l.forEach((l, n) => {
    if (l === -1) return;
    [[l, "yes"], [T.r[n], "no"]].forEach(([c, lab]) => {
      const x1 = X(n) + BW, y1 = pos[n].y, x2 = X(c), y2 = pos[c].y, xm = (x1 + x2) / 2;
      edges += `<path d="M${x1},${y1} C${xm},${y1} ${xm},${y2} ${x2},${y2}" fill="none" stroke="${css("--line")}" stroke-width="2"/><text x="${xm}" y="${(y1 + y2) / 2 + (lab === "yes" ? -5 : 11)}" text-anchor="middle" style="font-size:9.5px;font-weight:600">${lab}</text>`;
    });
  });
  T.l.forEach((l, n) => {
    const isLeaf = l === -1, f = T.f[n], x = X(n), y = pos[n].y, b = box(T.p[n]);
    const cond = isLeaf ? `${pct(T.p[n], 0)} readmitted` : f === 5 ? "No diabetes med?" : `${LBL[FEATS[f]]} ≤ ${Math.floor(T.t[n])}?`;
    nodes += `<g><rect x="${x}" y="${y - BH / 2}" width="${BW}" height="${BH}" rx="8" fill="${b.fill}" stroke="${css("--line")}"/><text x="${x + BW / 2}" y="${y - 2}" text-anchor="middle" style="fill:${b.text};font-weight:700;font-size:10.5px">${cond}</text><text x="${x + BW / 2}" y="${y + 10}" text-anchor="middle" style="fill:${b.text};opacity:.85;font-size:9.5px">n = ${T.n[n]}${isLeaf ? "" : " · " + pct(T.p[n], 0)}</text></g>`;
  });
  return `<div class="scroll"><svg viewBox="0 0 ${wd} ${ht}" width="100%" style="display:block;min-width:600px;max-width:${wd}px;height:auto;margin:0 auto" role="img" aria-label="Decision tree diagram">${edges}${nodes}</svg></div>`;
}

/* ---------- 4. models ---------- */
PAGES.models = () => {
  const p = h("<div></div>");
  p.append(titlebar("Analysis process: models and evaluation", "The steps we took, the three classification models, how we evaluated them on the 750 test stays, and which one we chose."));
  p.append(section("The three models",
    para("<b>First,</b> we split the data. <b>Then</b> we trained three classification models. <b>Next,</b> we evaluated each one on the test set with a confusion matrix, accuracy, sensitivity, specificity, and the ROC curve with the AUROC. <b>Finally,</b> we compared them."),
    h(`<div class="grid2" style="grid-template-columns:repeat(auto-fit,minmax(240px,1fr))">
      <div class="card"><span class="tag">Baseline</span><h4>Logistic regression</h4><p>Combines the six variables into one score. Simple, fast and easy to explain, because each variable gets one weight. We standardized the variables first so the weights can be compared.</p></div>
      <div class="card"><span class="tag">Rules</span><h4>Decision tree</h4><p>Asks a series of yes or no questions, like "more than 6 diagnoses?". We limited it to a depth of 4 and at least 30 stays per leaf so it does not memorize the training data.</p></div>
      <div class="card"><span class="tag">Ensemble</span><h4>Random forest</h4><p>Averages 300 different trees, each trained on a random piece of the data. It can pick up more complicated patterns and is usually steadier than one tree.</p></div></div>`)));
  p.append(dyn(() => { const best = MODELS.reduce((a, b) => ROC[a].auc > ROC[b].auc ? a : b); return `<div class="callout warn"><h4>Our decision: the ${best.toLowerCase()}</h4>It has the highest AUROC (${f3(ROC[best].auc)}) and accuracy of the three models, so we picked it. The models are very close though, so logistic regression is a fair alternative because it is simpler. We set the threshold at 0.40 (see the next tab).</div>`; }));
  p.append(settingsBar());
  p.append(dyn(() => { const c = confusion(D.probs[S.model], D.y_test, S.thr); return kpis([["Accuracy", pct(c.acc), `${c.tp + c.tn} of ${c.n} correct`], ["Sensitivity", pct(c.sens), "readmissions caught"], ["Specificity", pct(c.spec), "non-readmissions cleared"], ["Precision", pct(c.prec), "flagged who came back"], ["AUROC", f3(ROC[S.model].auc), "all thresholds"]]).outerHTML; }));
  p.append(section("Confusion matrix and curves",
    h(`<div class="grid2" id="g1"></div>`)));
  const g1 = $("#g1", p) || p.querySelector(".grid2:last-child");
  g1.append(card({
    title: "Confusion matrix", kind: "html", toggles: [{ key: "v", label: "Show as", value: "n", options: [["n", "Counts"], ["r", "Row %"], ["w", "100 patients"], ["all", "All 3 models"]] }],
    draw(st, body) {
      if (st.v === "all") { body.innerHTML = `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px">${MODELS.map((m, i) => { const q = confusion(D.probs[m], D.y_test, S.thr); return `<div><b style="color:${MC(i)}">${m}</b><div class="cm" style="grid-template-columns:1fr 1fr;font-size:.7rem"><div class="cell good" style="padding:10px">${q.tn}<small>TN</small></div><div class="cell bad" style="padding:10px">${q.fp}<small>FP</small></div><div class="cell bad" style="padding:10px">${q.fn}<small>FN</small></div><div class="cell good" style="padding:10px">${q.tp}<small>TP</small></div></div><div style="font-size:.78rem;color:var(--muted);text-align:center">sens ${f3(q.sens)} · spec ${f3(q.spec)}</div></div>`; }).join("")}</div>`; return; }
      const c = confusion(D.probs[S.model], D.y_test, S.thr), P = c.tp + c.fn, N = c.tn + c.fp, f = (v, d) => st.v === "n" ? v : pct(v / d, 0);
      if (st.v === "w") { body.innerHTML = waffle([{ label: "True negative", n: c.tn, color: css("--teal") }, { label: "False positive", n: c.fp, color: css("--sun") }, { label: "False negative", n: c.fn, color: css("--coral") }, { label: "True positive", n: c.tp, color: css("--violet") }]); return; }
      body.innerHTML = `<div class="cm"><div></div><div>Predicted<br>not readmitted</div><div>Predicted<br>readmitted</div>
        <div class="lab">Actual<br>not readm.</div><div class="cell good">${f(c.tn, N)}<small>TRUE NEGATIVE</small></div><div class="cell bad">${f(c.fp, N)}<small>FALSE POSITIVE</small></div>
        <div class="lab">Actual<br>readmitted</div><div class="cell bad">${f(c.fn, P)}<small>FALSE NEGATIVE</small></div><div class="cell good">${f(c.tp, P)}<small>TRUE POSITIVE</small></div></div>`;
    },
    caption: st => st.v === "all" ? "The same threshold applied to all three models. Top left is correct not readmitted, bottom right is correct readmitted, and the other two boxes are mistakes." : st.v === "w" ? "Each square is one out of every 100 test patients. Violet and teal squares are correct predictions. Yellow squares are false alarms and coral squares are readmissions the model missed." : "Each box counts test stays. Teal boxes are correct predictions, coral boxes are mistakes. Switch to Row % to see sensitivity (bottom right) and specificity (top left) directly."
  }));
  g1.append(card({
    title: "Performance curve", height: 320,
    toggles: [{ key: "c", label: "Curve", value: "roc", options: [["roc", "ROC"], ["pr", "Precision-recall"], ["gain", "Gain"], ["lift", "Lift"], ["cal", "Calibration"]] }, { key: "s", label: "Models", value: "all", options: [["all", "All three"], ["sel", "Selected"]] }],
    draw(st, cv) {
      const key = st.c, base = sum(D.y_test) / D.y_test.length, n = D.y_test.length, list = MODELS.map((m, i) => ({ m, i })).filter(o => st.s === "all" || o.m === S.model);
      const sty = (m, i) => ({ showLine: true, pointRadius: 0, borderColor: MC(i) + (m === S.model ? "" : "99"), borderWidth: m === S.model ? 3 : 1.4 });
      const order = m => D.probs[m].map((_, i) => i).sort((a, b) => D.probs[m][b] - D.probs[m][a]);
      const gain = m => { const o = order(m), pts = [{ x: 0, y: 0 }]; let c = 0; o.forEach((i, k) => { c += D.y_test[i]; if ((k + 1) % 15 === 0 || k === n - 1) pts.push({ x: (k + 1) / n, y: c / sum(D.y_test) }); }); return pts; };
      const cal = m => { const o = order(m).reverse(), pts = [], b = 10, sz = Math.floor(n / b); for (let j = 0; j < b; j++) { const part = o.slice(j * sz, j === b - 1 ? n : (j + 1) * sz); pts.push({ x: mean(part.map(i => D.probs[m][i])), y: mean(part.map(i => D.y_test[i])) }); } return pts; };
      let ds, xt, yt, xr = [0, 1], yr = [0, 1], type = "scatter";
      if (key === "roc") { ds = list.map(({ m, i }) => ({ label: `${m} (AUROC ${f3(ROC[m].auc)})`, data: ROC[m].roc, ...sty(m, i) })); ds.push({ label: "Coin flip", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderDash: [5, 5], borderColor: css("--muted"), borderWidth: 1 }); const c = confusion(D.probs[S.model], D.y_test, S.thr); ds.push({ label: "Current threshold", data: [{ x: 1 - c.spec, y: c.sens }], pointRadius: 7, backgroundColor: css("--ink"), borderColor: css("--paper"), borderWidth: 2 }); xt = "False positive rate (share of non-readmitted flagged, 0 to 1)"; yt = "Sensitivity (share of readmissions caught, 0 to 1)"; }
      else if (key === "pr") { ds = list.map(({ m, i }) => ({ label: `${m} (AP ${f3(PR[m].ap)})`, data: PR[m].pr, ...sty(m, i) })); ds.push({ label: `Always guess yes (${pct(base, 0)})`, data: [{ x: 0, y: base }, { x: 1, y: base }], showLine: true, pointRadius: 0, borderDash: [5, 5], borderColor: css("--muted"), borderWidth: 1 }); const c = confusion(D.probs[S.model], D.y_test, S.thr); ds.push({ label: "Current threshold", data: [{ x: c.sens, y: c.prec }], pointRadius: 7, backgroundColor: css("--ink"), borderColor: css("--paper"), borderWidth: 2 }); xt = "Recall (share of readmissions caught, 0 to 1)"; yt = "Precision (share of flagged who returned, 0 to 1)"; yr = [.2, 1]; }
      else if (key === "gain") { ds = list.map(({ m, i }) => ({ label: m, data: gain(m), ...sty(m, i) })); ds.push({ label: "Random order", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderDash: [5, 5], borderColor: css("--muted"), borderWidth: 1 }); xt = "Share of patients called, highest risk first (0 to 1)"; yt = "Share of all readmissions reached (0 to 1)"; }
      else if (key === "lift") { ds = list.map(({ m, i }) => ({ label: m, data: gain(m).filter(p => p.x > .04).map(p => ({ x: p.x, y: p.y / p.x })), ...sty(m, i) })); ds.push({ label: "Random order (1.0)", data: [{ x: .05, y: 1 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderDash: [5, 5], borderColor: css("--muted"), borderWidth: 1 }); xt = "Share of patients called, highest risk first (0 to 1)"; yt = "Lift (times better than random calling)"; yr = [.6, 2]; }
      else { ds = list.map(({ m, i }) => ({ label: m, data: cal(m), ...sty(m, i), pointRadius: 4 })); ds.push({ label: "Perfectly calibrated", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderDash: [5, 5], borderColor: css("--muted"), borderWidth: 1 }); xt = "Predicted probability of readmission (0 to 1)"; yt = "Actual share readmitted (0 to 1)"; xr = [.1, .8]; yr = [.1, .8]; }
      const clickRoc = key !== "roc" ? undefined : (e, _, ch) => { const x = ch.scales.x.getValueForPixel(e.x), y = ch.scales.y.getValueForPixel(e.y); let best = null, bd = 1e9; ROC[S.model].roc.forEach(pt => { const d = (pt.x - x) ** 2 + (pt.y - y) ** 2; if (d < bd) { bd = d; best = pt; } }); if (best && best.t <= 1) setThr(best.t); };
      mk(cv, { type, data: { datasets: ds }, options: { onClick: clickRoc, onHover: (e, a, ch) => { ch.canvas.style.cursor = key === "roc" ? "pointer" : "default"; }, scales: { x: { min: xr[0], max: xr[1], title: { display: true, text: xt } }, y: { min: yr[0], max: yr[1], title: { display: true, text: yt } } }, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
    },
    caption: st => ({ roc: "A curve that bows toward the top left is better, and the dashed diagonal is a coin flip. Ours hug the diagonal. The black dot is where the current threshold sits on the selected model. Click anywhere near the curve to move the threshold to that point: every chart in the app updates.", pr: "Precision is how often a flagged patient really comes back. The dashed line is what you would get by flagging everyone. Our curves stay close to it.", gain: "If we called patients from highest to lowest predicted risk, this shows what share of all readmissions we would reach. The dashed line is calling patients in random order. Higher above it is better.", lift: "Lift is how many times more readmissions we find than random calling would at the same number of calls. A lift of 1.0 means no better than random.", cal: "Each point is a tenth of the patients grouped by predicted risk. If a model were well calibrated, the points would follow the dashed line. Ours are squeezed into a narrow range because the models are not confident." })[st.c]
  }));
    p.append(section("How the predictions are spread",
    card({
      title: "Predicted probabilities for the selected model", height: 310,
      toggles: [{ key: "st", label: "Show as", value: "bar", options: [["bar", "Bars"], ["line", "Smoothed"], ["cdf", "Cumulative"]] }, { key: "b", label: "Bins", value: "20", options: [["10", "10"], ["20", "20"], ["40", "40"]] }],
      draw(st, cv) {
        const nb = +st.b, pr = D.probs[S.model], y = D.y_test, lo = Math.min(...pr), hi = Math.max(...pr), w = (hi - lo) / nb || 1;
        const cnt = g => { const c = Array(nb).fill(0); pr.forEach((q, i) => { if (y[i] === g) c[Math.min(nb - 1, Math.floor((q - lo) / w))]++; }); return c; };
        const n0 = y.filter(v => !v).length, n1 = y.length - n0, lab = Array.from({ length: nb }, (_, i) => (lo + i * w).toFixed(2));
        let a = cnt(0).map(v => v / n0 * 100), b = cnt(1).map(v => v / n1 * 100);
        if (st.st === "line") { const sm = x => x.map((_, i) => mean(x.slice(Math.max(0, i - 1), i + 2))); a = sm(a); b = sm(b); }
        if (st.st === "cdf") { let t = 0, u = 0; a = a.map(v => (t += v)); b = b.map(v => (u += v)); }
        const bar = st.st === "bar", mkd = (label, d, col) => bar ? { label, data: d, backgroundColor: col + "aa", borderRadius: 3 } : { label, data: d, borderColor: col, backgroundColor: col + "33", fill: st.st === "line", pointRadius: 0, tension: .3 };
        mk(cv, { type: bar ? "bar" : "line", data: { labels: lab, datasets: [mkd("Not readmitted", a, css("--teal")), mkd("Readmitted", b, css("--coral"))] }, options: { onClick: (e, _, ch) => { const ca = ch.chartArea; if (e.x < ca.left || e.x > ca.right) return; setThr(lo + (e.x - ca.left) / (ca.right - ca.left) * nb * w); }, onHover: (e, a, ch) => { ch.canvas.style.cursor = "pointer"; }, scales: { x: { title: { display: true, text: "Predicted probability of readmission (0 to 1)" }, grid: { display: false }, ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: st.st === "cdf" ? "Cumulative share of group (%)" : "Share of group (% of test stays)" } } } } });
      },
      caption: st => st.st === "cdf" ? "Each line shows the share of a group with a predicted probability at or below that value. A model that separates the groups well would have the lines far apart. Ours are nearly on top of each other." : "A good model pushes the teal shape to the left and the coral shape to the right. Ours overlap almost completely, which is why no threshold can separate the groups well. Click anywhere on the chart to set the threshold at that probability."
    })));
    p.append(section("Comparing the models", cmpCard(),
    card({
      title: "Comparison table", kind: "html", toggles: [],
      draw(st, body) {
        body.innerHTML = `<div class="scroll"><table><tr><th>Model</th><th class="n">Accuracy</th><th class="n">Sensitivity</th><th class="n">Specificity</th><th class="n">Precision</th><th class="n">F1</th><th class="n">AUROC</th></tr>${MODELS.map(m => { const c = confusion(D.probs[m], D.y_test, S.thr); return `<tr class="${m === S.model ? "hl" : ""}"><td>${m}</td><td class="n">${f3(c.acc)}</td><td class="n">${f3(c.sens)}</td><td class="n">${f3(c.spec)}</td><td class="n">${f3(c.prec)}</td><td class="n">${f3(c.f1)}</td><td class="n">${f3(ROC[m].auc)}</td></tr>`; }).join("")}</table></div>`;
      },
      caption: "This table uses the threshold from the slider at the top, and highlights the selected model."
    }),
    para("<b>Which model is best?</b> We chose the <b>random forest</b> because it has the highest AUROC and accuracy. The three models are very close though, with an AUROC around 0.59. Logistic regression is also a good choice because it is simpler and easier to explain. At a threshold of 0.5 every model has very low sensitivity because they mostly guess not readmitted. About 60% of stays are not readmitted, so 60% accuracy is the number to beat.")));
  p.append(section("What the models used",
    card({
      title: "Variable importance", height: 320, kind: "canvas",
      toggles: [{ key: "t", label: "Show as", value: "bar", options: [["bar", "Bars"], ["radar", "Radar"], ["all", "All 3 models"]] }],
      draw(st, cv) {
        const nrm = a => { const t = sum(a.map(Math.abs)); return a.map(v => Math.abs(v) / t * 100); }, imp = m => m === MODELS[0] ? nrm(D.lr.coef) : nrm(D.importance[m]);
        if (st.t === "all") return mk(cv, { type: "bar", data: { labels: FEATS.map(f => LBL[f]), datasets: MODELS.map((m, i) => ({ label: m, data: imp(m), backgroundColor: MC(i), borderRadius: 4 })) }, options: { scales: { y: { title: { display: true, text: "Share of importance (%)" } } } } });
        if (st.t === "radar") return mk(cv, { type: "radar", data: { labels: FEATS.map(f => LBL[f]), datasets: [{ label: S.model, data: imp(S.model), borderColor: MC(MODELS.indexOf(S.model)), backgroundColor: MC(MODELS.indexOf(S.model)) + "33" }] }, options: { scales: { r: { min: 0, ticks: { backdropColor: "transparent" } } } } });
        const m = S.model, vals = m === MODELS[0] ? D.lr.coef : D.importance[m], order = vals.map((v, i) => i).sort((a, b) => Math.abs(vals[b]) - Math.abs(vals[a]));
        mk(cv, { type: "bar", data: { labels: order.map(i => LBL[FEATS[i]]), datasets: [{ data: order.map(i => vals[i]), backgroundColor: order.map(i => m === MODELS[0] ? (vals[i] >= 0 ? css("--coral") : css("--teal")) : MC(MODELS.indexOf(m))), borderRadius: 6 }] }, options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: m === MODELS[0] ? "Standardized coefficient (change in log-odds per 1 standard deviation)" : "Importance (share of the model's total split gain, 0 to 1)" } } } } });
      },
      caption: st => st.t === "all" ? "Each model's importance scaled to add up to 100%, so you can compare which variables each one leaned on. Number of diagnoses stands out in all three." : S.model === MODELS[0] ? "Each bar is that variable's weight in the logistic regression. Coral pushes the prediction toward readmitted and teal pushes it away. Longer bars matter more." : "Longer bars mean the model relied on that variable more when splitting. Importance shows what the model used, not that the variable causes readmission."
    }),
        card({
      title: "Inside the models", kind: "html", toggles: [{ key: "v", label: "Show", value: "tree", options: [["tree", "Decision tree rules"], ["diag", "Tree diagram"], ["lr", "Logistic regression weights"]] }],
      draw(st, body) {
        if (st.v === "diag") return void (body.innerHTML = treeSVG());
        if (st.v === "tree") { const L = D.tree_text.replace(/\s+$/, "").split("\n"), cut = L.findIndex((l, i) => i > 0 && l.startsWith("|--- ")), esc = a => a.join("\n").replace(/</g, "&lt;"); body.innerHTML = cut > 0 ? `<div class="rules2"><pre>${esc(L.slice(0, cut))}</pre><pre>${esc(L.slice(cut))}</pre></div>` : `<pre>${esc(L)}</pre>`; }
        else body.innerHTML = `<div class="scroll"><table><tr><th>Variable</th><th class="n">Weight (per 1 SD)</th><th class="n">Odds ratio per 1 unit</th></tr>${FEATS.map((f, i) => `<tr><td>${LBL[f]}</td><td class="n">${D.lr.coef[i].toFixed(3)}</td><td class="n">${Math.exp(D.lr.coef[i] / D.lr.scale[i]).toFixed(3)}</td></tr>`).join("")}<tr><td>Intercept</td><td class="n">${D.lr.intercept.toFixed(3)}</td><td></td></tr></table></div>`;
      },
      caption: st => st.v === "diag" ? "Each box is a question or a final answer. Start at the left. Follow the upper branch when the answer is yes and the lower branch when it is no. Box color shows the share of readmitted stays in that group (coral is higher, teal is lower), and n is how many training stays reached the box. Every branch ends near the overall 40%, which is why the tree is a weak predictor." : st.v === "tree" ? "The rules are split at the first question: the left box is for 6 or fewer diagnoses and the right box is for 7 or more. Read each box from the top. At each line, follow the branch that is true for the patient. The class at the end of a branch is the prediction (0 = not readmitted, 1 = readmitted)." : "An odds ratio of 1.03 means each extra unit of that variable raises the odds of readmission by about 3%. Values close to 1 mean a tiny effect."
    })));
  p.append(section("Could the differences just be luck?",
    card({
      title: "Bootstrap check of the AUROC", height: 220,
      toggles: [{ key: "go", type: "button", label: "Run 300 resamples", onclick: (st, el) => { runBoot(); el.redraw(); } }],
      draw(st, cv) {
        if (!BOOT) { const c = cv.getContext("2d"); mk(cv, { type: "bar", data: { labels: [], datasets: [] } }); st._i = "Press the button to resample the test set 300 times."; return; }
        mk(cv, { type: "bar", data: { labels: MODELS, datasets: [{ data: BOOT.ci, backgroundColor: MODELS.map((_, i) => MC(i) + "cc"), borderRadius: 4 }] }, options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` 95% range: ${f3(c.raw[0])} to ${f3(c.raw[1])}` } } }, scales: { x: { min: .45, max: .75, title: { display: true, text: "AUROC (95% range across 300 resamples, 0.5 = coin flip)" } } } } });
        st._i = `The random forest beat logistic regression in ${pct(BOOT.rfBeatsLr, 0)} of resamples. All three ranges overlap and sit well below 0.7.`;
      },
      caption: st => `We redraw the 750 test stays with replacement 300 times and recompute each AUROC. Each bar shows where the middle 95% of results fall. Overlapping bars mean the models are not clearly different. ${st._i || ""}`
    })));
  return p;
};
function runBoot() {
  const R = rng(42), n = D.y_test.length, aucs = MODELS.map(() => []); let wins = 0;
  for (let b = 0; b < 300; b++) {
    const idx = Array.from({ length: n }, () => Math.floor(R() * n)), y = idx.map(i => D.y_test[i]), a = MODELS.map((m, k) => { const v = curves(idx.map(i => D.probs[m][i]), y).auc; aucs[k].push(v); return v; });
    if (a[2] > a[0]) wins++;
  }
  BOOT = { ci: aucs.map(a => [quant(a, .025), quant(a, .975)]), rfBeatsLr: wins / 300 };
}

/* ---------- 5. threshold ---------- */
PAGES.threshold = () => {
  const p = h("<div></div>");
  p.append(titlebar("Analysis process: choosing a threshold", "A model gives a probability. The threshold decides when we call that a prediction of readmitted."));
  p.append(settingsBar());
  p.append(section("In plain terms",
    dyn(() => {
      const c = confusion(D.probs[S.model], D.y_test, S.thr), k = 100 / c.n, P = (c.tp + c.fn) * k;
      return `<div class="card">Imagine <b>100 patients</b> leaving the hospital, using the <b>${S.model}</b> at a threshold of <b>${S.thr.toFixed(2)}</b>. About <b>${P.toFixed(0)}</b> of them will really come back. The model flags <b>${((c.tp + c.fp) * k).toFixed(0)}</b> patients for follow up. Of those, <b>${(c.tp * k).toFixed(0)}</b> truly would have come back and <b>${(c.fp * k).toFixed(0)}</b> were false alarms. It misses <b>${(c.fn * k).toFixed(0)}</b> patients who do come back.</div>`;
    })));
  p.append(section("Sweeping the threshold",
    card({
      title: "Metrics at every threshold", height: 330, kind: "both",
      toggles: [{ key: "t", label: "Show as", value: "line", options: [["line", "Lines"], ["area", "Area"], ["stack", "Outcome counts"], ["heat", "Heatmap"]] },
        { key: "m", label: "Metric", type: "select", value: "ss", options: [["ss", "Sensitivity + specificity"], ["acc", "Accuracy"], ["prec", "Precision"], ["f1", "F1 score"], ["j", "Youden's J"], ["flag", "Share flagged"]] },
        { key: "s", label: "Models", value: "sel", options: [["sel", "Selected"], ["all", "All three"]] }],
      draw(st, cv, el) {
        if (st.t === "heat") {
          const key = st.m === "ss" ? "sens" : st.m, ts = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], name = st.m === "ss" ? "Sensitivity" : { acc: "Accuracy", prec: "Precision", f1: "F1 score", j: "Youden's J", flag: "Share flagged" }[st.m];
          const vals = ts.map(t => MODELS.map(m => confusion(D.probs[m], D.y_test, t)[key])), all = vals.flat().filter(v => !isNaN(v)), lo = Math.min(...all), hi = Math.max(...all);
          showCanvas(el, false).innerHTML = `<div class="scroll"><table class="heat"><tr><th>Rows: threshold (0 to 1)<br>Cells: ${name} (0 to 1)</th>${MODELS.map(m => `<th>${m}</th>`).join("")}</tr>${ts.map((t, i) => `<tr><th style="text-align:right">${t.toFixed(2)}</th>${vals[i].map(v => `<td style="background:color-mix(in srgb,${css("--violet")} ${Math.round(10 + 80 * (v - lo) / (hi - lo || 1))}%,transparent)">${isNaN(v) ? "-" : v.toFixed(2)}</td>`).join("")}</tr>`).join("")}</table></div>`;
          return;
        }
        showCanvas(el, true);
        const ts = []; for (let t = 0.1; t <= 0.7001; t += 0.01) ts.push(+t.toFixed(2));
        if (st.t === "stack") { const cs = ts.map(t => confusion(D.probs[S.model], D.y_test, t)), mkS = (label, key, col, i) => ({ label, data: cs.map(c => c[key] / c.n * 100), backgroundColor: col + "cc", borderWidth: 0, fill: i === 0 ? "origin" : "-1", pointRadius: 0 }); return mk(cv, { type: "line", data: { labels: ts.map(t => t.toFixed(2)), datasets: [mkS("True positive", "tp", css("--violet"), 0), mkS("False positive", "fp", css("--sun"), 1), mkS("False negative", "fn", css("--coral"), 2), mkS("True negative", "tn", css("--teal"), 3)] }, options: { scales: { x: { title: { display: true, text: "Threshold (predicted probability, 0 to 1)" }, ticks: { maxTicksLimit: 13 } }, y: { stacked: true, max: 100, title: { display: true, text: "Share of test patients (%)" } } } } }); }
        const fill = st.t === "area";
        const line = (m, key, col, lab, dash) => ({ label: lab, data: ts.map(t => confusion(D.probs[m], D.y_test, t)[key]), borderColor: col, backgroundColor: col + "26", fill, pointRadius: 0, tension: .2, borderDash: dash || [] });
        let ds = [];
        if (st.m === "ss" && st.s === "sel") ds = [line(S.model, "sens", css("--coral"), "Sensitivity"), line(S.model, "spec", css("--teal"), "Specificity")];
        else if (st.m === "ss") ds = MODELS.flatMap((m, i) => [line(m, "sens", MC(i), m + " sens."), line(m, "spec", MC(i), m + " spec.", [5, 4])]);
        else ds = (st.s === "sel" ? [S.model] : MODELS).map(m => line(m, st.m, MC(MODELS.indexOf(m)), m));
        const mi = ts.indexOf(+S.thr.toFixed(2));
        mk(cv, { type: "line", data: { labels: ts.map(t => t.toFixed(2)), datasets: ds }, options: { onClick: (e, _, ch) => { const ca = ch.chartArea; if (e.x < ca.left || e.x > ca.right) return; setThr(0.1 + (e.x - ca.left) / (ca.right - ca.left) * 0.6); }, onHover: (e, a, ch) => { ch.canvas.style.cursor = "pointer"; }, scales: { x: { title: { display: true, text: "Threshold (predicted probability, 0 to 1)" }, ticks: { maxTicksLimit: 13 } }, y: { min: st.m === "j" ? -.1 : 0, max: st.m === "j" ? .3 : 1 } }, plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } } },
          plugins: [{ id: "cur", afterDraw(ch) { const x = ch.scales.x.getPixelForValue(mi), { top, bottom } = ch.chartArea, c = ch.ctx; c.save(); c.strokeStyle = css("--ink"); c.setLineDash([4, 4]); c.beginPath(); c.moveTo(x, top); c.lineTo(x, bottom); c.stroke(); c.restore(); } }] });
      },
      caption: st => st.t === "stack" ? "At each threshold, the 100% height is split into the four outcomes for the selected model. Moving right, the coral missed readmissions grow and the yellow false alarms shrink." : st.t === "heat" ? "Each cell is one metric value for a model at a threshold. Darker violet means a higher value in this table. Pick a metric above to compare how each model behaves as the threshold moves." : "The dashed vertical line is the current threshold, and you can click anywhere on the chart to move it. Moving right means fewer patients get flagged: specificity rises and sensitivity falls. Try Youden's J to see where the balance peaks, and Share flagged to see how many patients get called."
    })));
  p.append(section("Choosing a threshold by cost",
    lede("Missing a patient who comes back is probably worse than making an extra follow up call. This lets you say how much worse."),
    card({
      title: "Expected cost per 100 patients", height: 300,
      toggles: [{ key: "r", label: "A missed readmission costs this many false alarms:", type: "range", min: 1, max: 10, step: 0.5, value: 3 },
        { key: "s", label: "Models", value: "sel", options: [["sel", "Selected"], ["all", "All three"]] }],
      draw(st, cv) {
        const ts = []; for (let t = 0.1; t <= 0.7001; t += 0.01) ts.push(+t.toFixed(2));
        const costs = m => ts.map(t => { const c = confusion(D.probs[m], D.y_test, t); return (c.fn * st.r + c.fp) / c.n * 100; });
        const P = sum(D.y_test), n = D.y_test.length, none = P * st.r / n * 100, every = (n - P) / n * 100;
        const list = st.s === "sel" ? [S.model] : MODELS, ds = list.map(m => ({ label: m, data: costs(m), borderColor: MC(MODELS.indexOf(m)), pointRadius: 0, tension: .2 }));
        const best = costs(S.model), bi = best.indexOf(Math.min(...best)); st._b = ts[bi];
        ds.push({ label: "Flag nobody", data: ts.map(() => none), borderColor: css("--muted"), borderDash: [6, 5], pointRadius: 0, borderWidth: 1 }, { label: "Flag everybody", data: ts.map(() => every), borderColor: css("--ink"), borderDash: [2, 4], pointRadius: 0, borderWidth: 1 });
        ds.push({ type: "scatter", label: "Cheapest threshold (selected model)", data: [{ x: ts[bi].toFixed(2), y: best[bi] }], pointRadius: 7, backgroundColor: css("--coral") });
        mk(cv, { type: "line", data: { labels: ts.map(t => t.toFixed(2)), datasets: ds }, options: { scales: { x: { title: { display: true, text: "Threshold (predicted probability, 0 to 1)" }, ticks: { maxTicksLimit: 13 } }, y: { title: { display: true, text: "Expected cost per 100 patients (1 false alarm = 1)" }, beginAtZero: true } }, plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } } } });
      },
      caption: st => `Lower is better. A false alarm costs 1 and a missed readmission costs ${st.r}. For the selected model the cheapest threshold is <b>${st._b.toFixed(2)}</b>. The dotted lines are the costs of flagging nobody or everybody, so a useful model should dip below both. As missed readmissions get more costly, the best threshold moves left.`
    })));
  p.append(section("Threshold table",
    card({
      title: "Selected model at common thresholds", kind: "html", toggles: [],
      draw(st, body) {
        const rows = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6].map(t => { const c = confusion(D.probs[S.model], D.y_test, t); return `<tr class="${Math.abs(t - S.thr) < 0.005 ? "hl" : ""}"><td>${t.toFixed(2)}</td><td class="n">${f3(c.acc)}</td><td class="n">${f3(c.sens)}</td><td class="n">${f3(c.spec)}</td><td class="n">${f3(c.prec)}</td><td class="n">${f3(c.j)}</td><td class="n">${pct(c.flag, 0)}</td></tr>`; }).join("");
        body.innerHTML = `<div class="scroll"><table><tr><th>Threshold</th><th class="n">Accuracy</th><th class="n">Sensitivity</th><th class="n">Specificity</th><th class="n">Precision</th><th class="n">Youden's J</th><th class="n">Flagged</th></tr>${rows}</table></div>`;
      },
      caption: "The highlighted row matches the slider. Youden's J is sensitivity + specificity - 1, so higher means a better balance."
    }),
    h(`<div class="callout warn"><h4>Where we would set the threshold</h4>
      <p>Missing a patient who comes back is worse than making an extra follow up call, so we want higher sensitivity. For the random forest, a threshold of <b>0.40</b> gave the best balance (sensitivity about 0.66 and specificity about 0.48), which is also where Youden's J is highest. At the default 0.50 the model only catches about 21% of readmissions. Going below about 0.35 flags almost every patient, which makes the model useless for deciding who to call first.</p></div>`)));
  return p;
};

/* ---------- 6. try a patient ---------- */
const CFG = { time_in_hospital: [1, 14], num_lab_procedures: [1, 132], num_procedures: [0, 6], num_medications: [1, 81], number_diagnoses: [1, 16] };
PAGES.try = () => {
  const p = h("<div></div>");
  p.append(titlebar("Try a patient", "Enter the details of a made up stay and see what each model predicts."));
  p.append(settingsBar());
  const presets = { Typical: FEATS.map(f => Math.round(quant(D.data[f], .5))), "Long, complex stay": [12, 80, 4, 40, 9, 1], "Short, simple stay": [2, 20, 0, 8, 4, 0] };
  const form = h(`<div class="card"><div style="margin-bottom:12px"><b>Presets:</b> ${Object.keys(presets).map(k => `<button class="btn" data-p="${k}">${k}</button>`).join(" ")}</div>
    <div class="sliders">${Object.entries(CFG).map(([f, [a, b]]) => `<label>${LBL[f]}: <output id="o_${f}"></output><input type="range" id="s_${f}" min="${a}" max="${b}"></label>`).join("")}</div>
    <label style="display:block;margin-top:14px"><input type="checkbox" id="s_dm"> Prescribed a diabetes medication</label></div>`);
  const setP = v => { FEATS.forEach((f, i) => { if (f === "diabetes_med") $("#s_dm", form).checked = !!v[i]; else $("#s_" + f, form).value = v[i]; }); upd(); };
  PATIENT = () => get(); TRY_SET = v => setP(v);
  const get = () => FEATS.map(f => f === "diabetes_med" ? ($("#s_dm", form).checked ? 1 : 0) : +$("#s_" + f, form).value);
  const upd = () => { Object.keys(CFG).forEach(f => $("#o_" + f, form).textContent = $("#s_" + f, form).value); refresh(); };
  form.oninput = upd; form.onclick = e => { const b = e.target.closest("[data-p]"); if (b) setP(presets[b.dataset.p]); };
  p.append(form);
  p.append(dyn(() => `<div class="pred">${MODELS.map(m => { const q = predict(m, get()), yes = q >= S.thr; return `<div class="pcard ${yes ? "yes" : ""}"><span class="tag">${m}</span><b>${(q * 100).toFixed(0)}%</b><div class="bar"><i style="width:${q * 100}%"></i><u style="left:${S.thr * 100}%"></u></div><span>${yes ? "<b>Predicted: readmitted</b>" : "Predicted: not readmitted"}</span><br><small style="color:var(--muted)">Black mark = threshold ${S.thr.toFixed(2)}</small></div>`; }).join("")}</div>`));
  p.append(card({
    title: "Why did the logistic regression say that?", height: 280,
    toggles: [{ key: "v", label: "Show as", value: "c", options: [["c", "Contribution to the score"], ["z", "Patient vs average"], ["r", "Radar (percentile)"]] }],
    draw(st, cv) {
      const x = get(), z = x.map((v, i) => (v - D.lr.mean[i]) / D.lr.scale[i]), c = z.map((v, i) => v * D.lr.coef[i]), d = st.v === "c" ? c : z;
      if (st.v === "r") return mk(cv, { type: "radar", data: { labels: FEATS.slice(0, 5).map(f => LBL[f]), datasets: [{ label: "This patient", data: FEATS.slice(0, 5).map((f, i) => D.data[f].filter(a => a <= x[i]).length / D.data[f].length * 100), borderColor: css("--coral"), backgroundColor: css("--coral") + "33" }, { label: "Typical stay (50th percentile)", data: [50, 50, 50, 50, 50], borderColor: css("--teal"), borderDash: [5, 4], backgroundColor: "transparent", pointRadius: 0 }] }, options: { scales: { r: { min: 0, max: 100, ticks: { backdropColor: "transparent" } } } } });
      mk(cv, { type: "bar", data: { labels: FEATS.map(f => LBL[f]), datasets: [{ data: d, backgroundColor: d.map(v => v >= 0 ? css("--coral") : css("--teal")), borderRadius: 5 }] }, options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: st.v === "c" ? "Push on the log-odds of readmission (positive = toward readmitted)" : "Distance from the average stay (standard deviations, + above, - below)" } } } } });
    },
    caption: st => st.v === "r" ? "Each corner is a variable. The farther the coral shape reaches, the higher this patient is compared with other stays. The dashed teal ring is the typical stay." : st.v === "c" ? "Coral bars push this patient's prediction toward readmitted and teal bars push it away. The bars add up (with the intercept) to the logistic regression's score. Notice how small each push is." : "This shows how unusual each value is compared with the average stay in our data, measured in standard deviations."
  }));
  p.append(card({
    title: "Where this patient sits in the data", kind: "html", toggles: [],
    draw(st, body) {
      const x = get();
      body.innerHTML = `<div class="scroll"><table><tr><th>Variable</th><th class="n">Patient</th><th class="n">Average</th><th class="n">Percentile</th></tr>${FEATS.map((f, i) => { const v = D.data[f]; return `<tr><td>${LBL[f]}</td><td class="n">${f === "diabetes_med" ? (x[i] ? "Yes" : "No") : x[i]}</td><td class="n">${f === "diabetes_med" ? pct(mean(v), 0) + " yes" : mean(v).toFixed(1)}</td><td class="n">${f === "diabetes_med" ? "" : Math.round(v.filter(a => a <= x[i]).length / v.length * 100) + "th"}</td></tr>`; }).join("")}</table></div>`;
    },
    caption: "The percentile tells you what share of all 3,000 stays had this value or lower."
  }));
  p.append(card({
    title: "What if? Change one variable and watch the prediction", height: 320,
    toggles: [{ key: "v", label: "Vary", type: "select", value: "number_diagnoses", options: VAR_OPTS.slice(0, 5) }, { key: "t", label: "Show as", value: "line", options: [["line", "Lines"], ["area", "Area"]] }],
    draw(st, cv) {
      const f = st.v, j = FEATS.indexOf(f), [lo, hi] = CFG[f], base = get(), step = Math.max(1, Math.round((hi - lo) / 40)), xs = []; for (let v = lo; v <= hi; v += step) xs.push(v);
      const ds = MODELS.map((m, i) => ({ label: m, data: xs.map(v => { const x = base.slice(); x[j] = v; return predict(m, x) * 100; }), borderColor: MC(i), backgroundColor: MC(i) + "22", fill: st.t === "area", pointRadius: 0, tension: .2 }));
      ds.push({ label: `Threshold ${S.thr.toFixed(2)}`, data: xs.map(() => S.thr * 100), borderColor: css("--ink"), borderDash: [5, 4], pointRadius: 0, borderWidth: 1 });
      mk(cv, { type: "line", data: { labels: xs, datasets: ds }, options: { scales: { x: { title: { display: true, text: axT(f) }, ticks: { maxTicksLimit: 14 } }, y: { title: { display: true, text: "Predicted chance of readmission (%)" } } }, plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } } } });
    },
    caption: "This takes the patient you built above and changes only one variable. Each line is a model's predicted chance of readmission. Where a line crosses the dashed threshold, the prediction flips. The lines are fairly flat, which shows how weak each variable is on its own."
  }));
  setTimeout(() => setP(presets.Typical), 0);
  return p;
};

/* ---------- 7. results ---------- */
PAGES.results = () => {
  const p = h("<div></div>");
  p.append(titlebar("Results and conclusion", "Our answer to the research question."));
  p.append(section("The answer",
    h(`<div class="callout warn"><h4>These variables do not predict readmission very accurately.</h4>
      <p>Our best model, the random forest, reached an AUROC of about 0.59. A score of 0.5 is a coin flip and 1.0 is perfect, so this is only slightly better than guessing. Its accuracy of 61% at a threshold of 0.5 barely beats always guessing not readmitted, which is 60%. When we lowered the threshold to 0.40, the model caught about two thirds of readmissions, but it also flagged about half of the patients who did not come back.</p></div>`),
    dyn(() => {
      const rows = MODELS.map(m => { const a = confusion(D.probs[m], D.y_test, 0.5), b = confusion(D.probs[m], D.y_test, 0.4); return `<tr><td>${m}</td><td class="n">${f3(ROC[m].auc)}</td><td class="n">${f3(a.acc)}</td><td class="n">${f3(a.sens)}</td><td class="n">${f3(a.spec)}</td><td class="n">${f3(b.acc)}</td><td class="n">${f3(b.sens)}</td><td class="n">${f3(b.spec)}</td></tr>`; }).join("");
      return `<div class="card scroll"><table><tr><th></th><th class="n">AUROC</th><th class="n">Acc. @0.50</th><th class="n">Sens. @0.50</th><th class="n">Spec. @0.50</th><th class="n">Acc. @0.40</th><th class="n">Sens. @0.40</th><th class="n">Spec. @0.40</th></tr>${rows}</table></div>`;
    })));
  p.append(section("Scoreboard",
    card({
      title: "Default threshold vs our threshold", height: 300,
      toggles: [{ key: "m", label: "Metric", type: "select", value: "sens", options: [["sens", "Sensitivity"], ["spec", "Specificity"], ["acc", "Accuracy"], ["prec", "Precision"], ["f1", "F1 score"], ["j", "Youden's J"]] }, { key: "t", label: "Show as", value: "bar", options: [["bar", "Bars"], ["radar", "Radar"]] }],
      draw(st, cv) {
        const v = (t, m) => confusion(D.probs[m], D.y_test, t)[st.m];
        if (st.t === "radar") return mk(cv, { type: "radar", data: { labels: ["Sensitivity", "Specificity", "Accuracy", "Precision", "F1"], datasets: [0.5, 0.4].map((t, i) => ({ label: `${S.model} at ${t.toFixed(2)}`, data: ["sens", "spec", "acc", "prec", "f1"].map(k => confusion(D.probs[S.model], D.y_test, t)[k]), borderColor: [css("--violet"), css("--coral")][i], backgroundColor: [css("--violet"), css("--coral")][i] + "22" })) }, options: { scales: { r: { min: 0, max: 1, ticks: { backdropColor: "transparent" } } } } });
        mk(cv, { type: "bar", data: { labels: MODELS, datasets: [0.5, 0.4].map((t, i) => ({ label: `Threshold ${t.toFixed(2)}`, data: MODELS.map(m => v(t, m)), backgroundColor: [css("--violet"), css("--coral")][i], borderRadius: 6 })) }, options: { scales: { y: { min: st.m === "j" ? -.1 : 0, max: st.m === "j" ? .3 : 1 } } } });
      },
      caption: st => st.t === "radar" ? "Two shapes for the selected model: threshold 0.50 (violet) and threshold 0.40 (coral). Lowering the threshold stretches the sensitivity corner but pulls in specificity and precision." : "Violet bars use the default 0.50 threshold and coral bars use our 0.40 threshold. Pick sensitivity to see how much more of the readmissions the lower threshold catches."
    })));
  p.append(section("What this means",
    para("The numbers recorded during a stay, like days in the hospital, tests, and medications, are only weakly related to coming back. The number of diagnoses was the variable the models leaned on most, but even that does not separate the two groups well. Things like prior admissions, where the patient goes after discharge, and lab results such as HbA1c would probably help more, but they are not in this simplified file."),
    para("Because of that, the model could only be a rough screening tool, and it should not be used alone to make decisions about a patient.")));
  p.append(section("Limitations",
    h(`<div class="card"><ul><li>The sample is only 3,000 stays, and we used one random split. The bootstrap on the Models tab shows the AUROC could change by about 0.02 (roughly plus or minus 0.04 for a 95% range) with a different split.</li>
      <li>We did not tune the models, so a different setup might do slightly better.</li>
      <li>The data is from 1999 to 2008, and practice has changed since then.</li>
      <li>We left out age, gender and admission type on purpose. Adding them is a good next step.</li>
      <li>Importance and weights show what a model used, not what causes readmission.</li></ul></div>`)));
  p.append(section("Where to find each part of the project",
    h(`<div class="card scroll"><table><tr><th>Requirement</th><th>Where</th></tr>
      <tr><td>Research question, problem and why it matters</td><td><a href="#overview" data-go>Overview</a></td></tr>
      <tr><td>Dataset description, source and columns</td><td><a href="#dataset" data-go>Dataset</a></td></tr>
      <tr><td>Analysis process, models and how they were evaluated</td><td><a href="#models" data-go>Models</a> and <a href="#threshold" data-go>Threshold</a></td></tr>
      <tr><td>Best model and threshold choice</td><td><a href="#models" data-go>Models</a>, <a href="#threshold" data-go>Threshold</a></td></tr>
      <tr><td>Classifying a new patient</td><td><a href="#try" data-go>Try a patient</a></td></tr>
      <tr><td>Final conclusions</td><td>This page</td></tr>
      <tr><td>Full code and analysis notebook</td><td><a href="notebook.html">Notebook (rendered)</a> · <a href="https://github.com/ryanmerlino21-debug/CAI5735-Final-Project-Group-6">GitHub repository</a></td></tr></table></div>`)));
  return p;
};

/* ---------- 8. glossary ---------- */
const GLOSS = [
  ["Readmission", "When a patient comes back to the hospital after being discharged. Here we count any return, whether it was within 30 days or after."],
  ["Predictor", "A variable we give the model to help it make a prediction, like days in the hospital."],
  ["Response", "The thing we want to predict. In this project it is whether the patient was readmitted."],
  ["Training and test sets", "We train the models on 75% of the stays and test them on the other 25%. Testing on stays the model has not seen shows how it would do on new patients."],
  ["Stratified split", "A split that keeps the share of readmitted patients about the same in the training and test sets."],
  ["Baseline accuracy", "The accuracy you get by always guessing the most common outcome. Here that is not readmitted, which is about 60%. A model has to beat this to be useful."],
  ["Classification model", "A model that predicts a category, like readmitted or not readmitted, instead of a number."],
  ["Logistic regression", "A model that adds up weighted versions of each variable to produce a probability. Easy to explain."],
  ["Decision tree", "A model that asks a series of yes or no questions about the variables and follows the answers to a prediction."],
  ["Random forest", "Many decision trees trained on random parts of the data whose predictions are averaged together."],
  ["Overfitting", "When a model memorizes quirks of the training data and then does worse on new data. We limited our tree and forest depth to avoid it."],
  ["Threshold", "The predicted probability at which we label a patient as readmitted. Lowering it flags more patients."],
  ["Confusion matrix", "A 2 by 2 table of the model's predictions against what really happened: true positives, true negatives, false positives and false negatives."],
  ["True positive / false negative", "A true positive is a readmitted patient the model flagged. A false negative is a readmitted patient the model missed."],
  ["False positive / true negative", "A false positive is a patient who did not come back but was flagged. A true negative is a patient correctly left unflagged."],
  ["Accuracy", "The share of all predictions that were correct."],
  ["Sensitivity (recall)", "Of the patients who really were readmitted, the share the model caught."],
  ["Specificity", "Of the patients who were not readmitted, the share the model correctly left alone."],
  ["Precision", "Of the patients the model flagged, the share who really were readmitted."],
  ["F1 score", "A single number that balances precision and sensitivity."],
  ["ROC curve", "A curve showing sensitivity against the false positive rate for every possible threshold. A curve near the top left is good."],
  ["AUROC", "The area under the ROC curve. 0.5 is a coin flip and 1.0 is perfect. Ours is about 0.59."],
  ["Precision-recall curve", "A curve of precision against sensitivity across thresholds. It is useful when one outcome is less common."],
  ["Youden's J", "Sensitivity + specificity - 1. It is highest where the model balances both well."],
  ["Correlation", "A number from -1 to 1 that shows how strongly two variables rise and fall together. Near 0 means little relationship."],
  ["Effect size (Cohen's d)", "How far apart two group averages are, measured in standard deviations. About 0.2 is small."],
  ["Bootstrap", "Resampling the test set with replacement many times to see how much a result such as the AUROC could change by luck."],
  ["Variable importance", "How much a model relied on each variable. It shows what the model used, not what causes the outcome."],
  ["Odds ratio", "How much the odds of readmission are multiplied when a variable goes up by one unit. 1.00 means no effect."]
];
PAGES.glossary = () => {
  const p = h("<div></div>");
  p.append(titlebar("Glossary", "Plain meanings for the terms used in this app."));
  const box = h(`<input class="search" type="search" placeholder="Search terms, for example sensitivity" aria-label="Search glossary">`), dl = h(`<dl class="gl"></dl>`);
  const draw = q => { dl.innerHTML = GLOSS.filter(([t, d]) => (t + d).toLowerCase().includes(q.toLowerCase())).map(([t, d]) => `<dt>${t}</dt><dd>${d}</dd>`).join("") || "<dd>No matching terms.</dd>"; };
  box.oninput = () => draw(box.value); draw(""); p.append(box, dl);
  return p;
};

/* ================= shell: tabs, routing, display menu ================= */
const TABS = [["overview", "Research question", "🏠"], ["dataset", "Dataset", "📋"], ["explorer", "Interactive explorer", "🧭"], ["explore", "Explore", "🔍"], ["models", "Analysis: Models", "🧠"], ["threshold", "Analysis: Threshold", "🎚️"], ["try", "Try a patient", "🧪"], ["results", "Results", "🏁"], ["glossary", "Glossary", "📖"]];
function addChips(page) {
  const hs = $$("section.block > h3", page); if (hs.length < 3) return;
  hs.forEach((x, i) => x.id = x.id || `${S.tab}-${i}`);
  page.querySelector(".titlebar").after(h(`<div class="chips"><b>On this page</b><div class="chiplist">${hs.map(x => `<a href="#${x.id}" data-jump="${x.id}">${x.textContent}</a>`).join("")}</div></div>`));
}
function show(tab) {
  if (!PAGES[tab]) tab = "overview";
  S.tab = tab;
  $$("#tabs a").forEach(a => { a.classList.toggle("on", a.dataset.t === tab); if (a.dataset.t === tab) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  const main = $("#main");
  $$(".page", main).forEach(pg => pg.classList.remove("on"));
  if (!BUILT[tab]) { REG[tab] = []; const pg = PAGES[tab](); pg.className = "page"; pg.dataset.tab = tab; main.append(pg); addChips(pg); BUILT[tab] = pg; }
  BUILT[tab].classList.add("on");
  $$(".settings", BUILT[tab]).forEach(s => s.upd());
  refresh();
  document.title = "CAI5735 Final Project - Group 6";
}
function route() { const id = location.hash.slice(1), tab = TABS.find(t => t[0] === id); if (tab) { show(id); window.scrollTo(0, $(".wrap").offsetTop - 8); } else if (id && document.getElementById(id)) document.getElementById(id).scrollIntoView(); else if (!id) show("overview"); }
function setTheme(dark) { document.documentElement.dataset.theme = dark ? "dark" : "light"; try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch (e) { } themeCharts(); syncMenu(); refresh(); }
function syncMenu() { $("#mDark .sw").classList.toggle("on", document.documentElement.dataset.theme === "dark"); $("#mBig .sw").classList.toggle("on", document.documentElement.dataset.big === "on"); }

fetch("data.json").then(r => r.json()).then(d => {
  D = d; themeCharts();
  MODELS.forEach(m => { const c = curves(D.probs[m], D.y_test); ROC[m] = { pts: c.roc, auc: c.auc }; ROC[m].roc = c.roc; PR[m] = { pr: c.pr, ap: c.ap }; });
  const best = MODELS.reduce((a, b) => ROC[a].auc > ROC[b].auc ? a : b);
  $("#heroPills").innerHTML = [`${D.readmitted.length.toLocaleString()} stays`, "6 predictors", "3 models", `best AUROC ${f3(ROC[best].auc)}`, "chosen threshold 0.40"].map(t => `<span class="pill">${t}</span>`).join("");
  $("#tabs").innerHTML = TABS.map(([id, l, ic]) => `<a class="tab" href="#${id}" data-t="${id}"><span class="n">${ic}</span>${l}</a>`).join("");
  document.addEventListener("click", e => {
    const a = e.target.closest("a[data-go]"); if (a) { e.preventDefault(); location.hash = a.getAttribute("href"); }
    const j = e.target.closest("a[data-jump]"); if (j) { e.preventDefault(); document.getElementById(j.dataset.jump).scrollIntoView({ behavior: "smooth" }); }
    const m = $("#menu"); if (m.open && !m.contains(e.target)) m.open = false;
  });
  $("#mDark").onclick = () => setTheme(document.documentElement.dataset.theme !== "dark");
  $("#mBig").onclick = () => { const on = document.documentElement.dataset.big !== "on"; if (on) document.documentElement.dataset.big = "on"; else delete document.documentElement.dataset.big; try { localStorage.setItem("big", on ? "on" : "off"); } catch (e) { } syncMenu(); refresh(); };
  syncMenu(); window.addEventListener("hashchange", route); route();
  if (!location.hash) show("overview");
});
