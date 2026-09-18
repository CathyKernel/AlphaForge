/* AlphaForge dashboard — application logic.
   Fetches JSON from the FastAPI backend and renders every view. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = AF.fmt, pct = AF.pct, sf = AF.sf;
const pf = (v, d = 1) => (v == null ? "–" : sf(v) + (100 * v).toFixed(d) + "%"); // signed pct

async function getJSON(url, opts) {
  let r = await fetch(url, opts);
  // Static hosting (Netlify snapshot mode) cannot answer POST requests —
  // fall back to the pre-rendered GET snapshot for the same URL.
  if (!r.ok && opts && opts.method === "POST") r = await fetch(url);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* noop */ }
    throw new Error(detail);
  }
  return r.json();
}

/* Static-hosting detection: netlify.json exists only in the published static
   bundle (Netlify), never when the FastAPI backend serves this page. */
let AF_STATIC = false;
fetch("netlify.json", { method: "HEAD" })
  .then((r) => { AF_STATIC = r.ok; })
  .catch(() => { AF_STATIC = false; });

/* Factor-table URL: live API query, or pre-rendered snapshot per horizon. */
const factorTableURL = (h) => (AF_STATIC ? `/api/snap/factors_h${h}.json` : `/api/factors?horizon=${h}`);

function toast(msg, ms = 2600) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast._h);
  toast._h = setTimeout(() => t.classList.add("hidden"), ms);
}

/* ================= tabs ================= */
const TABS = ["overview", "factors", "backtest", "ml", "risk", "pit"];
function selectTab(name) {
  for (const t of TABS) {
    $("tab-" + t).setAttribute("aria-selected", String(t === name));
    $("panel-" + t).classList.toggle("hidden", t !== name);
  }
  const lazy = { factors: renderFactors, ml: renderML, risk: renderRisk, pit: renderPIT, backtest: null };
  if (lazy[name] && !loaded[name]) { lazy[name]().catch((e) => toast("⚠ " + e.message)); loaded[name] = true; }
  window.dispatchEvent(new Event("resize"));
}
TABS.forEach((t) => $("tab-" + t).addEventListener("click", () => selectTab(t)));

const loaded = { overview: false, factors: false, ml: false, risk: false, pit: false };

/* ================= overview ================= */
async function renderOverview() {
  const ov = await getJSON("/api/overview");
  loaded.overview = true;

  // ticker strip
  $("ticker-strip").innerHTML = [
    ["universe", `${ov.data.tickers} tickers`],
    ["panel", `${ov.data.trading_days.toLocaleString()} days`],
    ["span", `${ov.data.first_date} → ${ov.data.last_date}`],
    ["factors", `${ov.factors.n_factors} / ${ov.factors.categories.length} categories`],
  ].map(([k, v]) => `<span class="tk">${k} <b>${v}</b></span>`).join("");

  // hero stats: composite + ML
  const comp = ov.flagship.composite, mlb = ov.flagship.ml_ensemble;
  const best = comp && mlb ? (mlb.metrics.sharpe >= comp.metrics.sharpe ? mlb : comp) : (comp || mlb);
  const rows = [];
  if (best) {
    const m = best.metrics;
    rows.push(
      stat("Net CAGR", pf(m.cagr), `${best.label}`, m.cagr >= 0),
      stat("Sharpe", fmt(m.sharpe), `bench ${fmt(m.benchmark_sharpe)}`, m.sharpe >= 0),
      stat("Max DD", pf(m.max_drawdown), m.max_drawdown > -0.25 ? "contained" : "severe", false),
      stat("Info ratio", pf(m.information_ratio, 2), `TE ${fmt(m.tracking_error)}`, m.information_ratio >= 0),
    );
  }
  // prefer the LSTM reference run; fall back to the first row of whichever
  // ML summary the panel serves (the S&P 500 run has lightgbm/ridge/ensemble)
  const ml0 = (ov.ml || []).find((r) => r.model === "lstm") || (ov.ml || [])[0];
  if (ml0)
    rows.push(
      stat(
        `ML OOS IC (${ml0.model})`,
        fmt(ml0.ic_mean, 4),
        `t = ${fmt(ml0.ic_tstat, 1)}`,
        ml0.ic_mean > 0,
      ),
    );
  rows.push(stat("Data", `${ov.data.tickers} × ${(+ov.data.last_date.slice(0, 4) - +ov.data.first_date.slice(0, 4))}y`, "Yahoo Finance, adj.", true));
  $("ov-stats").innerHTML = rows.map((h) => `<div class="stat"><div class="k">${h.k}</div><div class="v ${h.cls || ""}">${h.v}</div><div class="s">${h.s}</div></div>`).join("");

  // flagship equity curves
  if (comp && mlb) {
    const dates = comp.dates;
    AF.lineChart($("ov-flagship-chart"), [
      { name: "SPY (benchmark)", values: comp.benchmark_equity, color: "#5c6878", tipFmt: (v) => fmt(v) },
      { name: `factor composite — Sharpe ${fmt(comp.metrics.sharpe)}`, values: scaleTo(comp.equity, dates.length), color: "#8b9dff" },
      { name: `ML ensemble OOS — Sharpe ${fmt(mlb.metrics.sharpe)}`, values: scaleTo(mlb.equity, dates.length), color: "#14b8a6" },
    ], { dates: comp.dates, legend: $("ov-flagship-legend") });
  }

  // headline tables
  $("ov-flagship-tables").innerHTML = [comp && [comp, "Factor composite (top-5 ICIR)"], mlb && [mlb, "ML ensemble (OOS)"]]
    .filter(Boolean).map(([b, label]) => {
      const m = b.metrics;
      return `<h3 class="ft-title">${label}</h3>
        <table class="table kv">
        ${kvRow("Net CAGR", pf(m.cagr))}${kvRow("Benchmark CAGR", pf(m.benchmark_cagr))}
        ${kvRow("Sharpe / Sortino", `${fmt(m.sharpe)} / ${fmt(m.sortino)}`)}
        ${kvRow("Max drawdown", pf(m.max_drawdown))}${kvRow("Info ratio", pf(m.information_ratio, 2))}
        ${kvRow("α / β", `${fmt(m.alpha, 3)} / ${fmt(m.beta)}`)}
        ${kvRow("VaR 95 (hist / CF)", `${pct(m.var95_historical)} / ${pct(m.var95_cornish_fisher)}`)}
        ${kvRow("Days positive", pct(m.pct_positive_days, 0))}</table>`;
    }).join("");

  // ml mini cards
  $("ov-ml-cards").innerHTML = (ov.ml || []).map((r) => {
    const cls = r.ic_tstat >= 2 ? "up" : r.ic_tstat >= 1 ? "" : "down";
    return `<div class="stat"><div class="k">${r.model} OOS IC</div>
      <div class="v ${cls}">${fmt(r.ic_mean, 4)}</div>
      <div class="s">t ${fmt(r.ic_tstat, 2)} · IR ${fmt(r.ic_ir, 2)} · ${Math.round(100 * r.ic_positive_rate)}% pos</div></div>`;
  }).join("") || '<p class="card-note">run scripts/train_models.py first</p>';

  // quality table
  $("ov-quality").innerHTML = `<table class="table"><thead><tr><th>check</th><th>status</th><th>flagged</th></tr></thead><tbody>${
    ov.quality.map((q) => `<tr><td class="txt">${q.name}</td><td>${
      q.passed ? '<span class="pill ok">pass</span>' : `<span class="pill ${q.severity === "error" ? "fail" : "warn"}">${q.severity}</span>`
    }</td><td>${q.n_flagged}</td></tr>`).join("")}</tbody></table>`;
}

function stat(k, v, s, good) {
  return { k, v, s, cls: good === true ? "up" : good === false ? "down" : "" };
}
function kvRow(k, v) { return `<tr><td>${k}</td><td>${v}</td></tr>`; }
function scaleTo(values, n) {  // resample series to common length (front-end only)
  if (values.length === n) return values;
  const out = new Array(n).fill(null);
  const r = values.length / n;
  for (let i = 0; i < n; i++) out[i] = values[Math.floor(i * r)];
  return out;
}

/* ================= factors ================= */
let fRows = [], fSort = { key: "ic_ir", dir: -1 };
async function renderFactors() {
  await refreshFactorTable(5);
  $("f-horizon").addEventListener("change", () => refreshFactorTable(+$("f-horizon").value));
  $("f-category").addEventListener("change", applyFactorFilters);
  $("f-search").addEventListener("input", applyFactorFilters);
}

async function refreshFactorTable(h) {
  const data = await getJSON(factorTableURL(h));
  fRows = data.factors;
  const cats = [...new Set(fRows.map((r) => r.category))].sort();
  const sel = $("f-category"), cur = sel.value || "all";
  sel.innerHTML = '<option value="all">all</option>' + cats.map((c) => `<option${c === cur ? " selected" : ""}>${c}</option>`).join("");
  applyFactorFilters();
}

function applyFactorFilters() {
  const cat = $("f-category").value, q = $("f-search").value.trim().toLowerCase();
  let rows = fRows.filter((r) => (cat === "all" || r.category === cat) &&
    (!q || r.name.includes(q) || (r.description || "").toLowerCase().includes(q)));
  rows = sortRows(rows, fSort);
  $("f-table").innerHTML = `<thead><tr>
    <th data-k="name">factor</th><th data-k="category">class</th>
    <th data-k="ic_mean">IC</th><th data-k="ic_ir">ICIR</th><th data-k="ic_tstat">NW t</th>
    <th data-k="ic_positive_rate">% pos</th><th data-k="quantile_spread_ann">Q sprd</th>
    <th data-k="quantile_spread_sharpe">Q Sharpe</th><th data-k="monotonicity">mono</th>
    <th data-k="turnover_daily">TO</th></tr></thead><tbody>` +
    rows.map((r) => `<tr class="clickable" data-name="${r.name}">
      <td title="${r.description || ""}">${r.name}</td><td class="txt">${r.category}</td>
      <td class="${r.ic_mean > 0 ? "up" : "down"}">${fmt(r.ic_mean, 4)}</td>
      <td class="${r.ic_ir > 0 ? "up" : "down"}">${fmt(r.ic_ir, 2)}</td>
      <td>${fmt(r.ic_tstat, 1)}</td><td>${pct(r.ic_positive_rate, 0)}</td>
      <td class="${r.quantile_spread_ann > 0 ? "up" : "down"}">${pf(r.quantile_spread_ann, 1)}</td>
      <td class="${r.quantile_spread_sharpe > 0 ? "up" : "down"}">${fmt(r.quantile_spread_sharpe, 2)}</td>
      <td>${fmt(r.monotonicity, 1)}</td><td>${fmt(r.turnover_daily)}</td></tr>`).join("") +
    "</tbody>";
  // sorting on header click
  $("f-table").querySelectorAll("th").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.k;
    fSort = { key: k, dir: fSort.key === k ? -fSort.dir : (k === "name" || k === "category" ? 1 : -1) };
    applyFactorFilters();
  }));
  // IC decay on row click
  $("f-table").querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => {
    $("f-table").querySelectorAll("tr.active").forEach((x) => x.classList.remove("active"));
    tr.classList.add("active");
    showDecay(tr.dataset.name);
  }));
}

function sortRows(rows, { key, dir }) {
  return [...rows].sort((a, b) => {
    const va = a[key], vb = b[key];
    const c = typeof va === "string" ? va.localeCompare(vb) : (va - vb);
    return c * dir;
  });
}

async function showDecay(name) {
  try {
    const hz = $("f-horizon").value;
    const d = await getJSON(AF_STATIC
      ? `/api/snap/decay/${name}_h${hz}.json`
      : `/api/factors/decay?name=${encodeURIComponent(name)}&horizon=${hz}`);
    $("f-decay-card").classList.remove("hidden");
    $("f-decay-name").textContent = name;
    AF.barChart($("f-decay-chart"), d.horizons.map(String), d.ic, {
      color: (v) => (v >= 0 ? "#14b8a6" : "#ef4444"),
      yFmt: (v) => fmt(v, 3),
    });
    $("f-decay-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) { toast("⚠ " + e.message); }
}

/* ================= backtest ================= */
const BT_DEFAULT = { factor: "composite_top5", rebalance: "W-FRI", side: "long_short", top_n: 15, cost_bps: 10 };

async function initBacktest() {
  const { factors } = await getJSON(factorTableURL(5));
  $("b-factor").innerHTML = '<option value="composite_top5">composite_top5 (top-5 ICIR)</option>' +
    factors.map((f) => `<option value="${f.name}">${f.name}</option>`).join("");
  $("b-topn").addEventListener("input", () => ($("b-topn-out").textContent = $("b-topn").value));
  $("b-cost").addEventListener("input", () => ($("b-cost-out").textContent = $("b-cost").value));
  $("b-run").addEventListener("click", runBacktest);
  // auto-run the flagship default once
  runBacktest();
}

async function runBacktest() {
  const btn = $("b-run"), st = $("b-status");
  btn.disabled = true;
  st.classList.remove("hidden", "err");
  const t0 = performance.now();
  st.textContent = AF_STATIC ? "loading pre-rendered snapshot …" : "running vectorised backtest …";
  try {
    const body = {
      factor: $("b-factor").value,
      rebalance: $("b-rebalance").value,
      side: $("b-side").value,
      top_n: +$("b-topn").value,
      cost_bps: +$("b-cost").value,
    };
    const b = await getJSON("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    // On static hosting only the flagship default is pre-rendered; custom
    // parameters need the live backend, so render the snapshot with its own
    // true labels instead of the requested ones.
    const req = AF_STATIC && JSON.stringify(body) !== JSON.stringify(BT_DEFAULT) ? BT_DEFAULT : body;
    if (req !== body) {
      toast("static demo — custom parameters need the local FastAPI backend (README: Deployment)", 4200);
    }
    st.textContent = AF_STATIC
      ? "pre-rendered snapshot — run the local backend for live backtests"
      : `done in ${((performance.now() - t0) / 1000).toFixed(1)} s — ${b.n_trading_days.toLocaleString()} trading days`;
    renderBacktest(b, req);
  } catch (e) {
    st.classList.add("err");
    st.textContent = "error: " + e.message;
  } finally { btn.disabled = false; }
}

function renderBacktest(b, req) {
  $("b-results").classList.remove("hidden");
  const m = b.metrics;
  $("b-title").textContent = `${b.name} — ${req.rebalance}, ${req.side}, top ${req.top_n}, ${req.cost_bps} bps/side`;

  $("b-stats").innerHTML = [
    stat("Net CAGR", pf(m.cagr), `bench ${pf(m.benchmark_cagr)}`, m.cagr >= 0),
    stat("Sharpe", fmt(m.sharpe), `bench ${fmt(m.benchmark_sharpe)}`, m.sharpe >= 0),
    stat("Max DD", pf(m.max_drawdown), `calmar ${fmt(m.calmar, 2)}`, false),
    stat("Info ratio", pf(m.information_ratio, 2), `TE ${fmt(m.tracking_error)}`, m.information_ratio >= 0),
    stat("Avg turnover", fmt(b.turnover_mean), "per rebalance", null),
    stat("VaR 95 CF", pct(m.var95_cornish_fisher), `CVaR ${pct(m.cvar95)}`, false),
  ].map((h) => `<div class="stat"><div class="k">${h.k}</div><div class="v ${h.cls || ""}">${h.v}</div><div class="s">${h.s}</div></div>`).join("");

  AF.lineChart($("b-equity-chart"), [
    { name: "strategy (net)", values: b.equity, color: "#14b8a6" },
    { name: "benchmark", values: b.benchmark_equity, color: "#5c6878" },
  ], { dates: b.dates, legend: $("b-legend") });

  AF.lineChart($("b-dd-chart"), [
    { name: "drawdown", values: b.drawdown, color: "#ef4444", tipFmt: (v) => pf(v) },
  ], { dates: b.dates, area: true, zeroBase: true });

  AF.lineChart($("b-rs-chart"), [
    { name: "rolling 126d Sharpe", values: b.rolling_sharpe, color: "#8b9dff" },
  ], { dates: b.dates, zeroBase: true });

  AF.monthlyHeatmap($("b-monthly"), b.monthly);

  const mo = { total_return: "Total return", cagr: "CAGR", benchmark_cagr: "Bench CAGR", ann_vol: "Ann vol", sharpe: "Sharpe", sortino: "Sortino", calmar: "Calmar", max_drawdown: "Max DD", alpha: "Alpha", beta: "Beta", up_capture: "Up capture", down_capture: "Down capture", var95_historical: "VaR95 hist", var95_cornish_fisher: "VaR95 CF", cvar95: "CVaR95", skew: "Skew", excess_kurtosis: "Ex. kurtosis", pct_positive_days: "% pos days", tracking_error: "Tracking err", information_ratio: "Info ratio" };
  $("b-metrics").innerHTML = "<tbody>" + Object.entries(m).map(([k, v]) => {
    const isPct = /return|cagr|drawdown|capture|var|cvar|days/.test(k) && Math.abs(v) < 3;
    return `<tr><td>${mo[k] || k}</td><td>${isPct ? pf(v) : fmt(v, 3)}</td></tr>`;
  }).sort((a, c) => a[0].localeCompare(c[0])) + "</tbody>";

  $("b-holdings").innerHTML = `<thead><tr><th>ticker</th><th>weight</th></tr></thead><tbody>${
    b.holdings.map((h) => `<tr><td>${h.ticker}</td><td>${pct(h.weight, 1)}</td></tr>`).join("")}</tbody>`;
}

/* ================= ML ================= */
let mlData = null;
async function renderML() {
  if (!mlData) mlData = await getJSON("/api/ml");
  const d = mlData;

  $("ml-cards").innerHTML = d.ic_summary.map((r) => {
    const cls = r.ic_tstat >= 2 ? "up" : r.ic_tstat >= 1 ? "" : "down";
    return `<div class="stat"><div class="k">${r.model}</div>
      <div class="v ${cls}">${fmt(r.ic_mean, 4)}</div>
      <div class="s">IC · t ${fmt(r.ic_tstat, 2)} · IR ${fmt(r.ic_ir, 2)}</div>
      <div class="s">${Math.round(100 * r.ic_positive_rate)}% positive days · ${r.n_days.toLocaleString()} OOS days</div></div>`;
  }).join("");

  drawMLSeries($("ml-model").value);

  // importance hbars
  const imp = d.importance.slice(0, 12);
  const mx = imp[0] ? imp[0].importance : 1;
  $("ml-importance").innerHTML = imp.map((f) => `
    <div class="hbar-row"><span class="hb-n">${f.feature}</span>
    <div class="hb-track"><div class="hb-fill" style="width:${(100 * f.importance / mx).toFixed(1)}%"></div></div>
    <span class="hb-v">${fmt(f.importance, 3)}</span></div>`).join("");
}

function drawMLSeries(model) {
  const d = mlData;
  const vals = d.ic_series[model];
  const cum = [];
  let acc = 0;
  for (const v of vals) { acc += v == null ? 0 : v; cum.push(acc); }
  AF.lineChart($("ml-cum-chart"), [
    { name: `cumulative IC — ${model}`, values: cum, color: "#14b8a6" },
  ], { dates: d.ic_dates, legend: $("ml-legend") });
  AF.lineChart($("ml-daily-chart"), [
    { name: `daily IC — ${model}`, values: vals, color: "#8b9dff" },
  ], { dates: d.ic_dates, area: true, zeroBase: true });
}

/* ================= risk & data ================= */
async function renderRisk() {
  const [r, u] = await Promise.all([getJSON("/api/risk"), getJSON("/api/universe")]);

  const rs = r.risk_summary, m = r.metrics;
  $("r-cards").innerHTML = [
    stat("VaR 95 (hist)", pct(rs.var95_historical), `CF ${pct(rs.var95_cornish_fisher)}`, false),
    stat("CVaR 95", pct(rs.cvar95), `worst day ${pct(rs.worst_day)}`, false),
    stat("Worst week", pct(rs.worst_week), "rolling 5d", false),
    stat("Max drawdown", pct(rs.max_drawdown), `strategy: ${r.strategy}`, false),
    stat("Skew / kurt", `${fmt(rs.skew, 2)} / ${fmt(rs.excess_kurtosis, 1)}`, "fat tails visible", null),
  ].map((h) => `<div class="stat"><div class="k">${h.k}</div><div class="v ${h.cls || ""}">${h.v}</div><div class="s">${h.s}</div></div>`).join("");

  $("r-dd-table").innerHTML = `<thead><tr><th>peak</th><th>trough</th><th>recovery</th><th>depth</th><th>length (d)</th><th>recov (d)</th></tr></thead><tbody>${
    r.drawdown_episodes.map((e) => `<tr><td>${e.peak}</td><td>${e.trough}</td><td class="txt">${e.recovery}</td>
      <td class="down">${pct(e.depth)}</td><td>${e.length_days}</td><td>${e.recovery_days ?? "–"}</td></tr>`).join("")}</tbody>`;

  $("r-stress-table").innerHTML = `<thead><tr><th>episode</th><th>days</th><th>strategy</th><th>ann vol</th><th>worst day</th></tr></thead><tbody>${
    r.stress_test.map((s) => `<tr><td class="txt">${s.episode}</td><td>${s.n_days}</td>
      <td class="${s.strategy_return >= 0 ? "up" : "down"}">${pf(s.strategy_return)}</td>
      <td>${pct(s.ann_vol_realised)}</td><td class="down">${pf(s.worst_day)}</td></tr>`).join("")}</tbody>`;

  renderUniverse(u.coverage);
  $("u-search").addEventListener("input", () => {
    const q = $("u-search").value.trim().toUpperCase();
    renderUniverse(u.coverage.filter((c) => c.ticker.includes(q)));
  });
}

function renderUniverse(rows) {
  $("r-universe").innerHTML = `<thead><tr><th>ticker</th><th>days</th><th>first</th><th>last</th><th>avg $vol (M)</th><th>% avail</th></tr></thead><tbody>${
    rows.map((c) => `<tr><td>${c.ticker}</td><td>${c.n_days.toLocaleString()}</td><td class="txt">${c.first_date}</td>
      <td class="txt">${c.last_date}</td><td>${fmt(c.avg_dollar_volume_musd, 0)}</td><td>${pct(c.pct_available, 0)}</td></tr>`).join("")}</tbody>`;
}

/* ================= universe & pit ================= */
const EXP_LABELS = {
  A_static_universe: "A · static universe (survivorship-biased)",
  B_pit_universe: "B · point-in-time universe",
  C_pit_sector_beta_neutral: "C · + sector &amp; beta neutral signal",
  D_pit_mean_variance: "D · + mean-variance construction (risk model)",
  E_pit_sqrt_impact: "E · + square-root market impact costs",
};

async function renderPIT() {
  const p = await getJSON("/api/universe/pit");
  const s = p.stats;

  $("pit-cards").innerHTML = [
    stat("Tickers today", `${s.members_last}`, `${s.n_tickers_ever} ever in sample`),
    stat("PIT members, Jan 2015", `${s.members_first}`, "additions gate + listing data"),
    stat("Mean members/day", `${fmt(s.members_mean, 0)}`, `range ${s.members_min}–${s.members_max}`),
    stat("Post-2015 additions", `${s.additions_after_start}`, "in today's list alone"),
    stat("Unknown addition dates", `${s.names_with_unknown_addition}`, "treated as always-in"),
  ].map((h) => `<div class="stat"><div class="k">${h.k}</div><div class="v">${h.v}</div><div class="s">${h.s}</div></div>`).join("");

  AF.lineChart($("pit-members-chart"),
    [{ name: "index members (PIT)", values: p.members_curve.n_members, dates: p.members_curve.dates }],
    { yLabel: "members" });

  const yrs = Object.keys(p.additions_by_year).sort();
  AF.barChart($("pit-additions-chart"), yrs, yrs.map((y) => p.additions_by_year[y]), { yLabel: "names added" });

  if (p.experiments) {
    const rows = Object.entries(p.experiments).map(([k, m]) => {
      const label = EXP_LABELS[k] || k;
      const cls = (k === "A_static_universe") ? "down" : (k === "B_pit_universe" ? "up" : "");
      return `<tr><td class="txt">${label}</td><td class="${m.net_cagr >= 0.15 ? "up" : ""}">${pct(m.net_cagr)}</td>
        <td>${fmt(m.sharpe, 2)}</td><td>${fmt(m.max_dd, 1)}</td><td>${m.info_ratio == null ? "–" : fmt(m.info_ratio, 2)}</td>
        <td>${fmt(m.turnover, 2)}</td><td>${pct(m.cost_drag)}</td></tr>`;
    });
    $("pit-experiments").innerHTML =
      `<thead><tr><th>variant</th><th>net CAGR</th><th>Sharpe</th><th>Max DD</th><th>IR vs SPY</th><th>turnover</th><th>cost drag</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  }

  if (p.ml_oos_pit) {
    const m = p.ml_oos_pit;
    $("pit-ml-table").innerHTML = [kvRow("net CAGR", pct(m.net_cagr)), kvRow("Sharpe", fmt(m.sharpe, 2)),
      kvRow("max drawdown", pct(m.max_dd)), kvRow("info ratio (vs SPY)", fmt(m.info_ratio, 2)),
      kvRow("turnover / rebalance", fmt(m.turnover_per_rebalance, 2)),
      kvRow("validation", "purged walk-forward, 6 folds")].join("");
  }

  if (p.feature_psi) {
    const t8 = p.feature_psi.mean_psi_top8;
    const entries = Object.entries(t8);
    const mx = entries.length ? Math.max(...entries.map((e) => e[1])) : 1;
    $("pit-psi").innerHTML = entries.map(([n, v]) => `
      <div class="hbar-row"><span class="hb-n">${n}</span>
      <div class="hb-track"><div class="hb-fill" style="width:${(100 * v / mx).toFixed(1)}%${v >= 0.25 ? ";background:#ef4444" : ""}"></div></div>
      <span class="hb-v">${fmt(v, 3)}</span></div>`).join("");
    const sig = p.feature_psi.significant_at_some_point || {};
    $("pit-psi-note").textContent = Object.keys(sig).length
      ? `${entries.length} features tracked; PSI>0.25 at some point: ${Object.keys(sig).join(", ")}`
      : `${entries.length} features tracked; none breached PSI 0.25 — distributions stable. ${p.feature_psi.thresholds}.`;
  }
}

/* ================= boot ================= */
renderOverview()
  .then(() => initBacktest())
  .catch((e) => toast("failed to load /api/overview: " + e.message, 6000));

window.addEventListener("resize", debounce(() => {
  if (loaded.overview) renderOverview();
}, 400));

function debounce(fn, ms) {
  let h;
  return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); };
}
