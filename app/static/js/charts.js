/* AlphaForge charts — a tiny hand-rolled SVG chart library.
   Zero dependencies, zero CDN: line/area charts with crosshair
   tooltips, bar charts and heatmaps. ~450 lines of vanilla JS. */
"use strict";

const AF = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const PALETTE = ["#14b8a6", "#8b9dff", "#f59e0b", "#ef4444", "#a78bfa", "#22c55e"];

  const el = (tag, attrs = {}, parent = null) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  };
  const fmt = (v, d = 2) =>
    v == null || Number.isNaN(v) ? "–" :
    Math.abs(v) >= 1000 ? v.toFixed(0) :
    Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(d);
  const pct = (v, d = 1) => v == null || Number.isNaN(v) ? "–" : (100 * v).toFixed(d) + "%";
  const sf = (v) => Math.sign(v) > 0 ? "+" : "";  // sign prefix
  const nice = (span) => {  // nice step for axis ticks
    const raw = span / 5, mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
    return 10 * mag;
  };

  /* ---------- shared frame: axes, grid, resize ---------- */
  function frame(mount, opts = {}) {
    mount.innerHTML = "";
    const W = mount.clientWidth || 720;
    const H = opts.height || (mount.closest(".chart-short") ? 260 : 320);
    const m = { t: 14, r: 62, b: 26, l: 8 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, mount);
    const g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
    const iw = W - m.l - m.r, ih = H - m.t - m.b;

    const label = (x, y, text, anchor, cls, parent) => {
      const t = el("text", {
        x, y, "text-anchor": anchor || "end",
        "dominant-baseline": "middle",
        class: cls || "ax",
      }, parent || g);
      t.textContent = text;
      return t;
    };
    return { svg, g, iw, ih, m, W, H, label };
  }

  const styleText = `.ax{fill:#5c6878;font:10.5px ui-monospace,Consolas,monospace}
    .grid{stroke:#1b2230;stroke-width:1}
    .cross{stroke:#3a465e;stroke-width:1;stroke-dasharray:3 3}`;

  let injected = false;
  function ensureStyle() {
    if (injected) return;
    injected = true;
    const s = document.createElement("style");
    s.textContent = styleText;
    document.head.appendChild(s);
  }

  /* ---------- multi-series line / area chart ---------- */
  function lineChart(mount, series, opts = {}) {
    ensureStyle();
    const F = frame(mount, opts);
    const { g, iw, ih, m } = F;

    const n = series[0]?.values.length || 0;
    if (!n) { empty(mount); return; }
    const xDates = opts.dates || [];
    const xf = (i) => (n === 1 ? iw / 2 : (i / (n - 1)) * iw);

    // y domain across all series (or fixed 0.. for pct-change baseline)
    let lo = Infinity, hi = -Infinity;
    for (const s of series)
      for (const v of s.values) if (v != null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!Number.isFinite(lo)) { empty(mount); return; }
    if (opts.zeroBase) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    if (opts.pad !== false) { const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad; }
    const yf = (v) => ih - ((v - lo) / (hi - lo)) * ih;

    // horizontal grid + y labels (right side, quant style)
    const step = nice(hi - lo);
    const y0 = Math.ceil(lo / step) * step;
    for (let v = y0; v <= hi + 1e-9; v += step) {
      const y = yf(v);
      el("line", { x1: 0, x2: iw, y1: y, y2: y, class: "grid" }, g);
      F.label(iw + 8, y, opts.yFmt ? opts.yFmt(v) : fmt(v), "start", "ax");
    }

    // x labels: ~6 evenly spaced dates
    const nx = Math.min(6, n);
    for (let k = 0; k < nx; k++) {
      const i = Math.round((k / (nx - 1)) * (n - 1));
      F.label(xf(i), ih + 14, xDates[i] || "", "middle", "ax");
    }

    // baseline (y=0) for return-style series
    if (lo < 0 && hi > 0) el("line", { x1: 0, x2: iw, y1: yf(0), y2: yf(0), stroke: "#39455e" }, g);

    // series paths
    const drawn = series.map((s, si) => {
      const color = s.color || PALETTE[si % PALETTE.length];
      let d = "", pen = false;
      s.values.forEach((v, i) => {
        if (v == null || !Number.isFinite(v)) { pen = false; return; }
        d += (pen ? "L" : "M") + xf(i).toFixed(2) + " " + yf(v).toFixed(2);
        pen = true;
      });
      if (opts.area) {
        const base = yf(opts.zeroBase ? Math.min(0, lo) : lo);
        const pts = s.values.map((v, i) => (v == null ? null : [xf(i), yf(v)]));
        const first = pts.findIndex((p) => p), last = n - 1 - [...pts].reverse().findIndex((p) => p);
        let ad = d + `L${xf(last)} ${base}L${xf(first)} ${base}Z`;
        el("path", { d: ad, fill: color, opacity: 0.14, stroke: "none" }, g);
      }
      el("path", { d, fill: "none", stroke: color, "stroke-width": 1.7, "stroke-linejoin": "round" }, g);
      return { s, color };
    });

    // crosshair + tooltip
    attachCrosshair(F, { xf, yf, lo, hi, n, xDates, drawn, iw, ih });

    // legend (rendered into optional legend element, else svg corner)
    const legendHost = opts.legend;
    if (legendHost) {
      legendHost.innerHTML = "";
      for (const { s, color } of drawn) {
        const li = document.createElement("span");
        li.className = "li";
        li.innerHTML = `<span class="sw" style="background:${color}"></span>${s.name}`;
        legendHost.appendChild(li);
      }
    }
    return F;
  }

  /* ---------- vertical bars (IC, decay) ---------- */
  function barChart(mount, labels, values, opts = {}) {
    ensureStyle();
    const F = frame(mount, opts);
    const { g, iw, ih } = F;
    if (!values.length) { empty(mount); return; }
    const lo = Math.min(0, ...values), hi = Math.max(0, ...values);
    const pad = (hi - lo) * 0.08 || 0.01; const lod = lo - pad, hid = hi + pad;
    const yf = (v) => ih - ((v - lod) / (hid - lod)) * ih;
    const bw = Math.max(2, (iw / values.length) * 0.62);
    const gap = iw / values.length;

    const step = nice(hid - lod);
    for (let v = Math.ceil(lod / step) * step; v <= hid; v += step) {
      const y = yf(v);
      el("line", { x1: 0, x2: iw, y1: y, y2: y, class: "grid" }, g);
      F.label(iw + 8, y, opts.yFmt ? opts.yFmt(v) : fmt(v, 3), "start", "ax");
    }
    if (lo < 0) el("line", { x1: 0, x2: iw, y1: yf(0), y2: yf(0), stroke: "#39455e" }, g);

    const tipHost = document.createElement("div");
    values.forEach((v, i) => {
      const x = gap * i + (gap - bw) / 2;
      const color = opts.color ? opts.color(v) : (v >= 0 ? "#14b8a6" : "#ef4444");
      const y = v >= 0 ? yf(v) : yf(0);
      const h = Math.max(1, Math.abs(yf(v) - yf(0)));
      const rect = el("rect", { x, y, width: bw, height: h, rx: 1.5, fill: color, opacity: 0.88 }, g);
      rect.style.cursor = "pointer";
      rect.addEventListener("mouseenter", () =>
        showTip(`<div class="t-d">${labels[i]}</div><div class="t-r"><span class="n">value</span><span>${fmt(v, 4)}</span></div>`, rect));
      rect.addEventListener("mouseleave", hideTip);
    });
    // x labels: thin out to ≤ 24
    const skip = Math.ceil(labels.length / 24);
    labels.forEach((lb, i) => {
      if (i % skip || !lb) return;
      F.label(gap * i + gap / 2, ih + 14, lb, "middle", "ax");
    });
    return F;
  }

  /* ---------- crosshair/tooltip helper ---------- */
  function attachCrosshair(F, ctx) {
    const { xf, yf, n, xDates, drawn, iw, ih } = ctx;
    const overlay = el("rect", { x: 0, y: 0, width: iw, height: ih, fill: "transparent" }, F.g);
    const vline = el("line", { x1: 0, x2: 0, y1: 0, y2: ih, class: "cross", opacity: 0 }, F.g);
    const dots = drawn.map(() => el("circle", { r: 3, opacity: 0, fill: "#0b0e14", stroke: "#fff", "stroke-width": 1.2 }, F.g));

    overlay.style.cursor = "crosshair";
    overlay.addEventListener("mousemove", (ev) => {
      const rect = F.svg.getBoundingClientRect();
      const px = (ev.clientX - rect.left) / rect.width * (F.iw + F.m.l + F.m.r) - F.m.l;
      const i = Math.max(0, Math.min(n - 1, Math.round((px / (iw || 1)) * (n - 1))));
      const x = xf(i);
      vline.setAttribute("x1", x); vline.setAttribute("x2", x); vline.setAttribute("opacity", 1);
      let rows = "";
      drawn.forEach(({ s }, k) => {
        const v = s.values[i];
        dots[k].setAttribute("cx", x);
        dots[k].setAttribute("cy", v == null ? -10 : yf(v));
        dots[k].setAttribute("opacity", v == null ? 0 : 1);
        dots[k].setAttribute("stroke", s.color || PALETTE[k % PALETTE.length]);
        const txt = s.tipFmt ? s.tipFmt(v) : fmt(v);
        rows += `<div class="t-r"><span class="n">${s.name}</span><span>${v == null ? "–" : txt}</span></div>`;
      });
      showTip(`<div class="t-d">${xDates[i] || ""}</div>${rows}`, overlay, ev);
    });
    overlay.addEventListener("mouseleave", () => {
      vline.setAttribute("opacity", 0);
      dots.forEach((d) => d.setAttribute("opacity", 0));
      hideTip();
    });
  }

  let tipEl = null;
  function showTip(html, anchor, ev) {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tip"; document.body.appendChild(tipEl); }
    tipEl.innerHTML = html;
    tipEl.classList.remove("hidden");
    const r = anchor.getBoundingClientRect ? anchor.getBoundingClientRect() : { right: 0, top: 0 };
    const x = ev ? ev.clientX + 14 : r.right;
    const y = ev ? ev.clientY + 14 : r.top;
    tipEl.style.left = Math.min(x, window.innerWidth - 190) + "px";
    tipEl.style.top = Math.min(y, window.innerHeight - 70) + "px";
  }
  function hideTip() { if (tipEl) tipEl.classList.add("hidden"); }

  function empty(mount) {
    mount.innerHTML = '<div style="color:#5c6878;padding:40px;text-align:center;font:12px var(--sans,monospace)">no data</div>';
  }

  /* ---------- monthly heatmap ---------- */
  function monthlyHeatmap(host, monthly) {
    const { years, months, values } = monthly;
    let html = '<table class="mh"><tr><th></th>';
    for (const mo of months) html += `<th>${MONTHS[mo - 1]}</th>`;
    html += "<th>Yr</th></tr>";
    for (let yi = 0; yi < years.length; yi++) {
      html += `<tr><th>${years[yi]}</th>`;
      let sum = 0, cnt = 0;
      for (let mi = 0; mi < 12; mi++) {
        const v = values[yi][mi];
        if (v != null) { sum += v; cnt++; }
        html += `<td style="background:${heatBg(v)};color:${heatFg(v)}">${v == null ? "" : (100 * v).toFixed(1)}</td>`;
      }
      const yr = cnt ? (Math.pow(1 + sum, 12 / Math.max(cnt, 1)) - 1) : null;
      html += `<td style="background:${heatBg(yr)};color:${heatFg(yr)};font-weight:600">${yr == null ? "" : (100 * yr).toFixed(1)}</td></tr>`;
    }
    html += "</table>";
    host.innerHTML = html;
  }
  const heatBg = (v) => v == null ? "#0e131d" :
    v > 0 ? `rgba(34,197,94,${Math.min(0.75, 0.1 + 2.6 * Math.abs(v))})` :
    `rgba(239,68,68,${Math.min(0.75, 0.1 + 2.6 * Math.abs(v))})`;
  const heatFg = (v) => v == null ? "#5c6878" : (Math.abs(v) > 0.12 ? "#fff" : "#c7d0db");

  return { lineChart, barChart, monthlyHeatmap, fmt, pct, sf, PALETTE, MONTHS };
})();
