/* PSX Sentient web app: a client for the FastAPI service (served behind nginx at /api). */
"use strict";

const API = "/api";
const HOST = window.location.hostname;
const LINKS = {
  grafana: `http://${HOST}:3000/d/psx-sentient`,
  mlflow: `http://${HOST}:5000`,
  prometheus: `http://${HOST}:9090/alerts`,
  docs: `http://${HOST}:8000/docs`,
  streamlit: `http://${HOST}:8501`,
};

const COMPANIES = {
  AKBL: ["Askari Bank", "Banks"], ATRL: ["Attock Refinery", "Refinery"], BAFL: ["Bank Alfalah", "Banks"],
  BAHL: ["Bank AL Habib", "Banks"], DGKC: ["D.G. Khan Cement", "Cement"], FCCL: ["Fauji Cement", "Cement"],
  FFC: ["Fauji Fertilizer", "Fertilizer"], HBL: ["Habib Bank", "Banks"], HUBC: ["Hub Power", "Power"],
  INDU: ["Indus Motor", "Automobile"], INIL: ["International Industries", "Engineering"],
  KAPCO: ["Kot Addu Power", "Power"], LUCK: ["Lucky Cement", "Cement"], MARI: ["Mari Energies", "Oil & Gas E&P"],
  MCB: ["MCB Bank", "Banks"], MLCF: ["Maple Leaf Cement", "Cement"], MTL: ["Millat Tractors", "Automobile"],
  NBP: ["National Bank of Pakistan", "Banks"], NML: ["Nishat Mills", "Textile"], NRL: ["National Refinery", "Refinery"],
  OGDC: ["Oil & Gas Development Co.", "Oil & Gas E&P"], PIOC: ["Pioneer Cement", "Cement"],
  PKGS: ["Packages Ltd", "Paper & Packaging"], POL: ["Pakistan Oilfields", "Oil & Gas E&P"],
  PPL: ["Pakistan Petroleum", "Oil & Gas E&P"], PSO: ["Pakistan State Oil", "Oil Marketing"], PTC: ["PTCL", "Telecom"],
  SNGP: ["Sui Northern Gas", "Gas Utility"], SSGC: ["Sui Southern Gas", "Gas Utility"], UBL: ["United Bank", "Banks"],
};

// Plain-English names and formats for the 27 model inputs
const FEATURES = {
  ret_1d: ["1-day return", "pct", "Momentum"], ret_2d: ["2-day return", "pct", "Momentum"],
  ret_3d: ["3-day return", "pct", "Momentum"], ret_5d: ["1-week return", "pct", "Momentum"],
  ret_10d: ["2-week return", "pct", "Momentum"], ret_20d: ["1-month return", "pct", "Momentum"],
  rsi_14: ["RSI (14-day)", "num", "Momentum"], macd_hist: ["MACD momentum", "num", "Momentum"],
  close_to_sma20: ["Price vs 20-day average", "ratio", "Trend"], close_to_sma50: ["Price vs 50-day average", "ratio", "Trend"],
  vol_20d: ["20-day volatility", "pct", "Volatility & volume"], atr_14_pct: ["Average daily range", "pct", "Volatility & volume"],
  bb_width_20: ["Bollinger band width", "pct", "Volatility & volume"], vol_shock_20: ["Volume vs 20-day average", "x", "Volatility & volume"],
  brent_ret_1d: ["Brent oil, 1-day change", "pct", "Macro"], brent_ret_5d: ["Brent oil, 1-week change", "pct", "Macro"],
  usdpkr_ret_1d: ["USD/PKR, 1-day change", "pct", "Macro"], usdpkr_ret_5d: ["USD/PKR, 1-week change", "pct", "Macro"],
  dxy_ret_1d: ["US dollar index, 1-day change", "pct", "Macro"], dxy_ret_5d: ["US dollar index, 1-week change", "pct", "Macro"],
  month_sin: ["Time of year (sin)", "num", "Calendar"], month_cos: ["Time of year (cos)", "num", "Calendar"],
  dow_sin: ["Day of week (sin)", "num", "Calendar"], dow_cos: ["Day of week (cos)", "num", "Calendar"],
  is_earnings_month: ["Earnings season", "flag", "Calendar"], is_winter_gas: ["Winter gas season", "flag", "Calendar"],
  is_monsoon: ["Monsoon season", "flag", "Calendar"],
};

const state = { latest: null, health: null, filter: "all", query: "", charts: {} };

/* ---------------- helpers ---------------- */
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const fmtNum = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const signed = (x, d = 2) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${Number(x).toFixed(d)}`);
const fmtDate = (s) => (s ? new Date(`${s}T00:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "—");
const company = (sym) => COMPANIES[sym] || [sym, ""];
const cls = (sig) => ({ UP: "up", DOWN: "down" }[sig] || "neutral");
const badge = (sig, extra = "") => `<span class="badge ${cls(sig)} ${extra}">${sig === "UP" ? "▲" : sig === "DOWN" ? "▼" : "●"} ${esc(sig)}</span>`;
const sentChip = (s) => (s == null ? `<span class="chip zero">no news</span>`
  : `<span class="chip ${s > 0.05 ? "pos" : s < -0.05 ? "neg" : "zero"}">${signed(s)}</span>`);

function fmtFeature(name, v) {
  const kind = (FEATURES[name] || [])[1];
  if (v == null) return "—";
  if (kind === "pct") return `${(v * 100).toFixed(2)}%`;
  if (kind === "ratio") return `${((v - 1) * 100).toFixed(1)}% ${v >= 1 ? "above" : "below"}`;
  if (kind === "x") return `${v.toFixed(2)}×`;
  if (kind === "flag") return v >= 0.5 ? "yes" : "no";
  return Number(v).toFixed(2);
}

async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function destroyCharts() { Object.values(state.charts).forEach((c) => c.destroy()); state.charts = {}; }
function chart(key, canvas, cfg) { if (state.charts[key]) state.charts[key].destroy(); state.charts[key] = new Chart(canvas, cfg); }
function setActiveNav(route) {
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
}
function errorBox(err) { return `<div class="error-box"><b>Couldn't load data.</b> ${esc(err.message || err)}</div>`; }

/* ---------------- header ---------------- */
async function loadHealth() {
  const pill = $("#model-pill");
  try {
    state.health = await api("/health");
    const m = state.health.model;
    pill.innerHTML = `<span class="dot"></span>Model v${esc(m.version)} · ${esc(m.model_type)} · ${esc(m.alias)}`;
  } catch (e) {
    pill.innerHTML = `<span class="dot off"></span>API offline`;
  }
}

/* ---------------- Home: pre-market brief ---------------- */
async function renderHome() {
  setActiveNav("home");
  const app = $("#app");
  app.innerHTML = `
    <div class="page-head"><div><h1>Pre-market brief</h1><p>Loading today's signals…</p></div></div>
    <div class="grid grid-4">${'<div class="card"><div class="skeleton" style="height:70px"></div></div>'.repeat(4)}</div>`;
  try {
    state.latest = await api("/signals/latest");
  } catch (e) { app.innerHTML = errorBox(e); return; }
  const d = state.latest;
  const pm = d.premarket;
  const sigs = d.signals;
  const count = (s) => sigs.filter((x) => x.final_signal === s).length;
  const modelCount = (s) => sigs.filter((x) => x.model_signal === s).length;
  const withNews = sigs.filter((x) => x.sentiment_S_t != null);
  const avgNews = withNews.length ? withNews.reduce((a, x) => a + x.sentiment_S_t, 0) / withNews.length : null;
  const whales = sigs.filter((x) => x.whale_score != null && Math.abs(x.whale_score) > 0);

  app.innerHTML = `
    <div class="page-head">
      <div>
        <h1>Pre-market brief</h1>
        <p>Signals for the next PSX session after <b>${fmtDate(d.date)}</b>
        ${pm ? ` · news & PSX files collected ${esc(pm.generated_at_pkt)} PKT` : ""}</p>
      </div>
      <span class="pill"><span class="dot"></span>${pm ? `${Object.values(pm.sources).filter((v) => v === "ok").length}/5 PSX files · ${esc(pm.news)}` : "Pre-market job not run yet"}</span>
    </div>

    <div class="grid grid-4">
      <div class="card kpi">
        <div class="kpi-icon">🎯</div><div class="kpi-label">Final signals</div>
        <div class="split"><b class="chg pos">${count("UP")}</b><span class="muted">up</span><b class="chg neg">${count("DOWN")}</b><span class="muted">down</span><b>${count("NEUTRAL")}</b><span class="muted">neutral</span></div>
        <div class="kpi-sub">Model's raw calls: ${modelCount("UP")} up · ${modelCount("DOWN")} down</div>
      </div>
      <div class="card kpi">
        <div class="kpi-icon">📰</div><div class="kpi-label">Stocks with fresh news</div>
        <div class="kpi-value">${withNews.length}<span class="muted" style="font-size:16px"> / 30</span></div>
        <div class="kpi-sub">Average sentiment ${avgNews == null ? "—" : signed(avgNews)}</div>
      </div>
      <div class="card kpi">
        <div class="kpi-icon">🐋</div><div class="kpi-label">Off-market whale blocks</div>
        <div class="kpi-value">${pm ? pm.counts.whale_blocks : "—"}</div>
        <div class="kpi-sub">${whales.length ? whales.map((w) => esc(w.symbol)).join(", ") : "No large block deals"}</div>
      </div>
      <div class="card kpi">
        <div class="kpi-icon">📢</div><div class="kpi-label">Company announcements</div>
        <div class="kpi-value">${pm ? pm.counts.announcements : "—"}</div>
        <div class="kpi-sub">Results, dividends, book closures</div>
      </div>
    </div>

    <div class="card section">
      <div class="card-head">
        <div><h2>All 30 stocks</h2><p>Click a stock for the full explanation, news and what-if analysis.</p></div>
      </div>
      <div class="toolbar">
        <div class="seg" id="filters">
          ${[["all", "All"], ["UP", "Up"], ["DOWN", "Down"], ["news", "With news"], ["whale", "Whale activity"]]
            .map(([k, l]) => `<button data-f="${k}" class="${state.filter === k ? "active" : ""}">${l}</button>`).join("")}
        </div>
        <div class="search"><span>⌕</span><input id="q" placeholder="Search symbol, company or sector" value="${esc(state.query)}"></div>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Stock</th><th>Close</th><th>Signal</th><th>Model confidence</th><th>P(down) · P(up)</th><th>News</th><th>Whale</th><th>Latest headline</th></tr></thead>
        <tbody id="rows"></tbody>
      </table></div>
    </div>`;

  $("#filters").addEventListener("click", (e) => {
    if (!e.target.dataset.f) return;
    state.filter = e.target.dataset.f;
    document.querySelectorAll("#filters button").forEach((b) => b.classList.toggle("active", b.dataset.f === state.filter));
    drawRows();
  });
  $("#q").addEventListener("input", (e) => { state.query = e.target.value; drawRows(); });
  drawRows();
}

function drawRows() {
  const q = state.query.trim().toLowerCase();
  const rows = state.latest.signals.filter((s) => {
    const [name, sector] = company(s.symbol);
    if (q && !`${s.symbol} ${name} ${sector}`.toLowerCase().includes(q)) return false;
    if (state.filter === "UP" || state.filter === "DOWN") return s.final_signal === state.filter || s.model_signal === state.filter;
    if (state.filter === "news") return s.sentiment_S_t != null;
    if (state.filter === "whale") return s.whale_score != null && Math.abs(s.whale_score) > 0;
    return true;
  });
  $("#rows").innerHTML = rows.map((s) => {
    const [name, sector] = company(s.symbol);
    const p = s.probabilities;
    const whale = s.whale_score == null || s.whale_score === 0 ? `<span class="faint">—</span>`
      : `<span class="chip ${s.whale_score > 0 ? "pos" : "neg"}">🐋 ${signed(s.whale_score)}</span>`;
    const finalNote = s.final_signal !== s.model_signal ? `<div class="co">model: ${esc(s.model_signal)}${s.veto_triggered ? " · vetoed" : ""}</div>` : "";
    return `<tr class="clickable" data-sym="${esc(s.symbol)}">
      <td><div class="sym">${esc(s.symbol)}</div><div class="co">${esc(name)} · ${esc(sector)}</div></td>
      <td class="num">${fmtNum(s.close)}</td>
      <td>${badge(s.final_signal)}${finalNote}</td>
      <td><div class="conf"><div class="conf-track"><div class="conf-fill" style="width:${(s.confidence * 100).toFixed(0)}%"></div></div><span class="num">${pct(s.confidence, 0)}</span></div></td>
      <td class="num"><span class="chg neg">${pct(p.DOWN, 0)}</span> · <span class="chg pos">${pct(p.UP, 0)}</span></td>
      <td>${s.sentiment_S_t == null && s.latest_headline ? `<span class="chip zero">older news</span>` : sentChip(s.sentiment_S_t)}</td>
      <td>${whale}</td>
      <td><div class="headline" title="${esc(s.latest_headline?.title || "")}">${esc(s.latest_headline?.title || "")}</div></td>
    </tr>`;
  }).join("") || `<tr><td colspan="8" class="empty">No stocks match this filter.</td></tr>`;
  document.querySelectorAll("#rows tr.clickable").forEach((tr) => tr.addEventListener("click", () => { location.hash = `#/stock/${tr.dataset.sym}`; }));
}

/* ---------------- Stock page ---------------- */
async function renderStock(sym) {
  setActiveNav("");
  const app = $("#app");
  const [name, sector] = company(sym);
  app.innerHTML = `<a class="back" href="#/">← Back to brief</a>
    <div class="grid grid-main"><div class="card"><div class="skeleton" style="height:340px"></div></div><div class="card"><div class="skeleton" style="height:340px"></div></div></div>`;
  let pred, hist, feats;
  try {
    [pred, hist, feats] = await Promise.all([
      api("/predict", { method: "POST", body: JSON.stringify({ symbol: sym }) }),
      api(`/history/${sym}?days=260`),
      api(`/features/${sym}`),
    ]);
  } catch (e) { app.innerHTML = `<a class="back" href="#/">← Back to brief</a>${errorBox(e)}`; return; }

  const closes = hist.close;
  const last = closes[closes.length - 1];
  const prev = closes[closes.length - 2];
  const chg = prev ? (last / prev - 1) : null;
  const ctx = pred.premarket_context || null;

  app.innerHTML = `
    <a class="back" href="#/">← Back to brief</a>
    <div class="page-head">
      <div class="stock-head">
        <div class="avatar">${esc(sym.slice(0, 4))}</div>
        <div><h1>${esc(sym)} <span class="muted" style="font-weight:500;font-size:18px">${esc(name)}</span></h1>
          <p>${esc(sector)} · data through ${fmtDate(pred.date)}</p></div>
      </div>
      <div style="text-align:right"><div class="price num">PKR ${fmtNum(last)}</div>
        <div class="chg ${chg >= 0 ? "pos" : "neg"} num">${chg == null ? "" : `${chg >= 0 ? "▲" : "▼"} ${pct(Math.abs(chg), 2)} on the day`}</div></div>
    </div>

    <div class="grid grid-main">
      <div class="card">
        <div class="card-head"><div><h2>Price</h2><p>Daily closing price (split-adjusted)</p></div>
          <div class="ranges" id="ranges">${[["1M", 22], ["3M", 66], ["6M", 130], ["1Y", 260]].map(([l, n]) => `<button data-n="${n}" class="${l === "6M" ? "active" : ""}">${l}</button>`).join("")}</div></div>
        <div class="chart-box"><canvas id="price-chart"></canvas></div>
      </div>
      <div class="card" id="pred-card"></div>
    </div>

    <div class="grid grid-2 section">
      <div class="card" id="why-card"></div>
      <div class="card" id="context-card"></div>
    </div>

    <div class="card section" id="whatif-card"></div>`;

  // Price chart
  const drawPrice = (n) => {
    const labels = hist.dates.slice(-n), data = closes.slice(-n);
    const up = data[data.length - 1] >= data[0];
    const color = up ? "#059669" : "#dc2626";
    const g = $("#price-chart").getContext("2d").createLinearGradient(0, 0, 0, 300);
    g.addColorStop(0, up ? "rgba(5,150,105,.18)" : "rgba(220,38,38,.18)"); g.addColorStop(1, "rgba(255,255,255,0)");
    chart("price", $("#price-chart"), {
      type: "line",
      data: { labels, datasets: [{ data, borderColor: color, backgroundColor: g, fill: true, borderWidth: 2, pointRadius: 0, tension: 0.25 }] },
      options: {
        responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` PKR ${fmtNum(c.parsed.y)}` } } },
        scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 6, color: "#94a3b8" } },
          y: { grid: { color: "#eef1f6" }, ticks: { color: "#94a3b8" } } },
      },
    });
  };
  $("#ranges").addEventListener("click", (e) => {
    if (!e.target.dataset.n) return;
    document.querySelectorAll("#ranges button").forEach((b) => b.classList.toggle("active", b === e.target));
    drawPrice(Number(e.target.dataset.n));
  });

  drawPrediction(pred);
  drawShap(pred);
  drawContext(ctx);
  drawWhatIf(sym, pred, feats.features);
  drawPrice(130);
}

function drawPrediction(pred) {
  const p = pred.probabilities;
  const vetoNote = pred.veto_triggered
    ? `<div class="notice warn">⚠️ <div><b>Conflict veto.</b> The news sentiment strongly contradicts the technical model, so the signal is held at NEUTRAL.</div></div>` : "";
  const newsNote = pred.sentiment_S_t != null
    ? `<div class="notice info">📰 <div>News sentiment ${signed(pred.sentiment_S_t)} (${esc(pred.sentiment_source)}) is blended into the final signal.</div></div>` : "";
  $("#pred-card").innerHTML = `
    <div class="card-head"><div><h2>Next-session signal</h2><p>Model v${esc(pred.model.version)} · ${esc(pred.model.model_type)}</p></div></div>
    <div class="signal-hero">${badge(pred.final_signal, "lg")}
      <div style="text-align:right"><div class="muted" style="font-size:13px">Model's raw call</div><b>${esc(pred.model_signal)} · ${pct(pred.confidence, 0)}</b></div></div>
    <div class="probbar"><div class="d" style="width:${p.DOWN * 100}%"></div><div class="n" style="width:${p.NEUTRAL * 100}%"></div><div class="u" style="width:${p.UP * 100}%"></div></div>
    <div class="prob-legend"><span>Down <b>${pct(p.DOWN, 0)}</b></span><span>Neutral <b>${pct(p.NEUTRAL, 0)}</b></span><span>Up <b>${pct(p.UP, 0)}</b></span></div>
    <div class="stat-row">
      <div class="stat"><div class="l">Technical edge</div><div class="v num">${signed(pred.E_tech)}</div></div>
      <div class="stat"><div class="l">News sentiment</div><div class="v num">${pred.sentiment_S_t == null ? "—" : signed(pred.sentiment_S_t)}</div></div>
      <div class="stat"><div class="l">Blended conviction</div><div class="v num">${signed(pred.C_t)}</div></div>
    </div>
    ${vetoNote}${newsNote}
    <p class="faint" style="font-size:12.5px;margin:14px 0 0">An UP/DOWN call needs blended conviction beyond ±0.20 and a neutral probability under 50%. Otherwise the signal stays NEUTRAL.</p>`;
}

function drawShap(pred) {
  const top = pred.explanation.top_features;
  const cls_ = pred.explanation.explained_class;
  const box = $("#why-card");
  box.innerHTML = `
    <div class="card-head"><div><h2>Why ${esc(cls_)}?</h2><p>SHAP contributions: what pushed the model towards or away from ${esc(cls_)}</p></div></div>
    <div class="chart-box" style="height:260px"><canvas id="shap-chart"></canvas></div>
    <ul class="reasons">${top.slice(0, 4).map((t) => {
      const name = (FEATURES[t.feature] || [t.feature])[0];
      const towards = t.shap >= 0;
      return `<li><span class="arrow ${towards ? "chg pos" : "chg neg"}">${towards ? "↑" : "↓"}</span>
        <span><b>${esc(name)}</b> (${esc(fmtFeature(t.feature, t.value))}) ${towards ? "pushed towards" : "pushed away from"} ${esc(cls_)}</span></li>`;
    }).join("")}</ul>`;
  chart("shap", $("#shap-chart"), {
    type: "bar",
    data: { labels: top.map((t) => (FEATURES[t.feature] || [t.feature])[0]),
      datasets: [{ data: top.map((t) => t.shap), backgroundColor: top.map((t) => (t.shap >= 0 ? "#059669" : "#dc2626")), borderRadius: 6 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` SHAP ${signed(c.parsed.x, 3)}` } } },
      scales: { x: { grid: { color: "#eef1f6" }, ticks: { color: "#94a3b8" } }, y: { grid: { display: false }, ticks: { color: "#334155" } } } },
  });
}

function drawContext(ctx) {
  const box = $("#context-card");
  if (!ctx) {
    box.innerHTML = `<div class="card-head"><div><h2>Pre-market context</h2><p>News, off-market deals, futures and announcements</p></div></div>
      <div class="empty">The pre-market job hasn't produced a snapshot for this stock yet. It runs every weekday at 09:00 PKT.</div>`;
    return;
  }
  const items = [];
  (ctx.sentiment?.latest_headlines || []).forEach((h) => items.push(
    `<div class="feed-item"><span class="feed-icon">📰</span><div style="flex:1">${esc(h.title)}</div>${sentChip(h.score)}</div>`));
  (ctx.announcements || []).forEach((a) => items.push(
    `<div class="feed-item"><span class="feed-icon">📢</span><div style="flex:1">${esc(a)}</div><span class="chip info">PSX notice</span></div>`));
  if (ctx.whale && ctx.whale.ndm_volume) {
    const w = ctx.whale;
    items.push(`<div class="feed-item"><span class="feed-icon">🐋</span><div style="flex:1"><b>Off-market block deal:</b>
      ${Number(w.ndm_volume).toLocaleString()} shares at PKR ${fmtNum(w.ndm_deal_rate)}, ${signed(w.ndm_spread_pct)}% vs the close
      (${fmtNum(w.ndm_vol_pct_of_sma20, 1)}% of normal daily volume)</div><span class="chip ${w.W_ndm > 0 ? "pos" : w.W_ndm < 0 ? "neg" : "zero"}">${signed(w.W_ndm)}</span></div>`);
  }
  if (ctx.futures_open_interest && ctx.futures_open_interest.pct_free_float != null) {
    items.push(`<div class="feed-item"><span class="feed-icon">📈</span><div style="flex:1"><b>Futures open interest:</b>
      ${fmtNum(ctx.futures_open_interest.pct_free_float)}% of free float (${Number(ctx.futures_open_interest.volume || 0).toLocaleString()} shares)</div></div>`);
  }
  if (ctx.short_sell) {
    items.push(`<div class="feed-item"><span class="feed-icon">🔻</span><div style="flex:1"><b>Short selling:</b> ${Number(ctx.short_sell.short_volume).toLocaleString()} shares</div></div>`);
  }
  box.innerHTML = `<div class="card-head"><div><h2>Pre-market context</h2><p>Collected ${fmtDate(ctx.trading_day)} → this morning · context only, not model inputs</p></div>
      ${sentChip(ctx.sentiment?.S_t)}</div>
    <div class="feed">${items.join("") || `<div class="empty">No news, block deals or announcements for this stock in the last 3 days.</div>`}</div>`;
}

function drawWhatIf(sym, base, features) {
  const groups = {};
  Object.keys(FEATURES).forEach((f) => { (groups[FEATURES[f][2]] ||= []).push(f); });
  const box = $("#whatif-card");
  box.innerHTML = `
    <div class="card-head"><div><h2>What-if analysis</h2><p>Change the news sentiment or any model input and re-run the live model.</p></div>
      <button class="btn ghost" id="reset-btn">Reset</button></div>
    <div class="slider-row"><span class="muted" style="min-width:150px">News sentiment</span>
      <input type="range" id="s-range" min="-1" max="1" step="0.05" value="${base.sentiment_S_t ?? 0}">
      <b id="s-val" class="num" style="min-width:48px;text-align:right">${signed(base.sentiment_S_t ?? 0)}</b></div>
    <details style="margin-top:14px"><summary style="cursor:pointer;font-weight:600">Edit model inputs (27 features)</summary>
      <div class="inputs-grid">${Object.entries(groups).map(([g, fs]) => `<div class="group-title">${esc(g)}</div>` + fs.map((f) =>
        `<label>${esc(FEATURES[f][0])}<input data-f="${f}" data-orig="${features[f]}" value="${Number(features[f]).toPrecision(6)}"></label>`).join("")).join("")}</div>
    </details>
    <div style="display:flex;gap:10px;align-items:center;margin-top:16px"><button class="btn" id="run-btn">Run prediction</button>
      <span id="whatif-out" class="muted"></span></div>`;

  const range = $("#s-range");
  range.addEventListener("input", () => { $("#s-val").textContent = signed(Number(range.value)); });
  box.querySelectorAll("input[data-f]").forEach((inp) => inp.addEventListener("input", () => {
    inp.classList.toggle("changed", Number(inp.value) !== Number(inp.dataset.orig));
  }));
  $("#reset-btn").addEventListener("click", () => drawWhatIf(sym, base, features));
  $("#run-btn").addEventListener("click", async () => {
    const feats = {};
    let edited = false;
    box.querySelectorAll("input[data-f]").forEach((inp) => {
      feats[inp.dataset.f] = Number(inp.value);
      if (Number(inp.value) !== Number(inp.dataset.orig)) edited = true;
    });
    const body = { symbol: sym, sentiment: Number(range.value) };
    if (edited) body.features = feats;
    $("#whatif-out").textContent = "Running…";
    try {
      const r = await api("/predict", { method: "POST", body: JSON.stringify(body) });
      r.date = r.date || base.date;
      drawPrediction(r);
      drawShap(r);
      $("#whatif-out").innerHTML = `Result: ${badge(r.final_signal)} (model ${esc(r.model_signal)} ${pct(r.confidence, 0)}) — prediction card and explanation updated above.`;
    } catch (e) { $("#whatif-out").textContent = e.message; }
  });
}

/* ---------------- Monitoring ---------------- */
async function renderMonitoring() {
  setActiveNav("monitoring");
  const app = $("#app");
  app.innerHTML = `<div class="page-head"><div><h1>Model monitoring</h1><p>Loading…</p></div></div>`;
  const [model, versions, drift] = await Promise.all([
    api("/model").catch(() => null), api("/model/versions").catch(() => ({ versions: [] })), api("/drift").catch(() => null)]);
  const served = model?.served || {};
  const lr = model?.last_retraining;

  const tl = (versions.versions || []).map((v) => {
    const when = v.created_ms ? new Date(v.created_ms).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
    const role = v.tags.role || (v.tags.trigger ? "challenger" : "trained by pipeline");
    return `<div class="tl-item"><div class="tl-ver">v${v.version}</div>
      <div class="tl-meta">${esc(v.tags.model_type || "")} · ${esc(role)}${v.tags.trigger ? ` · trigger: ${esc(v.tags.trigger)}` : ""}<br>${esc(when)}
      ${v.tags.compare_macro_f1 ? ` · recent-window F1 ${esc(v.tags.compare_macro_f1)}` : v.tags.test_2026_macro_f1 ? ` · Test 2026 F1 ${esc(v.tags.test_2026_macro_f1)}` : ""}</div>
      <div>${v.aliases.map((a) => `<span class="chip ${a === "champion" ? "pos" : a === "challenger" ? "info" : "zero"}">${esc(a)}</span>`).join(" ")}</div></div>`;
  }).join("");

  let cmp = "";
  if (lr) {
    const rows = [["Macro F1", "macro_f1"], ["Active-signal precision", "active_signal_precision"], ["Active-signal rate", "active_signal_rate"],
      ["Precision (down)", "precision_down"], ["Precision (up)", "precision_up"]];
    cmp = `<table class="compare"><thead><tr><th>Metric</th><th>Champion v${esc(lr.champion_before)}</th><th>Challenger v${esc(lr.challenger_version)}</th></tr></thead><tbody>
      ${rows.map(([l, k]) => { const a = lr.champion_metrics[k], b = lr.challenger_metrics[k];
        const tone = k === "active_signal_rate" ? "" : b > a ? "better" : b < a ? "worse" : "";
        return `<tr><td>${l}</td><td class="num">${pct(a)}</td><td class="num ${tone}">${pct(b)}</td></tr>`; }).join("")}</tbody></table>`;
  }

  app.innerHTML = `
    <div class="page-head"><div><h1>Model monitoring</h1><p>What's serving, how the data is drifting, and how retraining decided.</p></div>
      <div class="links">
        <a class="btn ghost" href="${LINKS.grafana}" target="_blank" rel="noopener">Grafana ↗</a>
        <a class="btn ghost" href="${LINKS.mlflow}" target="_blank" rel="noopener">MLflow ↗</a>
        <a class="btn ghost" href="${LINKS.prometheus}" target="_blank" rel="noopener">Alerts ↗</a>
      </div></div>

    <div class="grid grid-4">
      <div class="card kpi"><div class="kpi-icon">🤖</div><div class="kpi-label">Serving</div>
        <div class="kpi-value">v${esc(served.version || "—")}</div><div class="kpi-sub">${esc(served.model_type || "")} · ${esc(served.alias || "")}</div></div>
      <div class="card kpi"><div class="kpi-icon">🌊</div><div class="kpi-label">Market features drifting</div>
        <div class="kpi-value">${drift ? pct(drift.drift_share, 0) : "—"}</div><div class="kpi-sub">${drift ? `${drift.drifted_market_features.length} of 20 features` : "No report yet"}</div></div>
      <div class="card kpi"><div class="kpi-icon">🔁</div><div class="kpi-label">Retraining</div>
        <div class="kpi-value" style="font-size:22px">${drift ? (drift.retrain_recommended ? `<span class="chg neg">Recommended</span>` : `<span class="chg pos">Not needed</span>`) : "—"}</div>
        <div class="kpi-sub">Prediction-mix PSI ${drift?.prediction_class_psi ?? "—"}${lr && drift?.retrain_recommended ? ` · last run: ${lr.decision === "promoted" ? `promoted v${esc(lr.challenger_version)}` : "kept champion"}` : ""}</div></div>
      <div class="card kpi"><div class="kpi-icon">🏁</div><div class="kpi-label">Last decision</div>
        <div class="kpi-value" style="font-size:22px">${lr ? (lr.decision === "promoted" ? `<span class="chg pos">Promoted v${esc(lr.challenger_version)}</span>` : "Kept champion") : "—"}</div>
        <div class="kpi-sub">${lr ? `compared on ${fmtDate(lr.compare_window[0])} – ${fmtDate(lr.compare_window[1])}` : "No retraining yet"}</div></div>
    </div>

    <div class="grid grid-2 section">
      <div class="card"><div class="card-head"><div><h2>Feature drift (PSI)</h2>
        <p>${drift ? `Window ${fmtDate(drift.current_range[0])} – ${fmtDate(drift.current_range[1])} vs training data. Red = significant (PSI ≥ 0.2)` : ""}</p></div></div>
        <div class="chart-box" style="height:520px"><canvas id="psi-chart"></canvas></div></div>
      <div class="grid" style="align-content:start">
        <div class="card"><div class="card-head"><div><h2>Champion vs challenger</h2><p>${lr ? esc(lr.rule) : "Run the retraining loop to see a comparison."}</p></div></div>${cmp}</div>
        <div class="card"><div class="card-head"><div><h2>Model registry</h2><p>${esc(versions.name || "")}</p></div></div><div class="timeline">${tl || `<div class="empty">No versions found.</div>`}</div></div>
      </div>
    </div>`;

  if (drift) {
    const entries = Object.entries(drift.features).sort((a, b) => b[1].psi - a[1].psi);
    chart("psi", $("#psi-chart"), {
      type: "bar",
      data: { labels: entries.map(([f]) => (FEATURES[f] || [f])[0]),
        datasets: [{ data: entries.map(([, v]) => Math.min(v.psi, 1.5)), borderRadius: 5,
          backgroundColor: entries.map(([, v]) => (v.calendar ? "#cbd5e1" : v.psi >= 0.2 ? "#dc2626" : v.psi >= 0.1 ? "#f59e0b" : "#059669")) }] },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` PSI ${entries[c.dataIndex][1].psi.toFixed(3)}${entries[c.dataIndex][1].calendar ? " (calendar, ignored)" : ""}` } } },
        scales: { x: { grid: { color: "#eef1f6" }, ticks: { color: "#94a3b8" } }, y: { grid: { display: false }, ticks: { color: "#334155", font: { size: 11 } } } } },
    });
  }
}

/* ---------------- About ---------------- */
async function renderAbout() {
  setActiveNav("about");
  const app = $("#app");
  const perf = await api("/performance").catch(() => null);
  let rows = "";
  if (perf) {
    const add = (name, type, r, hl = false) => { if (!r) return;
      rows += `<tr class="${hl ? "highlight-row" : ""}"><td>${esc(name)}</td><td>${esc(type)}</td><td class="num">${pct(r.val_mean?.macro_f1)}</td>
        <td class="num">${pct(r.test_2026?.macro_f1)}</td><td class="num">${pct(r.test_2026?.active_signal_precision)}</td></tr>`; };
    Object.entries(perf.baselines?.baselines || {}).forEach(([n, r]) => add(n.replace(/([a-z])([A-Z])/g, "$1 $2"), "Baseline", r));
    Object.entries(perf.models?.models || {}).forEach(([n, r]) => add(n, "Primary model", r, n === perf.models.best_model));
    if (perf.stacking) add("Stacking meta-learner", "Late fusion", perf.stacking);
  }
  app.innerHTML = `
    <div class="page-head"><div><h1>How PSX Sentient works</h1><p>Next-day direction signals for 30 KSE-100 stocks, built as a production MLOps system.</p></div></div>
    <div class="steps">
      <div class="card"><div class="step-n">1</div><h3>Learn from 10 years of prices</h3><p class="muted">Split-adjusted prices for 30 stocks (2017–2026), technical indicators, oil, rupee and dollar moves, and seasonality train a CatBoost model with leak-free walk-forward validation.</p></div>
      <div class="card"><div class="step-n">2</div><h3>Read the morning</h3><p class="muted">At 09:00 PKT the system pulls the PSX daily files, off-market block deals, futures positioning, company announcements and overnight news, scored by FinBERT.</p></div>
      <div class="card"><div class="step-n">3</div><h3>Signal, explain, adapt</h3><p class="muted">Each stock gets UP / NEUTRAL / DOWN with probabilities and SHAP explanations. Drift is monitored, and a retrained challenger replaces the champion only if it's better.</p></div>
    </div>

    <div class="card section"><div class="card-head"><div><h2>Model performance</h2>
      <p>Test 2026 = Jan–Jun 2026, never used to choose the model. Active precision = share of UP/DOWN calls that were right (≈23% by chance).</p></div></div>
      <div class="table-wrap"><table><thead><tr><th>Model</th><th>Type</th><th>Validation F1 (2023–25)</th><th>Test F1</th><th>Test active precision</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="5" class="empty">Metrics unavailable.</td></tr>`}</tbody></table></div></div>

    <div class="grid grid-2 section">
      <div class="card prose"><h2>What the signal means</h2>
        <ul><li><b>UP / DOWN</b>: the model expects the next close-to-close move to clear a volatility-scaled threshold.</li>
        <li><b>NEUTRAL</b>: no move expected beyond normal noise, or not enough conviction.</li>
        <li><b>Conflict veto</b>: strong opposing news sentiment holds a technical call at NEUTRAL.</li>
        <li><b>Whales, futures, announcements</b>: shown as context. They aren't model inputs yet because there's no training history; the daily archive is building it.</li></ul></div>
      <div class="card prose"><h2>Honest limitations</h2>
        <ul><li>The edge is modest: about 28% of UP/DOWN calls are right versus about 23% by chance. It is untested after trading costs.</li>
        <li>The target is close-to-close, but a 09:00 signal can only be acted on at the 09:30 open.</li>
        <li>News sentiment and the veto run live only; there is no historical news archive to measure their effect yet.</li>
        <li>Dividend days are detected heuristically.</li></ul>
        <p class="faint" style="font-size:13px">Not investment advice.</p></div>
    </div>

    <div class="card section"><div class="card-head"><div><h2>Under the hood</h2><p>Open the engineering tools behind this site</p></div></div>
      <div class="links">
        <a class="btn ghost" href="${LINKS.docs}" target="_blank" rel="noopener">API docs ↗</a>
        <a class="btn ghost" href="${LINKS.grafana}" target="_blank" rel="noopener">Grafana ↗</a>
        <a class="btn ghost" href="${LINKS.mlflow}" target="_blank" rel="noopener">MLflow ↗</a>
        <a class="btn ghost" href="${LINKS.prometheus}" target="_blank" rel="noopener">Prometheus ↗</a>
        <a class="btn ghost" href="${LINKS.streamlit}" target="_blank" rel="noopener">Classic dashboard ↗</a>
        <a class="btn ghost" href="https://github.com/JunaidMu/PSX-Sentient" target="_blank" rel="noopener">GitHub ↗</a>
      </div></div>`;
}

/* ---------------- router ---------------- */
function route() {
  destroyCharts();
  window.scrollTo(0, 0);
  const h = location.hash.replace(/^#/, "") || "/";
  const m = h.match(/^\/stock\/([A-Z]+)/i);
  if (m) return renderStock(m[1].toUpperCase());
  if (h.startsWith("/monitoring")) return renderMonitoring();
  if (h.startsWith("/about")) return renderAbout();
  return renderHome();
}

window.addEventListener("hashchange", route);
loadHealth();
route();
