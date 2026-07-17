/**
 * Shared result rendering for popup + sidepanel.
 */

export function biasClass(bias) {
  const b = (bias || "neutral").toLowerCase();
  if (b === "bullish") return "bull";
  if (b === "bearish") return "bear";
  return "neutral";
}

export function fmtPrice(p, ref) {
  if (p == null || p === "" || Number.isNaN(Number(p))) return "—";
  const n = Number(p);
  const r = Math.abs(ref != null ? Number(ref) : n);
  if (r >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (r >= 1) return n.toLocaleString(undefined, { maximumFractionDigits: 4 });
  if (r >= 0.01) return n.toFixed(6);
  return n.toFixed(8);
}

export function renderResults(container, result) {
  if (!container) return;
  if (!result) {
    container.innerHTML = `<div class="empty-state">No analysis yet. Capture a chart to begin.</div>`;
    return;
  }
  if (result.ok === false) {
    container.innerHTML = `
      <div class="error-card">
        <div class="error-title">Analysis incomplete</div>
        <div class="error-body">${escapeHtml(result.message || result.error || "Unknown error")}</div>
      </div>`;
    return;
  }

  const bias = result.bias || "neutral";
  const bc = biasClass(bias);
  const conf = result.confidence != null ? Number(result.confidence).toFixed(1) : "—";
  const price = result.meta?.price ?? result.snapshot?.last;
  const plan = result.trade_plan || {};
  const factors = Array.isArray(result.factors) ? result.factors : [];
  const patterns = Array.isArray(result.patterns) ? result.patterns.slice(0, 6) : [];
  const levels = Array.isArray(result.key_levels) ? result.key_levels.slice(0, 6) : [];
  const news = result.news;
  const scenarios = result.scenarios;
  const tags = Array.isArray(result.strategy_tags) ? result.strategy_tags : [];
  const vision = result.vision || result._meta?.clientOcr;

  container.innerHTML = `
    <section class="hero ${bc}">
      <div class="hero-top">
        <div>
          <div class="symbol">${escapeHtml(result.symbol || "—")}</div>
          <div class="sub">${escapeHtml(result.exchange || "")} · ${escapeHtml(result.primary_tf || "")}</div>
        </div>
        <div class="bias-pill ${bc}">${escapeHtml(String(bias).toUpperCase())}</div>
      </div>
      <div class="hero-metrics">
        <div class="metric">
          <span class="label">Confidence</span>
          <span class="value">${conf}%</span>
          <div class="bar"><i style="width:${Math.min(100, Number(conf) || 0)}%"></i></div>
        </div>
        <div class="metric">
          <span class="label">Confluence</span>
          <span class="value">${result.confluence_total != null ? Number(result.confluence_total).toFixed(3) : "—"}</span>
        </div>
        <div class="metric">
          <span class="label">Price</span>
          <span class="value">${fmtPrice(price)}</span>
        </div>
      </div>
      <div class="setup-name">${escapeHtml(result.setup_name || "—")}</div>
      <div class="tags">${tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>
    </section>

    <section class="card">
      <h3>Primary Setup</h3>
      ${renderPlan(plan, price)}
    </section>

    <section class="card">
      <h3>Confluence Breakdown</h3>
      ${renderFactors(factors)}
    </section>

    <section class="card">
      <h3>Key Levels</h3>
      ${renderLevels(levels, price)}
    </section>

    <section class="card">
      <h3>Patterns</h3>
      ${
        patterns.length
          ? `<ul class="list">${patterns
              .map(
                (p) =>
                  `<li><strong>${escapeHtml(p.name)}</strong>
                  <span class="chip ${biasClass(p.bias)}">${escapeHtml(p.bias || "")}</span>
                  <span class="muted">${Number(p.confidence || 0).toFixed(0)}%</span>
                  <div class="note">${escapeHtml(p.note || "")}</div></li>`
              )
              .join("")}</ul>`
          : `<div class="muted">No high-confidence patterns</div>`
      }
    </section>

    <section class="card">
      <h3>Market Structure</h3>
      ${renderStructure(result.structure)}
    </section>

    <section class="card">
      <h3>News Impact</h3>
      ${renderNews(news)}
    </section>

    <section class="card">
      <h3>Risk Advice</h3>
      ${renderRisk(plan, result)}
    </section>

    <section class="card">
      <h3>Scenarios</h3>
      ${renderScenarios(scenarios)}
    </section>

    <section class="card">
      <h3>Trader Commentary</h3>
      <p class="commentary">${escapeHtml(result.trader_commentary || "—")}</p>
    </section>

    ${
      vision
        ? `<section class="card">
            <h3>Vision / OCR</h3>
            <pre class="code">${escapeHtml(JSON.stringify(vision, null, 2).slice(0, 1200))}</pre>
          </section>`
        : ""
    }

    ${
      Array.isArray(result.warnings) && result.warnings.length
        ? `<section class="card warn">
            <h3>Warnings</h3>
            <ul class="list">${result.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>
          </section>`
        : ""
    }

    <section class="disclaimer">
      ${escapeHtml(result.disclaimer || "NOT FINANCIAL ADVICE. For research/education only.")}
    </section>
  `;
}

function renderPlan(plan, price) {
  if (!plan || !plan.direction) {
    return `<div class="muted">No trade plan</div>`;
  }
  const tps = Array.isArray(plan.take_profits) ? plan.take_profits : [];
  const rrs = Array.isArray(plan.risk_reward) ? plan.risk_reward : [];
  return `
    <div class="grid-2">
      <div><span class="label">Direction</span><div class="value ${biasClass(plan.direction === "long" ? "bullish" : plan.direction === "short" ? "bearish" : "neutral")}">${escapeHtml(String(plan.direction).toUpperCase())}</div></div>
      <div><span class="label">Quality</span><div class="value">${escapeHtml(plan.quality || "—")}</div></div>
      <div><span class="label">Entry</span><div class="value">${fmtPrice(plan.entry_low, price)} – ${fmtPrice(plan.entry_high, price)}</div></div>
      <div><span class="label">Stop Loss</span><div class="value bear">${fmtPrice(plan.stop_loss, price)}</div></div>
    </div>
    <div class="tp-row">
      ${tps
        .map(
          (tp, i) =>
            `<div class="tp"><span class="label">TP${i + 1}</span><div class="value bull">${fmtPrice(tp, price)}</div><div class="muted">R:R ${rrs[i] != null ? Number(rrs[i]).toFixed(2) : "—"}</div></div>`
        )
        .join("")}
    </div>
    <div class="muted inv">${escapeHtml(plan.invalidation || "")}</div>
  `;
}

function renderFactors(factors) {
  if (!factors.length) return `<div class="muted">No factors</div>`;
  return `
    <div class="factors">
      ${factors
        .map((f) => {
          const s = Number(f.score || 0);
          const pct = Math.round((s + 1) * 50);
          const cls = s > 0.1 ? "bull" : s < -0.1 ? "bear" : "neutral";
          return `<div class="factor">
            <div class="factor-head">
              <span>${escapeHtml(f.name)}</span>
              <span class="${cls}">${s >= 0 ? "+" : ""}${s.toFixed(2)}</span>
            </div>
            <div class="factor-bar"><i class="${cls}" style="width:${pct}%"></i></div>
            <div class="note">${escapeHtml((f.detail || "").slice(0, 100))}</div>
          </div>`;
        })
        .join("")}
    </div>`;
}

function renderLevels(levels, price) {
  if (!levels.length) return `<div class="muted">No levels</div>`;
  return `<table class="table">
    <thead><tr><th>Kind</th><th>Side</th><th>Mid</th><th>Dist</th></tr></thead>
    <tbody>
      ${levels
        .map(
          (l) => `<tr>
            <td>${escapeHtml(l.kind || "")}</td>
            <td class="${biasClass(l.side)}">${escapeHtml(l.side || "")}</td>
            <td>${fmtPrice(l.mid, price)}</td>
            <td>${l.distance_pct != null ? Number(l.distance_pct).toFixed(2) + "%" : "—"}</td>
          </tr>`
        )
        .join("")}
    </tbody>
  </table>`;
}

function renderStructure(s) {
  if (!s) return `<div class="muted">No structure data</div>`;
  return `
    <div class="grid-2">
      <div><span class="label">Trend</span><div class="value">${escapeHtml(s.trend || "—")}</div></div>
      <div><span class="label">Wyckoff</span><div class="value">${escapeHtml(s.wyckoff_phase || "—")}</div></div>
      <div><span class="label">BOS</span><div class="value">${escapeHtml(s.last_bos || "—")}</div></div>
      <div><span class="label">CHoCH</span><div class="value">${escapeHtml(s.last_choch || "—")}</div></div>
    </div>
    <p class="note">${escapeHtml(s.summary || "")}</p>
    <p class="note">${escapeHtml(s.wyckoff_notes || "")}</p>
    <p class="note">${escapeHtml(s.elliott_notes || "")}</p>
  `;
}

function renderNews(news) {
  if (!news) return `<div class="muted">No news data</div>`;
  const items = Array.isArray(news.items) ? news.items.slice(0, 5) : [];
  return `
    <div class="news-head">
      <span class="chip ${biasClass(news.bias)}">${escapeHtml(news.bias || "neutral")}</span>
      <span class="muted">score ${news.aggregate_sentiment != null ? Number(news.aggregate_sentiment).toFixed(2) : "—"}</span>
    </div>
    <p class="note">${escapeHtml(news.summary || "")}</p>
    <ul class="list">
      ${items
        .map(
          (i) =>
            `<li><span class="muted">[${Number(i.sentiment_score || 0).toFixed(2)}]</span> ${escapeHtml(i.title || "")} <span class="muted">(${escapeHtml(i.source || "")})</span></li>`
        )
        .join("")}
    </ul>
  `;
}

function renderRisk(plan, result) {
  const notes = Array.isArray(plan?.notes) ? plan.notes : [];
  const size = plan?.position_size_units;
  const notional = plan?.position_size_notional;
  const riskAmt = plan?.risk_amount;
  const lev = plan?.leverage_suggested;
  return `
    <div class="grid-2">
      <div><span class="label">Position size</span><div class="value">${size != null ? Number(size).toPrecision(6) : "—"} units</div></div>
      <div><span class="label">Notional</span><div class="value">${notional != null ? "$" + Number(notional).toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}</div></div>
      <div><span class="label">Risk $</span><div class="value">${riskAmt != null ? "$" + Number(riskAmt).toFixed(2) : "—"} (${plan?.risk_pct != null ? Number(plan.risk_pct).toFixed(2) + "%" : "—"})</div></div>
      <div><span class="label">Leverage cap</span><div class="value">${lev != null ? Number(lev).toFixed(1) + "x" : "—"}</div></div>
    </div>
    <ul class="list">${notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
    ${
      plan?.quality === "poor"
        ? `<div class="warn-inline">Plan quality is poor — consider standing aside or half-size.</div>`
        : ""
    }
  `;
}

function renderScenarios(sc) {
  if (!sc) return `<div class="muted">No scenarios</div>`;
  const rows = [sc.bullish, sc.base, sc.bearish].filter(Boolean);
  return `<div class="scenarios">
    ${rows
      .map(
        (r) => `<div class="scenario">
        <div class="scenario-head">
          <strong>${escapeHtml(r.name || "")}</strong>
          <span class="chip">${r.probability != null ? Number(r.probability).toFixed(0) + "%" : "—"}</span>
        </div>
        <div class="note"><b>Trigger:</b> ${escapeHtml(r.trigger || "")}</div>
        <div class="note"><b>Target:</b> ${escapeHtml(String(r.target || ""))}</div>
        <div class="note"><b>Invalidation:</b> ${escapeHtml(String(r.invalidation || ""))}</div>
      </div>`
      )
      .join("")}
  </div>`;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
