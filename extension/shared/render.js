/**
 * Shared result rendering for popup + sidepanel.
 * Professional, scannable dark-theme report.
 */

/** Educational illustration only — matches backend request capital. */
const SIM_EXAMPLE_USD = 100;

export function biasClass(bias) {
  const b = (bias || "neutral").toLowerCase();
  if (b === "bullish" || b === "long") return "bull";
  if (b === "bearish" || b === "short") return "bear";
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

export function fmtMoney(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  const v = Number(n);
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  const abs = Math.abs(v);
  if (abs >= 100) return `${sign}$${abs.toFixed(0)}`;
  if (abs >= 10) return `${sign}$${abs.toFixed(1)}`;
  return `${sign}$${abs.toFixed(2)}`;
}

/**
 * Suggested hold window from timeframe + setup tags.
 */
export function suggestHoldTime(tf, setupName, tags = [], direction = "") {
  const t = String(tf || "").toLowerCase().trim();
  const blob = `${setupName || ""} ${(tags || []).join(" ")} ${direction || ""}`.toLowerCase();

  if (blob.includes("scalp") || ["1m", "3m", "5m"].includes(t)) {
    return { label: "Scalp", detail: "Suggested hold: 15–90 minutes" };
  }
  if (["15m", "30m"].includes(t)) {
    return { label: "Intraday", detail: "Suggested hold: 1–8 hours" };
  }
  if (["1h", "2h"].includes(t)) {
    return { label: "Day trade", detail: "Suggested hold: 4–24 hours" };
  }
  if (["4h", "6h", "8h", "12h"].includes(t)) {
    if (blob.includes("swing") || blob.includes("breakout")) {
      return { label: "Swing", detail: "Suggested hold: 2–7 days" };
    }
    return { label: "Swing", detail: "Suggested hold: 1–5 days" };
  }
  if (["1d", "3d", "1w", "1W"].includes(t) || t === "d" || t === "w") {
    return { label: "Position", detail: "Suggested hold: 2–14 days" };
  }
  if (blob.includes("swing")) {
    return { label: "Swing", detail: "Suggested hold: 2–7 days" };
  }
  return { label: "Intraday", detail: "Suggested hold: 4–24 hours" };
}

/**
 * Scale TP profits to the educational $100 example capital.
 */
export function simulationProfits(plan) {
  if (!plan) return [];
  const raw = Array.isArray(plan.potential_profits) ? plan.potential_profits : [];
  const cap = Number(plan.simulated_capital) || SIM_EXAMPLE_USD;
  const scale = cap > 0 ? SIM_EXAMPLE_USD / cap : 1;
  return raw.map((p) => Number(p) * scale);
}

export function renderResults(container, result) {
  if (!container) return;
  if (!result) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">◈</div>
        <div class="empty-title">No analysis yet</div>
        <div class="empty-sub">Capture a chart to run pro confluence analysis.</div>
      </div>`;
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
  const vision = result.vision;
  const hold = suggestHoldTime(
    result.primary_tf,
    result.setup_name,
    tags,
    plan.direction || result.direction
  );
  const reasons = Array.isArray(result.key_reasons) ? result.key_reasons.slice(0, 5) : [];
  const risks = Array.isArray(result.key_risks) ? result.key_risks.slice(0, 5) : [];

  container.innerHTML = `
    <section class="hero ${bc}">
      <div class="hero-top">
        <div>
          <div class="symbol">${escapeHtml(prettySymbol(result.symbol))}</div>
          <div class="sub">${escapeHtml(result.exchange || result.exchange_id || "")} · ${escapeHtml(result.primary_tf || "")}</div>
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
      <div class="hold-badge" title="Educational hold window from timeframe & setup type">
        <span class="hold-label">${escapeHtml(hold.label)}</span>
        <span class="hold-detail">${escapeHtml(hold.detail)}</span>
      </div>
      <div class="tags">${tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>
    </section>

    <section class="card card-accent">
      <h3>Primary Setup</h3>
      ${renderPlan(plan, price)}
    </section>

    <section class="card sim-card">
      <h3>Simulation Example</h3>
      ${renderSimulation(plan, result)}
    </section>

    ${
      reasons.length || risks.length
        ? `<section class="card">
            <h3>Why This Signal</h3>
            ${renderReasonsRisks(reasons, risks)}
          </section>`
        : ""
    }

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
          ? `<ul class="list clean-list">${patterns
              .map(
                (p) =>
                  `<li>
                    <div class="list-row">
                      <strong>${escapeHtml(p.name)}</strong>
                      <span class="chip ${biasClass(p.bias)}">${escapeHtml(p.bias || "")}</span>
                      <span class="muted">${Number(p.confidence || 0).toFixed(0)}%</span>
                    </div>
                    ${p.note ? `<div class="note">${escapeHtml(p.note)}</div>` : ""}
                  </li>`
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
      <h3>Scenarios</h3>
      ${renderScenarios(scenarios)}
    </section>

    <section class="card">
      <h3>Trader Commentary</h3>
      <p class="commentary">${escapeHtml(result.trader_commentary || "—")}</p>
      ${
        plan.leverage_reasoning
          ? `<p class="note lev-note"><b>Leverage logic:</b> ${escapeHtml(plan.leverage_reasoning)}</p>`
          : ""
      }
    </section>

    ${
      vision && (vision.notes || vision.ocr || vision.cv)
        ? `<section class="card collapsible-meta">
            <h3>Vision / OCR</h3>
            ${renderVisionSummary(vision)}
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
      ${escapeHtml(result.disclaimer || "NOT FINANCIAL ADVICE. Educational / research only. Perpetuals are high risk.")}
    </section>
  `;
}

function prettySymbol(sym) {
  if (!sym) return "—";
  const s = String(sym);
  if (s.includes("/")) return s.split(":")[0];
  return s;
}

function renderPlan(plan, price) {
  if (!plan || !plan.direction || plan.direction === "flat") {
    return `<div class="muted">No directional trade plan — stand aside or wait for clearer structure.</div>`;
  }
  const tps = Array.isArray(plan.take_profits) ? plan.take_profits : [];
  const rrs = Array.isArray(plan.risk_reward) ? plan.risk_reward : [];
  const lev = plan.leverage_suggested != null ? Number(plan.leverage_suggested).toFixed(1) : "—";
  return `
    <div class="grid-2">
      <div>
        <span class="label">Direction</span>
        <div class="value ${biasClass(plan.direction)}">${escapeHtml(String(plan.direction).toUpperCase())}</div>
      </div>
      <div>
        <span class="label">Suggested lev.</span>
        <div class="value">${lev}x</div>
      </div>
      <div>
        <span class="label">Entry zone</span>
        <div class="value">${fmtPrice(plan.entry_low, price)} – ${fmtPrice(plan.entry_high, price)}</div>
      </div>
      <div>
        <span class="label">Stop loss</span>
        <div class="value bear">${fmtPrice(plan.stop_loss, price)}</div>
      </div>
      <div>
        <span class="label">Quality</span>
        <div class="value">${escapeHtml((plan.quality || "—").toUpperCase())}</div>
      </div>
      <div>
        <span class="label">R:R (TP1)</span>
        <div class="value">${rrs[0] != null ? Number(rrs[0]).toFixed(2) : "—"}</div>
      </div>
    </div>
    <div class="tp-row">
      ${tps
        .map(
          (tp, i) =>
            `<div class="tp">
              <span class="label">TP${i + 1}</span>
              <div class="value bull">${fmtPrice(tp, price)}</div>
              <div class="muted">R:R ${rrs[i] != null ? Number(rrs[i]).toFixed(2) : "—"}</div>
            </div>`
        )
        .join("")}
    </div>
    ${
      plan.invalidation
        ? `<div class="muted inv"><b>Invalidation:</b> ${escapeHtml(plan.invalidation)}</div>`
        : ""
    }
  `;
}

/**
 * Educational simulation block — not a wallet.
 */
function renderSimulation(plan, result) {
  if (!plan || !plan.direction || plan.direction === "flat") {
    return `<div class="muted">No simulation — flat / neutral bias.</div>`;
  }

  const lev =
    plan.leverage_suggested != null ? Number(plan.leverage_suggested).toFixed(1) : "—";
  const profits = simulationProfits(plan);
  const atrPct = plan.atr_pct ?? result?.meta?.atr_pct;
  const conf = result?.confidence;

  const lines =
    profits.length > 0
      ? profits
          .slice(0, 3)
          .map((p, i) => {
            const cls = p >= 0 ? "bull" : "bear";
            return `<li class="${cls}">At TP${i + 1} → <strong>${fmtMoney(p)}</strong> profit</li>`;
          })
          .join("")
      : `<li class="muted">Profit targets unavailable for this plan.</li>`;

  const metaBits = [];
  if (atrPct != null) metaBits.push(`ATR ${Number(atrPct).toFixed(2)}%`);
  if (conf != null) metaBits.push(`conf ${Number(conf).toFixed(0)}%`);
  metaBits.push("funding-aware lev");

  return `
    <div class="sim-header">
      <span class="sim-badge">For illustration only</span>
    </div>
    <p class="sim-lead">
      If you trade this signal with <strong>$${SIM_EXAMPLE_USD}</strong> at the suggested
      <strong>${escapeHtml(String(lev))}x</strong> leverage:
    </p>
    <ul class="sim-list">
      ${lines}
    </ul>
    <p class="sim-meta muted">
      Leverage is model-suggested from ${escapeHtml(metaBits.join(" · "))} — not exchange max margin.
      Not a real balance or order.
    </p>
  `;
}

function renderReasonsRisks(reasons, risks) {
  return `
    <div class="rr-grid">
      <div>
        <div class="label">Key reasons</div>
        ${
          reasons.length
            ? `<ul class="list tight">${reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
            : `<div class="muted">—</div>`
        }
      </div>
      <div>
        <div class="label">Key risks</div>
        ${
          risks.length
            ? `<ul class="list tight risk-list">${risks.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
            : `<div class="muted">—</div>`
        }
      </div>
    </div>
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
            <div class="note">${escapeHtml((f.detail || "").slice(0, 110))}</div>
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
  `;
}

function renderNews(news) {
  if (!news) return `<div class="muted">No news data</div>`;
  const items = Array.isArray(news.items) ? news.items.slice(0, 4) : [];
  return `
    <div class="news-head">
      <span class="chip ${biasClass(news.bias)}">${escapeHtml(news.bias || "neutral")}</span>
      <span class="muted">score ${news.aggregate_sentiment != null ? Number(news.aggregate_sentiment).toFixed(2) : "—"}</span>
    </div>
    <p class="note">${escapeHtml(news.summary || "")}</p>
    <ul class="list clean-list">
      ${items
        .map(
          (i) =>
            `<li><span class="muted">[${Number(i.sentiment_score || 0).toFixed(2)}]</span> ${escapeHtml(i.title || "")}</li>`
        )
        .join("")}
    </ul>
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
        <div class="note"><b>Target:</b> ${escapeHtml(String(r.target ?? ""))}</div>
        <div class="note"><b>Invalidation:</b> ${escapeHtml(String(r.invalidation ?? ""))}</div>
        ${r.narrative ? `<div class="note scenario-narr">${escapeHtml(r.narrative)}</div>` : ""}
      </div>`
      )
      .join("")}
  </div>`;
}

function renderVisionSummary(vision) {
  const notes = typeof vision.notes === "string" ? vision.notes : "";
  const ocr = vision.ocr || {};
  const cv = vision.cv || {};
  return `
    <div class="grid-2">
      <div><span class="label">OCR symbol</span><div class="value">${escapeHtml(ocr.symbol || "—")}</div></div>
      <div><span class="label">OCR TF</span><div class="value">${escapeHtml(ocr.timeframe || "—")}</div></div>
      <div><span class="label">CV trend</span><div class="value">${escapeHtml(cv.trend_guess || "—")}</div></div>
      <div><span class="label">Candles</span><div class="value">${escapeHtml(String(cv.candles_detected ?? "—"))}</div></div>
    </div>
    ${notes ? `<p class="note">${escapeHtml(notes.slice(0, 280))}</p>` : ""}
  `;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
