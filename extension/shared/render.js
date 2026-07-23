/**
 * Shared result rendering for popup + sidepanel.
 * Aggressive crypto-perp pro setup cards + full confluence report.
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
 * Day-trade biased hold window. Max 12–24h for most signals;
 * longer only for strong swing (backend usually already computed).
 */
export function suggestHoldTime(tf, setupName, tags = [], direction = "", plan = {}) {
  if (plan.hold_detail || plan.hold_label) {
    return {
      label: plan.hold_label || "Day trade",
      detail: plan.hold_detail || "Suggested hold: 4–24 hours",
    };
  }
  const t = String(tf || "").toLowerCase().trim();
  const blob = `${setupName || ""} ${(tags || []).join(" ")} ${direction || ""}`.toLowerCase();
  const strongSwing =
    blob.includes("swing") &&
    (t === "4h" || t === "1d" || t === "12h" || blob.includes("trend"));

  if (blob.includes("scalp") || ["1m", "3m", "5m"].includes(t)) {
    return { label: "Scalp", detail: "Suggested hold: 15–90 minutes (max 2h)" };
  }
  if (["15m", "30m"].includes(t)) {
    return { label: "Intraday", detail: "Suggested hold: 1–8 hours (max 12h)" };
  }
  if (["1h", "2h"].includes(t)) {
    return { label: "Day trade", detail: "Suggested hold: 4–12 hours (max 24h)" };
  }
  if (["4h", "6h", "8h", "12h"].includes(t)) {
    if (strongSwing) {
      return { label: "Swing", detail: "Suggested hold: 1–3 days (strong swing only)" };
    }
    return { label: "Day / short swing", detail: "Suggested hold: 8–24 hours" };
  }
  if (["1d", "3d", "1w"].includes(t)) {
    if (strongSwing) {
      return { label: "Swing", detail: "Suggested hold: 2–5 days (strong HTF swing)" };
    }
    return { label: "Day / short swing", detail: "Suggested hold: 12–24 hours — reassess daily" };
  }
  return { label: "Day trade", detail: "Suggested hold: 4–24 hours" };
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
  const primary = result.primary_setup || {};
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
    plan.direction || result.direction,
    plan
  );
  const reasons = Array.isArray(result.key_reasons) ? result.key_reasons.slice(0, 5) : [];
  const risks = Array.isArray(result.key_risks) ? result.key_risks.slice(0, 5) : [];
  const headline =
    plan.headline ||
    primary.headline ||
    result.signal?.headline ||
    buildHeadline(result.symbol, plan.direction || result.direction);
  const lev =
    plan.leverage_suggested != null
      ? Math.round(Number(plan.leverage_suggested))
      : primary.leverage_suggested != null
        ? Math.round(Number(primary.leverage_suggested))
        : "—";

  container.innerHTML = `
    <section class="hero ${bc}">
      <div class="setup-headline">${escapeHtml(headline)}</div>
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
          <span class="label">Leverage</span>
          <span class="value lev-hot">${escapeHtml(String(lev))}x</span>
        </div>
      </div>
      <div class="setup-name">${escapeHtml(result.setup_name || "—")}</div>
      <div class="hold-badge" title="Hold window — day-trade biased">
        <span class="hold-label">${escapeHtml(hold.label)}</span>
        <span class="hold-detail">${escapeHtml(hold.detail)}</span>
      </div>
      <div class="tags">${tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>
    </section>

    <section class="card card-accent pro-setup-card">
      <h3>🚨 Trade Setup</h3>
      ${renderProSetup(plan, primary, price, headline, hold, lev)}
    </section>

    <section class="card sim-card">
      <h3>Simulation Example</h3>
      ${renderSimulation(plan, result, lev)}
    </section>

    <section class="card">
      <h3>⚠ Risk Advice</h3>
      ${renderRiskAdvice(plan, risks, lev)}
    </section>

    ${
      reasons.length
        ? `<section class="card">
            <h3>Why This Signal</h3>
            ${renderReasonsRisks(reasons, [])}
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
      ${escapeHtml(result.disclaimer || "NOT FINANCIAL ADVICE. High leverage perps can liquidate quickly.")}
    </section>
  `;
}

function buildHeadline(symbol, direction) {
  const base = prettySymbol(symbol).split("/")[0] || "PAIR";
  const d = String(direction || "").toLowerCase();
  if (d === "long") return `🚨 ${base} LONG SETUP`;
  if (d === "short") return `🚨 ${base} SHORT SETUP`;
  return `⏸ ${base} NO TRADE — STAND ASIDE`;
}

function prettySymbol(sym) {
  if (!sym) return "—";
  const s = String(sym);
  if (s.includes("/")) return s.split(":")[0];
  return s;
}

/**
 * Bold pro trader setup block.
 */
function renderProSetup(plan, primary, price, headline, hold, lev) {
  if (!plan || !plan.direction || plan.direction === "flat") {
    return `<div class="muted">No directional trade plan — stand aside or wait for clearer structure.</div>`;
  }

  const tps = Array.isArray(plan.take_profits)
    ? plan.take_profits
    : [primary.tp1, primary.tp2, primary.tp3, primary.tp4].filter((x) => x != null);
  const rrs = Array.isArray(plan.risk_reward)
    ? plan.risk_reward
    : [primary.rr_tp1, primary.rr_tp2, primary.rr_tp3, primary.rr_tp4];
  const dir = String(plan.direction).toUpperCase();
  const dirCls = biasClass(plan.direction);
  const dirEmoji = dir === "LONG" ? "🟢" : dir === "SHORT" ? "🔴" : "⚪";

  const altLow = plan.alternative_entry_low ?? primary.alternative_entry?.low;
  const altHigh = plan.alternative_entry_high ?? primary.alternative_entry?.high;
  const altNote = plan.alternative_entry_note || primary.alternative_entry?.note || "";

  const tpEmojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"];

  return `
    <div class="pro-setup">
      <div class="pro-headline">${escapeHtml(headline)}</div>
      <div class="pro-rows">
        <div class="pro-row">
          <span class="pro-k">${dirEmoji} Direction</span>
          <span class="pro-v ${dirCls}"><strong>${escapeHtml(dir)}</strong></span>
        </div>
        <div class="pro-row">
          <span class="pro-k">🎯 Entry Zone</span>
          <span class="pro-v"><strong>${fmtPrice(plan.entry_low, price)} – ${fmtPrice(plan.entry_high, price)}</strong></span>
        </div>
        ${
          altLow != null && altHigh != null
            ? `<div class="pro-row">
                <span class="pro-k">🔄 Alternative Entry</span>
                <span class="pro-v"><strong>${fmtPrice(altLow, price)} – ${fmtPrice(altHigh, price)}</strong>
                ${altNote ? `<span class="muted"> · ${escapeHtml(altNote)}</span>` : ""}</span>
              </div>`
            : ""
        }
        <div class="pro-row">
          <span class="pro-k">🛑 Stop-Loss</span>
          <span class="pro-v bear"><strong>${fmtPrice(plan.stop_loss, price)}</strong></span>
        </div>
        ${tps
          .slice(0, 4)
          .map(
            (tp, i) =>
              `<div class="pro-row">
                <span class="pro-k">${tpEmojis[i] || "•"} TP${i + 1}</span>
                <span class="pro-v bull"><strong>${fmtPrice(tp, price)}</strong>
                <span class="muted"> · R:R ${rrs[i] != null ? Number(rrs[i]).toFixed(2) : "—"}</span></span>
              </div>`
          )
          .join("")}
        <div class="pro-row">
          <span class="pro-k">⚡ Leverage</span>
          <span class="pro-v lev-hot"><strong>${escapeHtml(String(lev))}x</strong></span>
        </div>
        <div class="pro-row">
          <span class="pro-k">⏱ Hold</span>
          <span class="pro-v"><strong>${escapeHtml(hold.detail)}</strong></span>
        </div>
        <div class="pro-row">
          <span class="pro-k">❌ Invalidation</span>
          <span class="pro-v">${escapeHtml(plan.invalidation || "—")}</span>
        </div>
        <div class="pro-row">
          <span class="pro-k">⭐ Quality</span>
          <span class="pro-v"><strong>${escapeHtml((plan.quality || "—").toUpperCase())}</strong></span>
        </div>
      </div>
    </div>
  `;
}

function renderSimulation(plan, result, lev) {
  if (!plan || !plan.direction || plan.direction === "flat") {
    return `<div class="muted">No simulation — flat / neutral bias.</div>`;
  }

  const levStr = lev != null && lev !== "—" ? String(lev) : "—";
  const profits = simulationProfits(plan);
  const atrPct = plan.atr_pct ?? result?.meta?.atr_pct;
  const conf = result?.confidence;

  const lines =
    profits.length > 0
      ? profits
          .slice(0, 4)
          .map((p, i) => {
            const cls = p >= 0 ? "bull" : "bear";
            return `<li class="${cls}">At TP${i + 1} → <strong>${fmtMoney(p)}</strong> profit</li>`;
          })
          .join("")
      : `<li class="muted">Profit targets unavailable for this plan.</li>`;

  const metaBits = [];
  if (atrPct != null) metaBits.push(`ATR ${Number(atrPct).toFixed(2)}%`);
  if (conf != null) metaBits.push(`conf ${Number(conf).toFixed(0)}%`);
  metaBits.push("20x–100x band");

  return `
    <div class="sim-header">
      <span class="sim-badge">For illustration only</span>
    </div>
    <p class="sim-lead">
      If you trade this signal with <strong>$${SIM_EXAMPLE_USD}</strong> at the suggested
      <strong>${escapeHtml(levStr)}x</strong> leverage:
    </p>
    <ul class="sim-list">
      ${lines}
    </ul>
    <p class="sim-meta muted">
      Leverage is model-suggested from ${escapeHtml(metaBits.join(" · "))} — not exchange max margin.
      Not a real balance or order. High leverage can liquidate fast.
    </p>
  `;
}

function renderRiskAdvice(plan, keyRisks, lev) {
  const notes = Array.isArray(plan?.notes) ? plan.notes.filter((n) => /risk|leverage|Low confidence|High leverage|R:R/i.test(n)) : [];
  const items = [
    ...(keyRisks || []),
    ...notes.slice(0, 3),
    lev != null && Number(lev) >= 50
      ? `Suggested ${lev}x is aggressive — trail after TP1 and never move stop against you.`
      : null,
    "Risk a small % of capital at the stop; do not max margin on the full notional.",
  ].filter(Boolean);

  if (!items.length) {
    return `<div class="muted">Respect the stop. Day-trade style holds — flat if thesis breaks.</div>`;
  }
  return `<ul class="list tight risk-list">${items
    .slice(0, 6)
    .map((r) => `<li>${escapeHtml(r)}</li>`)
    .join("")}</ul>`;
}

function renderReasonsRisks(reasons, risks) {
  return `
    <div class="rr-grid${risks.length ? "" : " rr-single"}">
      <div>
        <div class="label">Key reasons</div>
        ${
          reasons.length
            ? `<ul class="list tight">${reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
            : `<div class="muted">—</div>`
        }
      </div>
      ${
        risks.length
          ? `<div>
              <div class="label">Key risks</div>
              <ul class="list tight risk-list">${risks.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
            </div>`
          : ""
      }
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

export function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
