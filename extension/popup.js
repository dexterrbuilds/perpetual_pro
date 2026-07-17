/**
 * Perpetual Pro — popup controller
 */
import { lightOcr } from "./shared/ocr.js";
import { renderResults } from "./shared/render.js";

const $ = (id) => document.getElementById(id);

const els = {
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  errorBanner: $("errorBanner"),
  loading: $("loading"),
  loadingText: $("loadingText"),
  results: $("results"),
  thumb: $("thumb"),
  settingsPanel: $("settingsPanel"),
  btnSettings: $("btnSettings"),
  btnCapture: $("btnCapture"),
  btnVisible: $("btnVisible"),
  btnSidePanel: $("btnSidePanel"),
  btnSaveSettings: $("btnSaveSettings"),
  btnRefresh: $("btnRefresh"),
  btnClear: $("btnClear"),
  apiBase: $("apiBase"),
  symbol: $("symbol"),
  timeframe: $("timeframe"),
  exchange: $("exchange"),
  higher: $("higher"),
  balance: $("balance"),
  risk: $("risk"),
  noNews: $("noNews"),
  // Manual symbol recovery UI
  manualPanel: $("manualSymbolPanel"),
  manualSymbol: $("manualSymbol"),
  manualTimeframe: $("manualTimeframe"),
  manualDesc: $("manualSymbolDesc"),
  manualHint: $("manualSymbolHint"),
  btnManualAnalyze: $("btnManualAnalyze"),
  btnManualDismiss: $("btnManualDismiss"),
  manualChips: $("manualSymbolChips"),
};

/** @type {null | {
 *   dataUrl: string,
 *   timeframe: string,
 *   exchange: string,
 *   apiBase: string,
 *   clientOcr: object | null,
 *   reason: string
 * }} */
let pendingCapture = null;
let busy = false;

init();

async function init() {
  await loadSettingsIntoForm();
  await refreshBackendStatus();
  await loadLastResult();
  await maybeProcessPending();

  els.btnSettings.addEventListener("click", () => {
    const open = els.settingsPanel.style.display !== "none";
    els.settingsPanel.style.display = open ? "none" : "grid";
  });

  els.btnSaveSettings.addEventListener("click", saveSettingsFromForm);
  els.btnCapture.addEventListener("click", () => startCapture("select-area"));
  els.btnVisible.addEventListener("click", () => startCapture("visible"));
  els.btnSidePanel.addEventListener("click", openSidePanel);
  els.btnRefresh.addEventListener("click", async () => {
    await refreshBackendStatus();
    await loadLastResult();
  });
  els.btnClear.addEventListener("click", async () => {
    await chrome.runtime.sendMessage({ type: "CLEAR_RESULT" });
    els.results.innerHTML = "";
    renderResults(els.results, null);
    hideError();
    hideManualSymbolPanel();
    pendingCapture = null;
  });

  // Manual symbol panel
  els.btnManualAnalyze.addEventListener("click", () => runManualAnalyze());
  els.btnManualDismiss.addEventListener("click", () => {
    hideManualSymbolPanel();
    setStatusUI("idle", "Ready — enter a symbol anytime, or capture again");
  });
  els.manualSymbol.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      runManualAnalyze();
    }
  });
  els.manualChips?.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-symbol]");
    if (!btn) return;
    els.manualSymbol.value = btn.getAttribute("data-symbol") || "";
    els.manualSymbol.focus();
    els.manualSymbol.select();
  });

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type === "STATUS") setStatusUI(msg.phase, msg.message);
    if (msg?.type === "RESULT") {
      setBusy(false);
      hideManualSymbolPanel();
      pendingCapture = null;
      setStatusUI("done", "Analysis complete");
      renderResults(els.results, msg.result);
    }
    if (msg?.type === "PENDING_IMAGE") {
      maybeProcessPending();
    }
  });
}

async function loadSettingsIntoForm() {
  const res = await chrome.runtime.sendMessage({ type: "GET_SETTINGS" });
  const s = res?.settings || {};
  els.apiBase.value = s.apiBase || "http://localhost:8000";
  els.symbol.value = s.defaultSymbol || "";
  els.timeframe.value = s.defaultTimeframe || "";
  els.exchange.value = s.defaultExchange || "binanceusdm";
  els.higher.value = s.higher || "1h,4h,1d";
  els.balance.value = s.balance ?? "";
  els.risk.value = s.risk ?? "";
  els.noNews.checked = !!s.noNews;
}

async function saveSettingsFromForm() {
  const settings = {
    apiBase: els.apiBase.value.trim() || "http://localhost:8000",
    defaultSymbol: els.symbol.value.trim(),
    defaultTimeframe: els.timeframe.value.trim(),
    defaultExchange: els.exchange.value,
    higher: els.higher.value.trim(),
    balance: els.balance.value,
    risk: els.risk.value,
    noNews: els.noNews.checked,
    darkTheme: true,
  };
  await chrome.runtime.sendMessage({ type: "SAVE_SETTINGS", settings });
  setStatusUI("idle", "Settings saved");
  await refreshBackendStatus();
}

async function refreshBackendStatus() {
  const apiBase = els.apiBase.value.trim() || "http://localhost:8000";
  const res = await chrome.runtime.sendMessage({ type: "CHECK_BACKEND", apiBase });
  if (res?.ok) {
    setStatusUI("idle", "Backend online · ready");
    els.statusDot.className = "status-dot ok";
    hideError();
  } else {
    const msg =
      res?.error ||
      "Backend not running. Start: uvicorn main_server:app --reload --port 8000";
    setStatusUI("error", msg);
    els.statusDot.className = "status-dot err";
    showError(msg);
  }
}

async function loadLastResult() {
  const res = await chrome.runtime.sendMessage({ type: "GET_LAST_RESULT" });
  if (res?.result) renderResults(els.results, res.result);
  else renderResults(els.results, null);
}

async function startCapture(mode) {
  hideError();
  hideManualSymbolPanel();
  setBusy(true, mode === "select-area" ? "Select chart area on the page…" : "Capturing tab…");
  await saveSettingsFromForm();
  const res = await chrome.runtime.sendMessage({
    type: "START_CAPTURE",
    mode,
    symbol: els.symbol.value.trim(),
    timeframe: els.timeframe.value.trim(),
    exchange: els.exchange.value,
  });
  if (!res?.ok) {
    setBusy(false);
    showError(res?.error || "Capture failed");
    setStatusUI("error", res?.error || "Capture failed");
  } else if (mode === "select-area") {
    setStatusUI("selecting", "Drag on the page to select the chart…");
    window.close();
  }
}

async function maybeProcessPending() {
  const claim = await chrome.runtime.sendMessage({
    type: "CLAIM_PENDING_IMAGE",
    who: "popup",
  });
  const pending = claim?.pending;
  if (!claim?.ok || !pending?.dataUrl) return;

  setBusy(true, "Light OCR (Tesseract.js)…");
  els.thumb.src = pending.dataUrl;
  els.thumb.classList.add("show");

  let symbol = els.symbol.value.trim() || pending.symbol || "";
  let timeframe = els.timeframe.value.trim() || pending.timeframe || "";
  let clientOcr = null;

  try {
    clientOcr = await lightOcr(pending.dataUrl);
    if (!symbol && clientOcr.symbol) symbol = clientOcr.symbol;
    if (!timeframe && clientOcr.timeframe) timeframe = clientOcr.timeframe;
    if (clientOcr.symbol && !els.symbol.value) {
      els.symbol.value = clientOcr.symbol.replace(/USDT$/i, "");
    }
    if (clientOcr.timeframe && !els.timeframe.value) {
      els.timeframe.value = clientOcr.timeframe;
    }
    setStatusUI("ocr", `OCR: ${symbol || "no symbol"} · ${timeframe || "?"} `);
  } catch (e) {
    setStatusUI("ocr", "OCR failed — enter symbol manually");
  }

  // Keep capture context for manual retry
  pendingCapture = {
    dataUrl: pending.dataUrl,
    timeframe: timeframe || els.timeframe.value.trim() || "",
    exchange: pending.exchange || els.exchange.value,
    apiBase: pending.apiBase || els.apiBase.value,
    clientOcr,
    reason: "ocr_miss",
  };

  // If still no symbol after OCR → prompt user (don't call backend empty)
  if (!normalizeSymbolInput(symbol)) {
    setBusy(false);
    showManualSymbolPanel({
      reason: clientOcr?.error
        ? "OCR error — enter the symbol to continue."
        : "OCR couldn’t read the ticker from this chart. Enter the symbol to continue analysis.",
      timeframe: pendingCapture.timeframe,
      hint: clientOcr?.raw
        ? `OCR text snippet: “${String(clientOcr.raw).slice(0, 80).replace(/\s+/g, " ")}…”`
        : "Tip: BTC, ETH, or full form BTC/USDT:USDT",
    });
    setStatusUI("error", "Symbol required — enter ticker below");
    return;
  }

  await runAnalyzeWithSymbol(symbol, timeframe, pendingCapture);
}

/**
 * User clicked Analyze on the manual symbol panel.
 */
async function runManualAnalyze() {
  const symbol = normalizeSymbolInput(els.manualSymbol.value);
  if (!symbol) {
    els.manualHint.textContent = "Please enter a symbol (e.g. BTC or BTC/USDT:USDT).";
    els.manualHint.classList.add("error");
    els.manualSymbol.focus();
    els.manualSymbol.classList.add("invalid");
    return;
  }
  els.manualSymbol.classList.remove("invalid");
  els.manualHint.classList.remove("error");
  els.manualHint.textContent = "";

  // Sync into settings field for next captures
  els.symbol.value = symbol.replace(/\/USDT:USDT$/i, "").replace(/USDT$/i, "") || symbol;
  const timeframe =
    normalizeTimeframe(els.manualTimeframe.value) ||
    els.timeframe.value.trim() ||
    pendingCapture?.timeframe ||
    "";
  if (timeframe) {
    els.timeframe.value = timeframe;
    if (pendingCapture) pendingCapture.timeframe = timeframe;
  }

  if (!pendingCapture?.dataUrl) {
    // No capture yet — save symbol and prompt to capture
    await saveSettingsFromForm();
    hideManualSymbolPanel();
    setStatusUI("idle", `Symbol set to ${symbol} — capture a chart to analyze`);
    showError("No chart captured yet. Click “Capture & Analyze Chart”, then confirm.");
    return;
  }

  hideManualSymbolPanel();
  await runAnalyzeWithSymbol(symbol, timeframe, pendingCapture);
}

async function runAnalyzeWithSymbol(symbol, timeframe, ctx) {
  hideError();
  setBusy(true, `Analyzing ${symbol}…`);
  setStatusUI("analyzing", `Sending ${symbol} to backend…`);

  const analyze = await chrome.runtime.sendMessage({
    type: "ANALYZE_DATA_URL",
    dataUrl: ctx.dataUrl,
    symbol,
    timeframe: timeframe || "",
    exchange: ctx.exchange || els.exchange.value,
    apiBase: ctx.apiBase || els.apiBase.value,
    clientOcr: ctx.clientOcr,
  });

  setBusy(false);

  if (analyze?.ok && analyze.result) {
    hideError();
    hideManualSymbolPanel();
    pendingCapture = null;
    setStatusUI("done", "Analysis complete");
    renderResults(els.results, analyze.result);
    // Persist successful symbol
    els.symbol.value = symbol.replace(/\/USDT:USDT$/i, "").replace(/USDT$/i, "") || symbol;
    await saveSettingsFromForm();
    return;
  }

  const err = analyze?.error || "Analysis failed";
  const needsSymbol =
    /symbol/i.test(err) ||
    analyze?.result?.error === "symbol_required" ||
    analyze?.result?.error === "invalid_symbol";

  if (needsSymbol) {
    showManualSymbolPanel({
      reason: err,
      timeframe: timeframe || ctx.timeframe || "",
      hint: "Check the ticker format, e.g. BTC or BTC/USDT:USDT",
      prefill: symbol,
    });
    setStatusUI("error", "Symbol issue — try again");
  } else {
    showError(err);
    setStatusUI("error", err);
  }
}

function showManualSymbolPanel({ reason, timeframe, hint, prefill } = {}) {
  els.manualPanel.hidden = false;
  els.manualDesc.textContent =
    reason ||
    "OCR couldn’t read the ticker from the chart. Enter the symbol to continue analysis.";
  els.manualHint.textContent = hint || "";
  els.manualHint.classList.remove("error");
  els.manualTimeframe.value = timeframe || els.timeframe.value.trim() || "";
  if (prefill) {
    els.manualSymbol.value = prefill;
  } else if (!els.manualSymbol.value && els.symbol.value) {
    els.manualSymbol.value = els.symbol.value;
  }
  // Focus after paint
  requestAnimationFrame(() => {
    els.manualSymbol.focus();
    els.manualSymbol.select();
  });
  // Scroll panel into view inside popup
  els.manualPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideManualSymbolPanel() {
  els.manualPanel.hidden = true;
  els.manualSymbol.classList.remove("invalid");
  if (els.manualHint) {
    els.manualHint.textContent = "";
    els.manualHint.classList.remove("error");
  }
}

function normalizeSymbolInput(raw) {
  if (!raw) return "";
  let s = String(raw).trim().toUpperCase().replace(/\s+/g, "");
  if (!s) return "";
  // Keep unified form if user typed it; otherwise bare base is fine for backend
  return s;
}

function normalizeTimeframe(raw) {
  if (!raw) return "";
  return String(raw).trim().toLowerCase();
}

async function openSidePanel() {
  await chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL" });
}

function setBusy(on, text) {
  busy = on;
  els.btnCapture.disabled = on;
  els.btnVisible.disabled = on;
  if (els.btnManualAnalyze) els.btnManualAnalyze.disabled = on;
  if (on) {
    els.loading.classList.add("show");
    els.loadingText.textContent = text || "Working…";
    els.statusDot.className = "status-dot busy";
  } else {
    els.loading.classList.remove("show");
  }
}

function setStatusUI(phase, message) {
  els.statusText.textContent = message || phase || "Ready";
  if (phase === "error") els.statusDot.className = "status-dot err";
  else if (phase === "done" || phase === "idle") els.statusDot.className = "status-dot ok";
  else els.statusDot.className = "status-dot busy";
}

function showError(msg) {
  els.errorBanner.textContent = msg;
  els.errorBanner.classList.add("show");
}

function hideError() {
  els.errorBanner.classList.remove("show");
  els.errorBanner.textContent = "";
}
