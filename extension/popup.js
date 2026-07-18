/**
 * Perpetual Pro — popup controller
 */
import { fuseHints, lightOcr } from "./shared/ocr.js";
import { parseChartUrl } from "./shared/url_parse.js";
import { lightVision } from "./shared/vision_light.js";
import { renderResults } from "./shared/render.js";

const $ = (id) => document.getElementById(id);

const els = {
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  errorBanner: $("errorBanner"),
  successBanner: $("successBanner"),
  loading: $("loading"),
  loadingText: $("loadingText"),
  loadingSteps: $("loadingSteps"),
  results: $("results"),
  thumb: $("thumb"),
  thumbWrap: $("thumbWrap"),
  settingsPanel: $("settingsPanel"),
  btnSettings: $("btnSettings"),
  btnCapture: $("btnCapture"),
  btnSelectArea: $("btnSelectArea"),
  btnSidePanel: $("btnSidePanel"),
  btnSaveSettings: $("btnSaveSettings"),
  btnRefresh: $("btnRefresh"),
  btnClear: $("btnClear"),
  symbol: $("symbol"),
  timeframe: $("timeframe"),
  exchange: $("exchange"),
  higher: $("higher"),
  noNews: $("noNews"),
  manualPanel: $("manualSymbolPanel"),
  manualSymbol: $("manualSymbol"),
  manualTimeframe: $("manualTimeframe"),
  manualDesc: $("manualSymbolDesc"),
  manualHint: $("manualSymbolHint"),
  btnManualAnalyze: $("btnManualAnalyze"),
  btnManualDismiss: $("btnManualDismiss"),
  manualChips: $("manualSymbolChips"),
};

/** Capture context for the current image only (never reused across captures). */
let pendingCapture = null;
/** Last shown capture data URL (for Refresh). */
let lastCaptureDataUrl = null;
let busy = false;
/** Symbol the user typed as an intentional override for the next capture only. */
let captureOverrideSymbol = "";

init();

async function init() {
  await loadSettingsIntoForm();
  await refreshBackendStatus();
  await loadLastResult({ restoreThumb: true });
  await maybeProcessPending();

  els.btnSettings.addEventListener("click", () => {
    const open = !els.settingsPanel.hidden;
    els.settingsPanel.hidden = open;
    els.btnSettings.classList.toggle("active", !open);
  });

  els.btnSaveSettings.addEventListener("click", async () => {
    await saveSettingsFromForm();
    flashSuccess("Settings saved");
    setStatusUI("idle", "Settings saved · ready");
  });

  els.btnCapture.addEventListener("click", () => startCapture("visible"));
  els.btnSelectArea?.addEventListener("click", () => startCapture("select-area"));
  els.btnSidePanel.addEventListener("click", openSidePanel);

  els.btnRefresh.addEventListener("click", onRefresh);
  els.btnClear.addEventListener("click", onClear);

  // Track intentional symbol override (typed by user, not auto-filled)
  els.symbol.addEventListener("input", () => {
    captureOverrideSymbol = els.symbol.value.trim();
  });

  els.btnManualAnalyze.addEventListener("click", () => runManualAnalyze());
  els.btnManualDismiss.addEventListener("click", () => {
    hideManualSymbolPanel();
    setStatusUI("idle", "Ready — enter a symbol or capture again");
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
      const img = msg.result?._meta?.imageDataUrl || lastCaptureDataUrl;
      if (img) showThumb(img);
      flashSuccess("Analysis complete");
    }
    if (msg?.type === "PENDING_IMAGE") {
      maybeProcessPending();
    }
    if (msg?.type === "CLEARED") {
      applyLocalClear();
    }
  });
}

async function onRefresh() {
  hideError();
  setBusy(true, "Refreshing…");
  els.loadingSteps.textContent = "Backend · last report · capture";
  try {
    await refreshBackendStatus();
    await loadLastResult({ restoreThumb: true });
    // Re-show capture if we still have it in memory or storage
    if (lastCaptureDataUrl) {
      showThumb(lastCaptureDataUrl);
    } else if (pendingCapture?.dataUrl) {
      showThumb(pendingCapture.dataUrl);
    }
    if (!busy) setStatusUI("idle", "Refreshed");
    flashSuccess("Refreshed");
  } finally {
    setBusy(false);
  }
}

async function onClear() {
  hideError();
  hideSuccess();
  hideManualSymbolPanel();
  pendingCapture = null;
  lastCaptureDataUrl = null;
  captureOverrideSymbol = "";

  // Clear form fields (do not keep old symbol/TF)
  els.symbol.value = "";
  els.timeframe.value = "";
  els.manualSymbol.value = "";
  els.manualTimeframe.value = "";
  hideThumb();
  els.results.innerHTML = "";
  renderResults(els.results, null);

  await chrome.runtime.sendMessage({ type: "CLEAR_RESULT" });
  // Persist empty defaults so next capture never inherits stale symbol
  await chrome.runtime.sendMessage({
    type: "SAVE_SETTINGS",
    settings: {
      defaultSymbol: "",
      defaultTimeframe: els.timeframe.value.trim(),
      defaultExchange: els.exchange.value,
      higher: els.higher.value.trim(),
      noNews: els.noNews.checked,
      darkTheme: true,
    },
  });

  setStatusUI("idle", "Cleared · ready");
  flashSuccess("All clear");
}

function applyLocalClear() {
  pendingCapture = null;
  lastCaptureDataUrl = null;
  captureOverrideSymbol = "";
  els.symbol.value = "";
  els.timeframe.value = "";
  hideThumb();
  hideManualSymbolPanel();
  hideError();
  renderResults(els.results, null);
}

async function loadSettingsIntoForm() {
  const res = await chrome.runtime.sendMessage({ type: "GET_SETTINGS" });
  const s = res?.settings || {};
  // Do NOT load a saved defaultSymbol into the form — prevents stale pair reuse.
  // User can still type an override for the next capture.
  els.symbol.value = "";
  captureOverrideSymbol = "";
  els.timeframe.value = s.defaultTimeframe || "";
  els.exchange.value = s.defaultExchange || "binanceusdm";
  els.higher.value = s.higher || "5m,1h,4h,1d";
  els.noNews.checked = !!s.noNews;
}

async function saveSettingsFromForm() {
  const settings = {
    // Never persist symbol as a sticky default — override is session-only
    defaultSymbol: "",
    defaultTimeframe: els.timeframe.value.trim(),
    defaultExchange: els.exchange.value,
    higher: els.higher.value.trim(),
    noNews: els.noNews.checked,
    darkTheme: true,
  };
  captureOverrideSymbol = els.symbol.value.trim();
  await chrome.runtime.sendMessage({ type: "SAVE_SETTINGS", settings });
}

async function refreshBackendStatus() {
  const res = await chrome.runtime.sendMessage({ type: "CHECK_BACKEND" });
  if (res?.ok) {
    setStatusUI("idle", "Online · ready");
    els.statusDot.className = "status-dot ok";
    hideError();
  } else {
    const msg =
      res?.error ||
      "Cannot reach Perpetual Pro API (Render may be waking up — retry in ~30s).";
    setStatusUI("error", msg);
    els.statusDot.className = "status-dot err";
    showError(msg);
  }
}

async function loadLastResult({ restoreThumb = false } = {}) {
  const res = await chrome.runtime.sendMessage({ type: "GET_LAST_RESULT" });
  if (res?.result) {
    renderResults(els.results, res.result);
    if (restoreThumb) {
      const img = res.result._meta?.imageDataUrl;
      if (img) {
        lastCaptureDataUrl = img;
        showThumb(img);
      }
    }
  } else {
    renderResults(els.results, null);
  }
}

async function startCapture(mode) {
  hideError();
  hideSuccess();
  hideManualSymbolPanel();
  // Drop previous analysis UI so we never show / send a stale pair
  pendingCapture = null;
  els.results.innerHTML = "";
  renderResults(els.results, null);

  // Intentional override for THIS capture only (typed in settings)
  const override = (captureOverrideSymbol || els.symbol.value).trim();

  setBusy(
    true,
    mode === "select-area" ? "Select chart area on the page…" : "Capturing full tab…"
  );
  els.loadingSteps.textContent =
    mode === "select-area" ? "Drag on the chart region" : "Full visible tab";

  await saveSettingsFromForm();

  const res = await chrome.runtime.sendMessage({
    type: "START_CAPTURE",
    mode: mode || "visible",
    // Only send symbol if user typed an override for this capture
    symbol: override,
    timeframe: els.timeframe.value.trim(),
    exchange: els.exchange.value,
    // Force fresh resolution — background must not inject old defaults
    freshCapture: true,
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

  setBusy(true, "Reading chart…");
  els.loadingSteps.textContent = "OCR · vision · URL fusion";
  showThumb(pending.dataUrl);
  lastCaptureDataUrl = pending.dataUrl;

  const pageUrl = pending.pageUrl || "";
  const urlHints = parseChartUrl(pageUrl);
  if (urlHints.symbol) {
    setStatusUI("ocr", `URL: ${urlHints.symbol}`);
  }

  let clientOcr = null;
  let clientVision = null;
  try {
    clientOcr = await lightOcr(pending.dataUrl);
  } catch (e) {
    clientOcr = { error: String(e?.message || e), symbol: "", timeframe: "" };
  }
  try {
    clientVision = await lightVision(pending.dataUrl);
  } catch (e) {
    clientVision = { trend_guess: "unknown", confidence: 0, notes: [String(e)] };
  }

  // Fresh fusion: prefer OCR + URL; only use intentional override for THIS capture
  const userOverride =
    (pending.freshCapture ? pending.symbol : "") ||
    captureOverrideSymbol ||
    els.symbol.value.trim() ||
    "";

  const fused = fuseHints({
    userSymbol: userOverride,
    userTf: els.timeframe.value.trim() || pending.timeframe || "",
    urlHints,
    ocr: clientOcr,
    vision: clientVision,
    preferDetection: true,
  });

  let symbol = fused.symbol;
  let timeframe = fused.timeframe;
  let exchange = fused.exchange || pending.exchange || els.exchange.value;

  // Reflect detected TF (not sticky symbol from old charts)
  if (timeframe) els.timeframe.value = timeframe;
  if (exchange && els.exchange) {
    try {
      els.exchange.value = exchange;
    } catch (_) {}
  }

  setStatusUI(
    "ocr",
    `Detected: ${symbol || "—"} · ${timeframe || "?"} · ${clientVision?.trend_guess || "?"}`
  );

  pendingCapture = {
    dataUrl: pending.dataUrl,
    timeframe: timeframe || "",
    exchange,
    pageUrl,
    clientOcr,
    clientVision,
    clientHints: { url: urlHints, fuse_notes: fused.notes },
    captureId: pending.id || `${Date.now()}`,
  };

  if (!normalizeSymbolInput(symbol)) {
    setBusy(false);
    const snippet = clientOcr?.raw
      ? `OCR: “${String(clientOcr.raw).slice(0, 80).replace(/\s+/g, " ")}…”`
      : pageUrl
        ? `Page: ${pageUrl.slice(0, 60)}…`
        : "Tip: open a TradingView chart or type BTC / ETH";
    showManualSymbolPanel({
      reason:
        "Could not auto-detect the symbol from OCR or the page URL. Enter the ticker to continue.",
      timeframe: pendingCapture.timeframe,
      hint: snippet,
    });
    setStatusUI("error", "Symbol required");
    return;
  }

  // Clear one-shot override so the next capture starts clean
  captureOverrideSymbol = "";
  els.symbol.value = "";

  await runAnalyzeWithSymbol(symbol, timeframe, pendingCapture);
}

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
    hideManualSymbolPanel();
    setStatusUI("idle", `Symbol ${symbol} noted — capture a chart first`);
    showError("No chart captured yet. Click “Capture & Analyze”, then enter the symbol if needed.");
    // Store as one-shot override for the next capture only
    captureOverrideSymbol = symbol;
    els.symbol.value = symbol;
    return;
  }

  hideManualSymbolPanel();
  captureOverrideSymbol = "";
  els.symbol.value = "";
  await runAnalyzeWithSymbol(symbol, timeframe, pendingCapture);
}

async function runAnalyzeWithSymbol(symbol, timeframe, ctx) {
  hideError();
  setBusy(true, `Analyzing ${displayBase(symbol)}…`);
  els.loadingSteps.textContent = "Live data · confluence · risk plan";
  setStatusUI("analyzing", `Sending ${displayBase(symbol)}…`);

  const analyze = await chrome.runtime.sendMessage({
    type: "ANALYZE_DATA_URL",
    dataUrl: ctx.dataUrl,
    symbol,
    timeframe: timeframe || "",
    exchange: ctx.exchange || els.exchange.value,
    pageUrl: ctx.pageUrl || "",
    clientOcr: ctx.clientOcr,
    clientVision: ctx.clientVision,
    clientHints: ctx.clientHints,
    captureId: ctx.captureId,
  });

  setBusy(false);

  if (analyze?.ok && analyze.result) {
    hideError();
    hideManualSymbolPanel();
    pendingCapture = null;
    lastCaptureDataUrl = ctx.dataUrl;
    setStatusUI("done", "Analysis complete");
    renderResults(els.results, analyze.result);
    showThumb(ctx.dataUrl);
    flashSuccess(`${displayBase(symbol)} · analysis ready`);
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
      hint: "Try BTC, ETH, or BTC/USDT:USDT",
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
    reason || "Couldn’t read the ticker. Enter the symbol to continue.";
  els.manualHint.textContent = hint || "";
  els.manualHint.classList.remove("error");
  els.manualTimeframe.value = timeframe || els.timeframe.value.trim() || "";
  if (prefill) els.manualSymbol.value = prefill;
  requestAnimationFrame(() => {
    els.manualSymbol.focus();
    els.manualSymbol.select();
  });
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
  return String(raw).trim().toUpperCase().replace(/\s+/g, "");
}

function normalizeTimeframe(raw) {
  if (!raw) return "";
  return String(raw).trim().toLowerCase();
}

function displayBase(sym) {
  if (!sym) return "—";
  return String(sym).split("/")[0].replace(/USDT$/i, "") || sym;
}

function showThumb(dataUrl) {
  if (!dataUrl) return;
  els.thumb.src = dataUrl;
  els.thumb.classList.add("show");
  if (els.thumbWrap) els.thumbWrap.hidden = false;
  lastCaptureDataUrl = dataUrl;
}

function hideThumb() {
  els.thumb.removeAttribute("src");
  els.thumb.classList.remove("show");
  if (els.thumbWrap) els.thumbWrap.hidden = true;
}

async function openSidePanel() {
  await chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL" });
}

function setBusy(on, text) {
  busy = on;
  els.btnCapture.disabled = on;
  if (els.btnSelectArea) els.btnSelectArea.disabled = on;
  if (els.btnRefresh) els.btnRefresh.disabled = on;
  if (els.btnClear) els.btnClear.disabled = on;
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
  hideSuccess();
  els.errorBanner.textContent = msg;
  els.errorBanner.classList.add("show");
}

function hideError() {
  els.errorBanner.classList.remove("show");
  els.errorBanner.textContent = "";
}

function flashSuccess(msg) {
  if (!els.successBanner) return;
  hideError();
  els.successBanner.textContent = msg;
  els.successBanner.classList.add("show");
  clearTimeout(flashSuccess._t);
  flashSuccess._t = setTimeout(() => hideSuccess(), 2200);
}

function hideSuccess() {
  if (!els.successBanner) return;
  els.successBanner.classList.remove("show");
  els.successBanner.textContent = "";
}
