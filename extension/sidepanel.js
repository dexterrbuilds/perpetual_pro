/**
 * Perpetual Pro — side panel (full results + OCR claim)
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
  btnCapture: $("btnCapture"),
  btnVisible: $("btnVisible"),
  btnRefresh: $("btnRefresh"),
  btnClear: $("btnClear"),
  symbol: $("symbol"),
  timeframe: $("timeframe"),
};

let processingPending = false;

init();

async function init() {
  const settings = (await chrome.runtime.sendMessage({ type: "GET_SETTINGS" }))?.settings || {};
  els.symbol.value = settings.defaultSymbol || "";
  els.timeframe.value = settings.defaultTimeframe || "";

  await refreshBackend();
  await loadLast();
  await claimPending();

  els.btnCapture.addEventListener("click", () => start("select-area"));
  els.btnVisible.addEventListener("click", () => start("visible"));
  els.btnRefresh.addEventListener("click", async () => {
    await refreshBackend();
    await loadLast();
  });
  els.btnClear.addEventListener("click", async () => {
    await chrome.runtime.sendMessage({ type: "CLEAR_RESULT" });
    renderResults(els.results, null);
    hideError();
  });

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type === "STATUS") setStatus(msg.phase, msg.message);
    if (msg?.type === "RESULT") {
      setBusy(false);
      setStatus("done", "Analysis complete");
      hideError();
      renderResults(els.results, msg.result);
    }
    if (msg?.type === "PENDING_IMAGE") claimPending();
  });

  // Poll briefly for pending captures started while panel was closed
  const poll = setInterval(() => claimPending(), 800);
  setTimeout(() => clearInterval(poll), 8000);
}

async function refreshBackend() {
  const settings = (await chrome.runtime.sendMessage({ type: "GET_SETTINGS" }))?.settings || {};
  const res = await chrome.runtime.sendMessage({
    type: "CHECK_BACKEND",
    apiBase: settings.apiBase,
  });
  if (res?.ok) {
    setStatus("idle", "Backend online");
    els.statusDot.className = "status-dot ok";
  } else {
    const msg =
      res?.error ||
      "Backend not running. Start: uvicorn main_server:app --reload --port 8000";
    setStatus("error", msg);
    els.statusDot.className = "status-dot err";
    showError(msg);
  }
}

async function loadLast() {
  const res = await chrome.runtime.sendMessage({ type: "GET_LAST_RESULT" });
  renderResults(els.results, res?.result || null);
}

async function start(mode) {
  hideError();
  setBusy(true, mode === "select-area" ? "Select area on the page…" : "Capturing…");
  // save overrides
  const settings = (await chrome.runtime.sendMessage({ type: "GET_SETTINGS" }))?.settings || {};
  await chrome.runtime.sendMessage({
    type: "SAVE_SETTINGS",
    settings: {
      ...settings,
      defaultSymbol: els.symbol.value.trim(),
      defaultTimeframe: els.timeframe.value.trim(),
    },
  });
  const res = await chrome.runtime.sendMessage({
    type: "START_CAPTURE",
    mode,
    symbol: els.symbol.value.trim(),
    timeframe: els.timeframe.value.trim(),
  });
  if (!res?.ok) {
    setBusy(false);
    showError(res?.error || "Capture failed");
  }
}

async function claimPending() {
  if (processingPending) return;
  const claim = await chrome.runtime.sendMessage({
    type: "CLAIM_PENDING_IMAGE",
    who: "sidepanel",
  });
  const pending = claim?.pending;
  if (!claim?.ok || !pending?.dataUrl) return;

  processingPending = true;
  try {
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
      if (clientOcr.symbol) els.symbol.value = clientOcr.symbol.replace(/USDT$/i, "");
      if (clientOcr.timeframe) els.timeframe.value = clientOcr.timeframe;
      setStatus("ocr", `OCR hint: ${symbol || "—"} ${timeframe || ""}`);
    } catch (_) {
      setStatus("ocr", "Client OCR skipped");
    }

    setBusy(true, "Sending to FastAPI /analyze…");
    const analyze = await chrome.runtime.sendMessage({
      type: "ANALYZE_DATA_URL",
      dataUrl: pending.dataUrl,
      symbol,
      timeframe,
      exchange: pending.exchange,
      apiBase: pending.apiBase,
      clientOcr,
    });

    setBusy(false);
    if (analyze?.ok && analyze.result) {
      hideError();
      setStatus("done", "Analysis complete");
      renderResults(els.results, analyze.result);
    } else {
      const err = analyze?.error || "Analysis failed";
      showError(err);
      setStatus("error", err);
    }
  } finally {
    processingPending = false;
  }
}

function setBusy(on, text) {
  els.btnCapture.disabled = on;
  els.btnVisible.disabled = on;
  if (on) {
    els.loading.classList.add("show");
    els.loadingText.textContent = text || "Working…";
    els.statusDot.className = "status-dot busy";
  } else {
    els.loading.classList.remove("show");
  }
}

function setStatus(phase, message) {
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
}
