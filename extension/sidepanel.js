/**
 * Perpetual Pro — side panel (full results + OCR claim)
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
  loading: $("loading"),
  loadingText: $("loadingText"),
  results: $("results"),
  thumb: $("thumb"),
  btnCapture: $("btnCapture"),
  btnSelectArea: $("btnSelectArea"),
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

  els.btnCapture.addEventListener("click", () => start("visible"));
  els.btnSelectArea?.addEventListener("click", () => start("select-area"));
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
  const res = await chrome.runtime.sendMessage({ type: "CHECK_BACKEND" });
  if (res?.ok) {
    setStatus("idle", "API online");
    els.statusDot.className = "status-dot ok";
  } else {
    const msg =
      res?.error ||
      "Cannot reach Perpetual Pro API (Render may be waking up — retry in ~30s).";
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
  setBusy(true, mode === "select-area" ? "Select area on the page…" : "Capturing full tab…");
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
    mode: mode || "visible",
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
    setBusy(true, "OCR + vision + URL fusion…");
    els.thumb.src = pending.dataUrl;
    els.thumb.classList.add("show");

    const pageUrl = pending.pageUrl || "";
    const urlHints = parseChartUrl(pageUrl);

    let clientOcr = null;
    let clientVision = null;
    try {
      clientOcr = await lightOcr(pending.dataUrl);
    } catch (_) {
      clientOcr = { symbol: "", timeframe: "" };
    }
    try {
      clientVision = await lightVision(pending.dataUrl);
    } catch (_) {
      clientVision = { trend_guess: "unknown", confidence: 0 };
    }

    const fused = fuseHints({
      userSymbol: els.symbol.value.trim() || pending.symbol || "",
      userTf: els.timeframe.value.trim() || pending.timeframe || "",
      urlHints,
      ocr: clientOcr,
      vision: clientVision,
    });

    let symbol = fused.symbol;
    let timeframe = fused.timeframe;
    if (clientOcr?.symbol) els.symbol.value = clientOcr.symbol.replace(/USDT$/i, "");
    else if (urlHints.symbol) els.symbol.value = urlHints.symbol.replace(/USDT$/i, "");
    if (timeframe) els.timeframe.value = timeframe;

    setStatus(
      "ocr",
      `Fused ${symbol || "?"} · ${timeframe || "?"} · vision ${clientVision?.trend_guess || "?"}`
    );

    setBusy(true, "Sending to API /analyze…");
    const analyze = await chrome.runtime.sendMessage({
      type: "ANALYZE_DATA_URL",
      dataUrl: pending.dataUrl,
      symbol,
      timeframe,
      exchange: fused.exchange || pending.exchange,
      pageUrl,
      clientOcr,
      clientVision,
      clientHints: { url: urlHints, fuse_notes: fused.notes },
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
  if (els.btnSelectArea) els.btnSelectArea.disabled = on;
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
