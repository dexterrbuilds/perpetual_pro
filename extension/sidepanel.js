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
  successBanner: $("successBanner"),
  loading: $("loading"),
  loadingText: $("loadingText"),
  loadingSteps: $("loadingSteps"),
  results: $("results"),
  thumb: $("thumb"),
  thumbWrap: $("thumbWrap"),
  btnCapture: $("btnCapture"),
  btnSelectArea: $("btnSelectArea"),
  btnRefresh: $("btnRefresh"),
  btnClear: $("btnClear"),
  symbol: $("symbol"),
  timeframe: $("timeframe"),
};

let processingPending = false;
let lastCaptureDataUrl = null;
/** Intentional one-shot override for the next capture only. */
let captureOverrideSymbol = "";
let busy = false;

init();

async function init() {
  const settings = (await chrome.runtime.sendMessage({ type: "GET_SETTINGS" }))?.settings || {};
  // Never preload a sticky symbol
  els.symbol.value = "";
  els.timeframe.value = settings.defaultTimeframe || "";
  captureOverrideSymbol = "";

  els.symbol.addEventListener("input", () => {
    captureOverrideSymbol = els.symbol.value.trim();
  });

  await refreshBackend();
  await loadLast({ restoreThumb: true });
  await claimPending();

  els.btnCapture.addEventListener("click", () => start("visible"));
  els.btnSelectArea?.addEventListener("click", () => start("select-area"));
  els.btnRefresh.addEventListener("click", onRefresh);
  els.btnClear.addEventListener("click", onClear);

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type === "STATUS") setStatus(msg.phase, msg.message);
    if (msg?.type === "RESULT") {
      setBusy(false);
      setStatus("done", "Analysis complete");
      hideError();
      renderResults(els.results, msg.result);
      const img = msg.result?._meta?.imageDataUrl || lastCaptureDataUrl;
      if (img) showThumb(img);
      flashSuccess("Analysis complete");
    }
    if (msg?.type === "PENDING_IMAGE") claimPending();
    if (msg?.type === "CLEARED") applyLocalClear();
  });

  const poll = setInterval(() => claimPending(), 800);
  setTimeout(() => clearInterval(poll), 10000);
}

async function onRefresh() {
  hideError();
  setBusy(true, "Refreshing…");
  if (els.loadingSteps) els.loadingSteps.textContent = "Backend · last report · capture";
  try {
    await refreshBackend();
    await loadLast({ restoreThumb: true });
    if (lastCaptureDataUrl) showThumb(lastCaptureDataUrl);
    flashSuccess("Refreshed");
  } finally {
    setBusy(false);
  }
}

async function onClear() {
  hideError();
  hideSuccess();
  applyLocalClear();
  await chrome.runtime.sendMessage({ type: "CLEAR_RESULT" });
  await chrome.runtime.sendMessage({
    type: "SAVE_SETTINGS",
    settings: {
      defaultSymbol: "",
      defaultTimeframe: els.timeframe.value.trim(),
    },
  });
  setStatus("idle", "Cleared · ready");
  flashSuccess("All clear");
}

function applyLocalClear() {
  lastCaptureDataUrl = null;
  captureOverrideSymbol = "";
  processingPending = false;
  els.symbol.value = "";
  els.timeframe.value = "";
  hideThumb();
  hideError();
  renderResults(els.results, null);
}

async function refreshBackend() {
  const res = await chrome.runtime.sendMessage({ type: "CHECK_BACKEND" });
  if (res?.ok) {
    setStatus("idle", "Online · ready");
    els.statusDot.className = "status-dot ok";
    hideError();
  } else {
    const msg =
      res?.error ||
      "Cannot reach Perpetual Pro API (Render may be waking up — retry in ~30s).";
    setStatus("error", msg);
    els.statusDot.className = "status-dot err";
    showError(msg);
  }
}

async function loadLast({ restoreThumb = false } = {}) {
  const res = await chrome.runtime.sendMessage({ type: "GET_LAST_RESULT" });
  const result = res?.result || null;
  renderResults(els.results, result);
  if (restoreThumb && result?._meta?.imageDataUrl) {
    lastCaptureDataUrl = result._meta.imageDataUrl;
    showThumb(lastCaptureDataUrl);
  }
}

async function start(mode) {
  hideError();
  hideSuccess();
  // Fresh capture: clear previous results so old symbol never lingers visually
  renderResults(els.results, null);

  const override = (captureOverrideSymbol || els.symbol.value).trim();

  setBusy(true, mode === "select-area" ? "Select area on the page…" : "Capturing full tab…");
  if (els.loadingSteps) {
    els.loadingSteps.textContent =
      mode === "select-area" ? "Drag on the chart region" : "Full visible tab";
  }

  const settings = (await chrome.runtime.sendMessage({ type: "GET_SETTINGS" }))?.settings || {};
  await chrome.runtime.sendMessage({
    type: "SAVE_SETTINGS",
    settings: {
      ...settings,
      defaultSymbol: "",
      defaultTimeframe: els.timeframe.value.trim(),
    },
  });

  const res = await chrome.runtime.sendMessage({
    type: "START_CAPTURE",
    mode: mode || "visible",
    symbol: override,
    timeframe: els.timeframe.value.trim(),
    freshCapture: true,
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
    setBusy(true, "Reading chart…");
    if (els.loadingSteps) els.loadingSteps.textContent = "OCR · vision · URL fusion";
    showThumb(pending.dataUrl);
    lastCaptureDataUrl = pending.dataUrl;

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

    // Only intentional override for THIS capture — never sticky previous pair
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
    let exchange = fused.exchange || pending.exchange || "";
    if (!exchange) exchange = "";
    if (timeframe) els.timeframe.value = timeframe;
    // Clear override field so it never sticks as "old symbol"
    captureOverrideSymbol = "";
    els.symbol.value = "";

    setStatus(
      "ocr",
      `Detected ${symbol || "—"} · ${timeframe || "?"} · ${clientVision?.trend_guess || "?"}`
    );

    if (!symbol) {
      setBusy(false);
      showError(
        "Symbol not detected. Type a ticker (e.g. BTC) in Symbol override, then capture again — or use the popup for manual entry."
      );
      setStatus("error", "Symbol required");
      return;
    }

    setBusy(true, `Analyzing ${symbol.split("/")[0]}…`);
    if (els.loadingSteps) els.loadingSteps.textContent = "Live data · confluence · risk plan";

    const analyze = await chrome.runtime.sendMessage({
      type: "ANALYZE_DATA_URL",
      dataUrl: pending.dataUrl,
      symbol,
      timeframe,
      exchange,
      pageUrl,
      clientOcr,
      clientVision,
      clientHints: { url: urlHints, fuse_notes: fused.notes },
      captureId: pending.id,
    });

    setBusy(false);
    if (analyze?.ok && analyze.result) {
      hideError();
      setStatus("done", "Analysis complete");
      renderResults(els.results, analyze.result);
      showThumb(pending.dataUrl);
      flashSuccess("Analysis complete");
    } else {
      const err = analyze?.error || "Analysis failed";
      showError(err);
      setStatus("error", err);
    }
  } finally {
    processingPending = false;
  }
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

function setBusy(on, text) {
  busy = on;
  els.btnCapture.disabled = on;
  if (els.btnSelectArea) els.btnSelectArea.disabled = on;
  if (els.btnRefresh) els.btnRefresh.disabled = on;
  if (els.btnClear) els.btnClear.disabled = on;
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
