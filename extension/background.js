/**
 * Perpetual Pro — MV3 service worker
 * Commands, context menus, capture orchestration, API bridge.
 */

/** Fixed production API — not user-configurable. */
const API_BASE = "https://perpetual-pro.onrender.com";

/**
 * Hidden educational simulation capital only.
 * Never shown as a wallet / settings field.
 */
const SIMULATED_CAPITAL_USD = 100;
const DEFAULT_RISK_PCT = 1;

const STORAGE_KEYS = {
  lastResult: "pp_last_result",
  settings: "pp_settings",
  status: "pp_status",
  pending: "pp_pending_image",
  lastCapture: "pp_last_capture",
};

// ---------------------------------------------------------------------------
// Install / menus / side panel
// ---------------------------------------------------------------------------

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "pp-analyze-image",
      title: "Analyze with Perpetual Pro",
      contexts: ["image"],
    });
    chrome.contextMenus.create({
      id: "pp-capture-analyze",
      title: "Capture & Analyze Chart (Perpetual Pro)",
      contexts: ["page", "frame", "selection"],
    });
    chrome.contextMenus.create({
      id: "pp-select-area",
      title: "Select chart area & analyze",
      contexts: ["page", "frame"],
    });
  });

  if (chrome.sidePanel?.setPanelBehavior) {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false }).catch(() => {});
  }
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;
  try {
    if (info.menuItemId === "pp-analyze-image" && info.srcUrl) {
      await runPipeline({
        tabId: tab.id,
        mode: "image-url",
        imageUrl: info.srcUrl,
        windowId: tab.windowId,
        pageUrl: tab.url || "",
        freshCapture: true,
      });
    } else if (info.menuItemId === "pp-select-area") {
      await runPipeline({
        tabId: tab.id,
        mode: "select-area",
        windowId: tab.windowId,
        pageUrl: tab.url || "",
        freshCapture: true,
      });
    } else if (info.menuItemId === "pp-capture-analyze") {
      await runPipeline({
        tabId: tab.id,
        mode: "visible",
        windowId: tab.windowId,
        pageUrl: tab.url || "",
        freshCapture: true,
      });
    }
  } catch (err) {
    await setStatus({ phase: "error", message: String(err?.message || err) });
  }
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "capture-analyze") return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;
  try {
    await runPipeline({
      tabId: tab.id,
      mode: "visible",
      windowId: tab.windowId,
      pageUrl: tab.url || "",
      freshCapture: true,
    });
  } catch (err) {
    await setStatus({ phase: "error", message: String(err?.message || err) });
  }
});

// ---------------------------------------------------------------------------
// Messaging
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender)
    .then(sendResponse)
    .catch((err) => sendResponse({ ok: false, error: String(err?.message || err) }));
  return true;
});

async function handleMessage(message, sender) {
  const type = message?.type;

  if (type === "GET_STATUS") {
    return { ok: true, ...(await getStatus()) };
  }
  if (type === "GET_LAST_RESULT") {
    const data = await chrome.storage.local.get([STORAGE_KEYS.lastResult]);
    return { ok: true, result: data[STORAGE_KEYS.lastResult] || null };
  }
  if (type === "GET_SETTINGS") {
    return { ok: true, settings: await getSettings() };
  }
  if (type === "SAVE_SETTINGS") {
    await saveSettings(message.settings || {});
    return { ok: true };
  }
  if (type === "CHECK_BACKEND") {
    return checkBackend();
  }
  if (type === "SCAN_SYMBOLS") {
    return scanSymbols(message);
  }
  if (type === "START_CAPTURE") {
    const tabId = message.tabId || sender.tab?.id;
    let tab;
    if (tabId) {
      tab = await chrome.tabs.get(tabId);
    } else {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      tab = tabs[0];
    }
    if (!tab?.id) throw new Error("No active tab");
    // Wipe previous result so UI never reuses stale analysis while capturing
    if (message.freshCapture) {
      await chrome.storage.local.remove([STORAGE_KEYS.lastResult]);
    }
    await runPipeline({
      tabId: tab.id,
      mode: message.mode || "visible",
      windowId: tab.windowId,
      // Only explicit override from UI — never pull sticky defaultSymbol
      symbol: message.symbol || "",
      timeframe: message.timeframe || "",
      exchange: message.exchange || "",
      pageUrl: tab.url || message.pageUrl || "",
      freshCapture: true,
    });
    return { ok: true };
  }
  if (type === "AREA_SELECTED") {
    const tabId = sender.tab?.id;
    if (!tabId) throw new Error("No tab for area selection");
    let pageUrl = sender.tab?.url || "";
    try {
      const t = await chrome.tabs.get(tabId);
      pageUrl = t.url || pageUrl;
    } catch (_) {}
    await runPipeline({
      tabId,
      mode: "crop",
      rect: message.rect,
      dpr: message.dpr || 1,
      windowId: sender.tab.windowId,
      symbol: message.symbol || "",
      timeframe: message.timeframe || "",
      exchange: message.exchange || "",
      pageUrl,
      freshCapture: true,
    });
    return { ok: true };
  }
  if (type === "AREA_CANCELLED") {
    await setStatus({ phase: "idle", message: "Selection cancelled" });
    broadcast({ type: "STATUS", phase: "idle", message: "Selection cancelled" });
    return { ok: true };
  }
  if (type === "OPEN_SIDE_PANEL") {
    const windowId = message.windowId || sender.tab?.windowId;
    if (windowId != null && chrome.sidePanel?.open) {
      await chrome.sidePanel.open({ windowId });
    } else {
      const win = await chrome.windows.getCurrent();
      if (chrome.sidePanel?.open) await chrome.sidePanel.open({ windowId: win.id });
    }
    return { ok: true };
  }
  if (type === "CLEAR_RESULT") {
    await chrome.storage.local.remove([
      STORAGE_KEYS.lastResult,
      STORAGE_KEYS.pending,
      STORAGE_KEYS.lastCapture,
    ]);
    await setStatus({ phase: "idle", message: "Ready" });
    broadcast({ type: "CLEARED" });
    broadcast({ type: "STATUS", phase: "idle", message: "Ready" });
    return { ok: true };
  }
  if (type === "ANALYZE_DATA_URL") {
    const settings = await getSettings();
    try {
      const result = await postAnalyze({
        dataUrl: message.dataUrl,
        // Always the symbol for THIS capture only
        symbol: message.symbol || "",
        timeframe: message.timeframe || "",
        exchange: message.exchange || settings.defaultExchange || "",
        settings,
        pageUrl: message.pageUrl || "",
        clientOcr: message.clientOcr || null,
        clientVision: message.clientVision || null,
        clientHints: message.clientHints || null,
      });
      const packaged = {
        ...result,
        _meta: {
          capturedAt: new Date().toISOString(),
          clientOcr: message.clientOcr || null,
          clientVision: message.clientVision || null,
          pageUrl: message.pageUrl || "",
          apiBase: API_BASE,
          captureId: message.captureId || null,
          // Persist image so Refresh can re-display the chart
          imageDataUrl: message.dataUrl || null,
          symbolUsed: message.symbol || result.symbol || "",
        },
      };
      await chrome.storage.local.set({
        [STORAGE_KEYS.lastResult]: packaged,
        [STORAGE_KEYS.lastCapture]: {
          dataUrl: message.dataUrl,
          ts: Date.now(),
          symbol: message.symbol || "",
        },
      });
      await chrome.storage.local.remove([STORAGE_KEYS.pending]);
      await setStatus({ phase: "done", message: "Analysis complete" });
      broadcast({ type: "RESULT", result: packaged });
      return { ok: true, result: packaged };
    } catch (err) {
      const msg = String(err?.message || err);
      const friendly = friendlyBackendError(msg);
      await setStatus({ phase: "error", message: friendly });
      broadcast({ type: "STATUS", phase: "error", message: friendly });
      return { ok: false, error: friendly };
    }
  }
  if (type === "GET_PENDING_IMAGE") {
    const data = await chrome.storage.local.get([STORAGE_KEYS.pending]);
    return { ok: true, pending: data[STORAGE_KEYS.pending] || null };
  }
  if (type === "CLAIM_PENDING_IMAGE") {
    const who = message.who || "ui";
    const pending = await tryClaimPending(who, message.id || null);
    return { ok: !!pending, pending };
  }
  if (type === "GET_LAST_CAPTURE") {
    const data = await chrome.storage.local.get([STORAGE_KEYS.lastCapture]);
    return { ok: true, capture: data[STORAGE_KEYS.lastCapture] || null };
  }

  return { ok: false, error: "Unknown message type" };
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

async function runPipeline(opts) {
  const settings = await getSettings();

  await setStatus({ phase: "capturing", message: "Capturing chart…" });
  broadcast({ type: "STATUS", phase: "capturing", message: "Capturing chart…" });

  const health = await checkBackend();
  if (!health.ok) {
    const msg = health.error || friendlyBackendError("unreachable");
    await setStatus({ phase: "error", message: msg });
    broadcast({ type: "STATUS", phase: "error", message: msg });
    try {
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icons/icon128.png",
        title: "Perpetual Pro",
        message: msg,
      });
    } catch (_) {}
    throw new Error(msg);
  }

  let dataUrl;

  if (opts.mode === "image-url" && opts.imageUrl) {
    dataUrl = await fetchImageAsDataUrl(opts.imageUrl);
  } else if (opts.mode === "select-area") {
    await setStatus({ phase: "selecting", message: "Drag to select chart area…" });
    broadcast({ type: "STATUS", phase: "selecting", message: "Drag to select chart area…" });
    await ensureContentScript(opts.tabId);
    try {
      await chrome.tabs.sendMessage(opts.tabId, {
        type: "PP_START_SELECTION",
        // Only pass intentional override for this capture
        symbol: opts.symbol || "",
        timeframe: opts.timeframe || settings.defaultTimeframe || "",
        exchange: opts.exchange || settings.defaultExchange || "",
      });
    } catch (err) {
      throw new Error(
        "Could not start area selection on this page. Try another tab or use Capture & Analyze."
      );
    }
    return;
  } else if (opts.mode === "crop" && opts.rect) {
    const full = await captureVisible(opts.windowId);
    dataUrl = await cropDataUrl(full, opts.rect, opts.dpr || 1);
  } else {
    dataUrl = await captureVisible(opts.windowId);
  }

  let pageUrl = opts.pageUrl || "";
  if (!pageUrl && opts.tabId) {
    try {
      const t = await chrome.tabs.get(opts.tabId);
      pageUrl = t.url || "";
    } catch (_) {}
  }

  await prepareAndAnalyze(dataUrl, {
    settings,
    // Never fall back to settings.defaultSymbol (stale previous pair)
    symbol: opts.symbol || "",
    timeframe: opts.timeframe || settings.defaultTimeframe || "",
    exchange: opts.exchange || settings.defaultExchange || "",
    tabId: opts.tabId,
    windowId: opts.windowId,
    pageUrl,
    freshCapture: opts.freshCapture !== false,
  });
}

async function prepareAndAnalyze(dataUrl, ctx) {
  const pendingId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  await chrome.storage.local.set({
    [STORAGE_KEYS.pending]: {
      id: pendingId,
      dataUrl,
      // Only the override for this capture (may be empty)
      symbol: ctx.symbol || "",
      timeframe: ctx.timeframe || "",
      exchange: ctx.exchange || "",
      pageUrl: ctx.pageUrl || "",
      ts: Date.now(),
      claimedBy: null,
      freshCapture: true,
    },
    [STORAGE_KEYS.lastCapture]: {
      dataUrl,
      ts: Date.now(),
      symbol: ctx.symbol || "",
    },
  });

  await setStatus({
    phase: "ocr",
    message: "Image captured — OCR + vision + URL fusion…",
  });
  broadcast({
    type: "PENDING_IMAGE",
    pending: {
      id: pendingId,
      symbol: ctx.symbol || "",
      timeframe: ctx.timeframe || "",
      exchange: ctx.exchange || "",
      pageUrl: ctx.pageUrl || "",
      ts: Date.now(),
      freshCapture: true,
    },
  });

  if (ctx.windowId != null && chrome.sidePanel?.open) {
    try {
      await chrome.sidePanel.open({ windowId: ctx.windowId });
    } catch (_) {}
  }

  // Fallback: if UI never claims, SW analyzes with backend OCR only
  await sleep(4500);
  const claimed = await tryClaimPending("service_worker", pendingId);
  if (!claimed) return;

  await setStatus({ phase: "analyzing", message: "Sending to backend…" });
  broadcast({ type: "STATUS", phase: "analyzing", message: "Sending to backend…" });

  try {
    const result = await postAnalyze({
      dataUrl,
      symbol: ctx.symbol || "",
      timeframe: ctx.timeframe || "",
      exchange: ctx.exchange || "",
      settings: ctx.settings,
      pageUrl: ctx.pageUrl || "",
    });

    const packaged = {
      ...result,
      _meta: {
        capturedAt: new Date().toISOString(),
        clientOcr: null,
        pageUrl: ctx.pageUrl || "",
        apiBase: API_BASE,
        path: "service_worker",
        imageDataUrl: dataUrl,
        symbolUsed: ctx.symbol || result.symbol || "",
      },
    };

    await chrome.storage.local.set({
      [STORAGE_KEYS.lastResult]: packaged,
      [STORAGE_KEYS.lastCapture]: {
        dataUrl,
        ts: Date.now(),
        symbol: ctx.symbol || result.symbol || "",
      },
    });
    await chrome.storage.local.remove([STORAGE_KEYS.pending]);
    await setStatus({ phase: "done", message: "Analysis complete" });
    broadcast({ type: "RESULT", result: packaged });

    try {
      const bias = (result.bias || "neutral").toUpperCase();
      const conf =
        result.confidence != null ? `${Number(result.confidence).toFixed(0)}%` : "—";
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icons/icon128.png",
        title: `Perpetual Pro · ${bias}`,
        message: `${result.symbol || "Chart"} · Confidence ${conf}`,
      });
    } catch (_) {}
  } catch (err) {
    const friendly = friendlyBackendError(String(err?.message || err));
    await setStatus({ phase: "error", message: friendly });
    broadcast({ type: "STATUS", phase: "error", message: friendly });
    await chrome.storage.local.remove([STORAGE_KEYS.pending]);
  }
}

/** @returns {Promise<object|null>} pending payload if claim succeeded */
async function tryClaimPending(who, expectedId = null) {
  const data = await chrome.storage.local.get([STORAGE_KEYS.pending]);
  const pending = data[STORAGE_KEYS.pending];
  if (!pending?.dataUrl) return null;
  if (expectedId && pending.id && pending.id !== expectedId) return null;
  if (pending.claimedBy && pending.claimedBy !== who) return null;
  pending.claimedBy = who;
  await chrome.storage.local.set({ [STORAGE_KEYS.pending]: pending });
  return pending;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ---------------------------------------------------------------------------
// Capture helpers
// ---------------------------------------------------------------------------

async function captureVisible(windowId) {
  const wid = windowId ?? (await chrome.windows.getCurrent()).id;
  return chrome.tabs.captureVisibleTab(wid, { format: "png" });
}

async function ensureContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "PP_PING" });
  } catch {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content/select-area.js"],
    });
    await chrome.scripting.insertCSS({
      target: { tabId },
      files: ["content/select-area.css"],
    });
  }
}

async function fetchImageAsDataUrl(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not fetch image (${res.status})`);
  const blob = await res.blob();
  return blobToDataUrl(blob);
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function cropDataUrl(dataUrl, rect, dpr = 1) {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  const bitmap = await createImageBitmap(blob);
  const sx = Math.max(0, Math.round(rect.x * dpr));
  const sy = Math.max(0, Math.round(rect.y * dpr));
  const sw = Math.max(1, Math.round(rect.w * dpr));
  const sh = Math.max(1, Math.round(rect.h * dpr));
  const canvas = new OffscreenCanvas(sw, sh);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh);
  const out = await canvas.convertToBlob({ type: "image/png" });
  return blobToDataUrl(out);
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

function friendlyBackendError(msg) {
  if (/Failed to fetch|NetworkError|ECONNREFUSED|abort|unreachable|Load failed/i.test(msg)) {
    return "Cannot reach Perpetual Pro API (Render may be waking up — retry in ~30s).";
  }
  return msg;
}

async function scanSymbols(message) {
  const symbols = Array.isArray(message?.symbols) ? message.symbols : [];
  const form = new FormData();
  form.append("symbols", symbols.join(","));
  if (message?.timeframe) form.append("timeframe", message.timeframe);
  if (message?.exchange) form.append("exchange", message.exchange);
  if (message?.noNews) form.append("no_news", "true");
  const res = await fetch(`${API_BASE}/scan`, { method: "POST", body: form });
  let body;
  try {
    body = await res.json();
  } catch {
    throw new Error(`Backend error HTTP ${res.status}`);
  }
  if (!res.ok) {
    throw new Error(body?.error || body?.message || `HTTP ${res.status}`);
  }
  return body;
}

async function checkBackend() {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 15000);
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) {
      return { ok: false, error: `Backend returned HTTP ${res.status}` };
    }
    const data = await res.json();
    return { ok: true, health: data };
  } catch (err) {
    return {
      ok: false,
      error: friendlyBackendError(String(err?.message || err)),
      detail: String(err?.message || err),
    };
  }
}

async function postAnalyze({
  dataUrl,
  symbol,
  timeframe,
  exchange,
  settings,
  pageUrl = "",
  clientOcr = null,
  clientVision = null,
  clientHints = null,
}) {
  const blob = await (await fetch(dataUrl)).blob();
  const form = new FormData();
  form.append("image", blob, "chart.png");
  if (symbol) form.append("symbol", symbol);
  if (timeframe) form.append("timeframe", timeframe);
  if (exchange) form.append("exchange", exchange);
  if (pageUrl) form.append("page_url", pageUrl);
  if (clientOcr) form.append("client_ocr", JSON.stringify(clientOcr));
  if (clientVision) form.append("client_vision", JSON.stringify(clientVision));
  if (clientHints) form.append("client_hints", JSON.stringify(clientHints));
  if (settings?.higher) form.append("higher", settings.higher);

  // Hidden educational simulation only — never a user wallet
  form.append("simulated_capital", String(SIMULATED_CAPITAL_USD));
  form.append("risk", String(DEFAULT_RISK_PCT));

  if (settings?.noNews) form.append("no_news", "true");
  form.append("dark_theme", "true");

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  let body;
  try {
    body = await res.json();
  } catch {
    throw new Error(`Backend error HTTP ${res.status}`);
  }

  if (res.status === 422) {
    const msg =
      body?.message || body?.detail || "Could not detect symbol — enter ticker and retry";
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  if (!res.ok) {
    const detail = body?.detail || body?.message || `HTTP ${res.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

// ---------------------------------------------------------------------------
// Settings / status / broadcast
// ---------------------------------------------------------------------------

async function getSettings() {
  const data = await chrome.storage.local.get(STORAGE_KEYS.settings);
  const s = data[STORAGE_KEYS.settings] || {};
  // Strip legacy user-facing API URL / balance fields
  const {
    apiBase: _legacyApi,
    balance: _legacyBal,
    risk: _legacyRisk,
    ...rest
  } = s;
  return {
    defaultTimeframe: "",
    defaultExchange: "bybit",
    higher: "5m,1h,4h",
    noNews: false,
    darkTheme: true,
    captureMode: "visible",
    ...rest,
    // Force non-sticky symbol — never reuse previous pairs
    defaultSymbol: "",
  };
}

async function saveSettings(partial) {
  const cur = await getSettings();
  const merged = { ...cur, ...partial };
  // Never persist user-facing capital or API URL
  delete merged.apiBase;
  delete merged.balance;
  delete merged.risk;
  // Never stick a default symbol across captures
  merged.defaultSymbol = "";
  await chrome.storage.local.set({
    [STORAGE_KEYS.settings]: merged,
  });
}

async function setStatus(status) {
  await chrome.storage.local.set({
    [STORAGE_KEYS.status]: { ...status, ts: Date.now() },
  });
}

async function getStatus() {
  const data = await chrome.storage.local.get(STORAGE_KEYS.status);
  return data[STORAGE_KEYS.status] || { phase: "idle", message: "Ready" };
}

function broadcast(msg) {
  chrome.runtime.sendMessage(msg).catch(() => {});
}
