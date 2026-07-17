/**
 * Light client-side OCR via locally bundled Tesseract.js + symbol/timeframe heuristics.
 * All assets live under extension/lib/ — no CDN / network required for OCR.
 */

const LIB = {
  api: "lib/tesseract.min.js",
  worker: "lib/worker.min.js",
  core: "lib/tesseract-core-simd-lstm.wasm.js",
  lang: "lib/lang-data",
};

let _tessReady = null;
let _workerPromise = null;

function libUrl(rel) {
  return chrome.runtime.getURL(rel);
}

/**
 * Load the UMD build into the page (exposes window.Tesseract).
 */
export async function ensureTesseract() {
  if (window.Tesseract) return window.Tesseract;
  if (_tessReady) return _tessReady;

  _tessReady = new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-pp-tesseract="1"]');
    if (existing) {
      existing.addEventListener("load", () => {
        if (window.Tesseract) resolve(window.Tesseract);
        else reject(new Error("Tesseract global missing after load"));
      });
      existing.addEventListener("error", () =>
        reject(new Error("Failed to load local Tesseract.js"))
      );
      return;
    }
    const s = document.createElement("script");
    s.src = libUrl(LIB.api);
    s.async = true;
    s.dataset.ppTesseract = "1";
    s.onload = () => {
      if (window.Tesseract) resolve(window.Tesseract);
      else reject(new Error("Tesseract global missing after load"));
    };
    s.onerror = () => reject(new Error("Failed to load local lib/tesseract.min.js"));
    document.head.appendChild(s);
  });

  return _tessReady;
}

function localPaths() {
  return {
    workerPath: libUrl(LIB.worker),
    corePath: libUrl(LIB.core),
    langPath: libUrl(LIB.lang),
    // Avoid gzip worker dependency issues on some Chromium builds: prefer .gz which is shipped
    gzip: true,
    cacheMethod: "none",
    logger: () => {},
  };
}

/**
 * Reuse a single worker across OCR calls for speed.
 */
async function getWorker() {
  if (_workerPromise) return _workerPromise;
  _workerPromise = (async () => {
    const Tesseract = await ensureTesseract();
    const paths = localPaths();
    const worker = await Tesseract.createWorker("eng", 1, {
      workerPath: paths.workerPath,
      corePath: paths.corePath,
      langPath: paths.langPath,
      cacheMethod: "none",
      logger: () => {},
      // workerBlobURL: true is default; works with local workerPath
    });
    return worker;
  })();
  try {
    return await _workerPromise;
  } catch (err) {
    _workerPromise = null;
    throw err;
  }
}

/**
 * @param {string} dataUrl
 * @returns {Promise<{symbol:string,timeframe:string,raw:string,confidence:number,error?:string}>}
 */
export async function lightOcr(dataUrl) {
  try {
    // Prefer createWorker (explicit local paths). Fall back to recognize() with paths.
    let raw = "";
    let conf = 0;
    try {
      const worker = await getWorker();
      const result = await worker.recognize(dataUrl);
      raw = (result?.data?.text || "").trim();
      conf = Number(result?.data?.confidence || 0) / 100;
    } catch (workerErr) {
      console.warn("[Perpetual Pro] Worker OCR failed, trying recognize():", workerErr);
      const Tesseract = await ensureTesseract();
      const paths = localPaths();
      const result = await Tesseract.recognize(dataUrl, "eng", paths);
      raw = (result?.data?.text || "").trim();
      conf = Number(result?.data?.confidence || 0) / 100;
    }
    const parsed = parseChartText(raw);
    return { ...parsed, raw: raw.slice(0, 400), confidence: conf };
  } catch (err) {
    console.warn("[Perpetual Pro] Light OCR skipped:", err);
    _workerPromise = null;
    return {
      symbol: "",
      timeframe: "",
      raw: "",
      confidence: 0,
      error: String(err?.message || err),
    };
  }
}

export function parseChartText(text) {
  const upper = (text || "").toUpperCase();
  let timeframe = "";
  const tfRe = /\b(1M|3M|5M|15M|30M|45M|1H|2H|4H|6H|12H|1D|3D|1W)\b/g;
  const tfs = upper.match(tfRe);
  if (tfs && tfs.length) {
    const map = {
      "1M": "1m",
      "3M": "3m",
      "5M": "5m",
      "15M": "15m",
      "30M": "30m",
      "45M": "45m",
      "1H": "1h",
      "2H": "2h",
      "4H": "4h",
      "6H": "6h",
      "12H": "12h",
      "1D": "1d",
      "3D": "3d",
      "1W": "1w",
    };
    for (const p of [
      "15M",
      "5M",
      "1H",
      "4H",
      "1D",
      "30M",
      "1M",
      "3M",
      "45M",
      "2H",
      "6H",
      "12H",
      "3D",
      "1W",
    ]) {
      if (tfs.includes(p)) {
        timeframe = map[p] || p.toLowerCase();
        break;
      }
    }
  }

  const blacklist = new Set([
    "USD",
    "USDT",
    "USDC",
    "PERP",
    "SPOT",
    "LONG",
    "SHORT",
    "BUY",
    "SELL",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME",
    "PRICE",
    "CHART",
    "TIME",
    "BINANCE",
    "BYBIT",
    "OKX",
    "BITGET",
    "UTC",
    "GMT",
    "CROSS",
    "ISOLATED",
    "MARKET",
    "LIMIT",
    "RSI",
    "MACD",
    "EMA",
    "SMA",
    "ATR",
    "VWAP",
    "THE",
    "AND",
    "FOR",
    "ROE",
    "PNL",
  ]);

  let symbol = "";
  const patterns = [
    /\b([A-Z]{2,10})[\/\-_]?USDT\b/,
    /\b([A-Z]{2,10})[\/\-_]?USD\b/,
    /\b(1000[A-Z]{2,8})USDT\b/,
    /\b(BTC|ETH|SOL|XRP|DOGE|BNB|ADA|AVAX|LINK|DOT|PEPE|WIF|SUI|ARB|OP|TIA|SEI|NEAR|APT|INJ)\b/,
  ];
  for (const re of patterns) {
    const m = upper.match(re);
    if (m) {
      const base = m[1];
      if (base && !blacklist.has(base)) {
        symbol = base.endsWith("USDT") ? base : `${base}USDT`;
        break;
      }
    }
  }

  return { symbol, timeframe };
}
