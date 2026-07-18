/**
 * Light client-side OCR via locally bundled Tesseract.js (Manifest V3 safe).
 *
 * Critical MV3 rules:
 * - All assets live under extension/lib/ (no CDN).
 * - Paths use chrome.runtime.getURL(...).
 * - workerBlobURL MUST be false so the worker is a real chrome-extension://
 *   Worker (not a blob). Blob workers cannot importScripts() extension URLs.
 * - Single reusable worker instance for the popup/sidepanel lifetime.
 */

const LIB = {
  api: "lib/tesseract.min.js",
  worker: "lib/worker.min.js",
  /** Directory containing tesseract-core-*.wasm.js (+ sibling .wasm files) */
  coreDir: "lib",
  /** Prefer explicit SIMD core file for deterministic loads */
  coreFile: "lib/tesseract-core-simd-lstm.wasm.js",
  lang: "lib/lang-data",
};

let _tessReady = null;
/** @type {Promise<any> | null} */
let _workerPromise = null;
/** @type {any | null} */
let _workerInstance = null;

function libUrl(rel) {
  // Strip leading ./ and ensure no double slashes in join
  const clean = String(rel).replace(/^\.\//, "");
  return chrome.runtime.getURL(clean);
}

/**
 * Tesseract.js createWorker options that work under MV3 extension pages.
 */
export function getTesseractWorkerOptions(extra = {}) {
  return {
    // Load worker as chrome-extension://…/worker.min.js (NOT a blob)
    workerPath: libUrl(LIB.worker),
    // Explicit core glue script (resolves sibling .wasm next to it)
    corePath: libUrl(LIB.coreFile),
    // Directory: fetches eng.traineddata.gz from here
    langPath: libUrl(LIB.lang),
    // REQUIRED for MV3: false → Worker(chrome-extension URL)
    // true → blob Worker → importScripts(chrome-extension://…) fails
    workerBlobURL: false,
    gzip: true,
    cacheMethod: "none",
    // Avoid legacy path probes that hit network
    legacyCore: false,
    legacyLang: false,
    logger: () => {},
    ...extra,
  };
}

/**
 * Load the UMD build (window.Tesseract). Prefer preloaded <script> in HTML.
 */
export async function ensureTesseract() {
  if (typeof window !== "undefined" && window.Tesseract) {
    return window.Tesseract;
  }
  if (_tessReady) return _tessReady;

  _tessReady = new Promise((resolve, reject) => {
    if (typeof window !== "undefined" && window.Tesseract) {
      resolve(window.Tesseract);
      return;
    }

    const existing = document.querySelector('script[data-pp-tesseract="1"]');
    if (existing) {
      const done = () => {
        if (window.Tesseract) resolve(window.Tesseract);
        else reject(new Error("Tesseract global missing after load"));
      };
      if (window.Tesseract) {
        done();
        return;
      }
      existing.addEventListener("load", done);
      existing.addEventListener("error", () =>
        reject(new Error("Failed to load local Tesseract.js"))
      );
      // If script already completed before listener
      setTimeout(() => {
        if (window.Tesseract) resolve(window.Tesseract);
      }, 50);
      return;
    }

    const s = document.createElement("script");
    s.src = libUrl(LIB.api);
    s.async = false;
    s.dataset.ppTesseract = "1";
    s.onload = () => {
      if (window.Tesseract) resolve(window.Tesseract);
      else reject(new Error("Tesseract global missing after load"));
    };
    s.onerror = () =>
      reject(
        new Error(
          `Failed to load ${LIB.api} via ${libUrl(LIB.api)}. Check extension packaging.`
        )
      );
    (document.head || document.documentElement).appendChild(s);
  });

  return _tessReady;
}

/**
 * Single shared worker — create once, reuse for all lightOcr() calls.
 */
export async function getWorker() {
  if (_workerInstance) return _workerInstance;
  if (_workerPromise) return _workerPromise;

  _workerPromise = (async () => {
    const Tesseract = await ensureTesseract();
    if (!Tesseract?.createWorker) {
      throw new Error("Tesseract.createWorker unavailable");
    }

    const opts = getTesseractWorkerOptions();
    // createWorker(langs, oem, options)
    // oem 1 = LSTM only
    const worker = await Tesseract.createWorker("eng", 1, opts);

    // loadLanguage/initialize are handled by createWorker('eng') in v5,
    // but re-assert in case of partial init
    if (typeof worker.reinitialize === "function") {
      // already initialized with eng
    }

    _workerInstance = worker;
    return worker;
  })();

  try {
    return await _workerPromise;
  } catch (err) {
    _workerPromise = null;
    _workerInstance = null;
    throw err;
  }
}

/**
 * Terminate the singleton worker (call on popup unload if desired).
 */
export async function terminateWorker() {
  const w = _workerInstance;
  _workerInstance = null;
  _workerPromise = null;
  if (w && typeof w.terminate === "function") {
    try {
      await w.terminate();
    } catch (_) {
      /* ignore */
    }
  }
}

/**
 * Full light OCR pass — extract all text + structured fields.
 * @param {string} dataUrl
 */
export async function lightOcr(dataUrl) {
  try {
    let raw = "";
    let conf = 0;

    // Prefer single long-lived worker (MV3-safe paths)
    try {
      const worker = await getWorker();
      const result = await worker.recognize(dataUrl);
      raw = (result?.data?.text || "").trim();
      conf = Number(result?.data?.confidence || 0) / 100;
    } catch (workerErr) {
      console.warn(
        "[Perpetual Pro] createWorker path failed, trying Tesseract.recognize():",
        workerErr
      );
      // Second path: still MV3-safe options, no blob worker
      const Tesseract = await ensureTesseract();
      const opts = getTesseractWorkerOptions();
      const result = await Tesseract.recognize(dataUrl, "eng", opts);
      raw = (result?.data?.text || "").trim();
      conf = Number(result?.data?.confidence || 0) / 100;
      // recognize() may spin its own worker; drop our singleton if tainted
      _workerPromise = null;
      _workerInstance = null;
    }

    const parsed = parseChartText(raw);
    return {
      ...parsed,
      raw: raw.slice(0, 2000),
      all_text: raw.slice(0, 4000),
      confidence: conf,
      engine: "tesseract.js",
    };
  } catch (err) {
    console.warn("[Perpetual Pro] Light OCR skipped:", err);
    _workerPromise = null;
    _workerInstance = null;
    return {
      symbol: "",
      timeframe: "",
      prices: [],
      indicators: [],
      raw: "",
      all_text: "",
      confidence: 0,
      engine: "none",
      error: String(err?.message || err),
    };
  }
}

export function parseChartText(text) {
  const upper = (text || "").toUpperCase();
  let timeframe = "";
  const tfRe = /\b(1M|3M|5M|15M|30M|45M|1H|2H|4H|6H|12H|1D|3D|1W)\b/g;
  const tfs = upper.match(tfRe) || [];
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
      timeframe = map[p];
      break;
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
    "AVG",
  ]);

  let symbol = "";
  const patterns = [
    /\b([A-Z]{2,10})[\/\-_]?USDT\.?P?\b/,
    /\b([A-Z]{2,10})[\/\-_]?USDC\b/,
    /\b(1000[A-Z]{2,8})USDT\b/,
    /\b(BTC|ETH|SOL|XRP|DOGE|BNB|ADA|AVAX|LINK|DOT|PEPE|WIF|SUI|ARB|OP|TIA|SEI|NEAR|APT|INJ)\b/,
  ];
  for (const re of patterns) {
    const m = upper.match(re);
    if (m) {
      const base = m[1];
      if (base && !blacklist.has(base)) {
        symbol =
          base.endsWith("USDT") || base.endsWith("USDC") ? base : `${base}USDT`;
        break;
      }
    }
  }

  const prices = [];
  const priceRe = /\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d{2,8}\b/g;
  let pm;
  while ((pm = priceRe.exec(text || "")) !== null) {
    const v = parseFloat(pm[0].replace(/,/g, ""));
    if (!Number.isFinite(v) || v <= 0) continue;
    if (v > 1900 && v < 2100 && Number.isInteger(v)) continue;
    prices.push(v);
  }

  const indicators = [];
  const indList = [
    "RSI",
    "MACD",
    "EMA",
    "SMA",
    "VWAP",
    "ATR",
    "BB",
    "BOLL",
    "STOCH",
    "VOLUME",
    "OI",
    "FUNDING",
    "ICHIMOKU",
    "CVD",
    "OBV",
    "SUPERTREND",
    "ADX",
  ];
  for (const kw of indList) {
    if (upper.includes(kw)) indicators.push(kw);
  }

  return {
    symbol,
    timeframe,
    prices: [...new Set(prices.map((p) => Math.round(p * 1e8) / 1e8))].slice(0, 40),
    indicators,
  };
}

/**
 * Fuse user / URL / OCR / vision hints into best symbol + timeframe.
 */
export function fuseHints({ userSymbol, userTf, urlHints, ocr, vision }) {
  const notes = [];
  let symbol = (userSymbol || "").trim();
  let timeframe = (userTf || "").trim();
  let exchange = "";
  let source = "user";

  if (!symbol && urlHints?.symbol && urlHints.confidence >= 0.5) {
    symbol = urlHints.symbol;
    source = urlHints.source || "url";
    notes.push(`symbol from URL (${source}, conf ${urlHints.confidence})`);
  }
  if (!symbol && ocr?.symbol) {
    symbol = ocr.symbol;
    source = "ocr";
    notes.push("symbol from client OCR");
  }

  if (!timeframe && urlHints?.timeframe) {
    timeframe = urlHints.timeframe;
    notes.push("timeframe from URL");
  }
  if (!timeframe && ocr?.timeframe) {
    timeframe = ocr.timeframe;
    notes.push("timeframe from OCR");
  }

  if (urlHints?.exchange) exchange = urlHints.exchange;

  if (vision?.trend_guess && vision.trend_guess !== "unknown") {
    notes.push(`client vision trend≈${vision.trend_guess}`);
  }

  return {
    symbol,
    timeframe,
    exchange,
    source,
    notes,
    canAnalyzeWithoutSymbol: false,
  };
}
