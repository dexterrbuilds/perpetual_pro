/**
 * Light client-side OCR via locally bundled Tesseract.js + rich chart text parsing.
 * Backend also runs Tesseract + EasyOCR for maximum recall.
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
    gzip: true,
    cacheMethod: "none",
    logger: () => {},
  };
}

async function getWorker() {
  if (_workerPromise) return _workerPromise;
  _workerPromise = (async () => {
    const Tesseract = await ensureTesseract();
    const paths = localPaths();
    return Tesseract.createWorker("eng", 1, {
      workerPath: paths.workerPath,
      corePath: paths.corePath,
      langPath: paths.langPath,
      cacheMethod: "none",
      logger: () => {},
    });
  })();
  try {
    return await _workerPromise;
  } catch (err) {
    _workerPromise = null;
    throw err;
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
  for (const p of ["15M", "5M", "1H", "4H", "1D", "30M", "1M", "3M", "45M", "2H", "6H", "12H", "3D", "1W"]) {
    if (tfs.includes(p)) {
      timeframe = map[p];
      break;
    }
  }

  const blacklist = new Set([
    "USD", "USDT", "USDC", "PERP", "SPOT", "LONG", "SHORT", "BUY", "SELL", "OPEN",
    "HIGH", "LOW", "CLOSE", "VOLUME", "PRICE", "CHART", "TIME", "BINANCE", "BYBIT",
    "OKX", "BITGET", "UTC", "GMT", "CROSS", "ISOLATED", "MARKET", "LIMIT", "RSI",
    "MACD", "EMA", "SMA", "ATR", "VWAP", "THE", "AND", "FOR", "ROE", "PNL", "AVG",
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
        symbol = base.endsWith("USDT") || base.endsWith("USDC") ? base : `${base}USDT`;
        break;
      }
    }
  }

  // Prices
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
    "RSI", "MACD", "EMA", "SMA", "VWAP", "ATR", "BB", "BOLL", "STOCH",
    "VOLUME", "OI", "FUNDING", "ICHIMOKU", "CVD", "OBV", "SUPERTREND", "ADX",
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

  // Vision cannot give symbol but can reinforce direction
  if (vision?.trend_guess && vision.trend_guess !== "unknown") {
    notes.push(`client vision trend≈${vision.trend_guess}`);
  }

  return {
    symbol,
    timeframe,
    exchange,
    source,
    notes,
    canAnalyzeWithoutSymbol: false, // backend may still use vision-only path if we force a default — we don't
  };
}
