/**
 * Extract symbol / timeframe / exchange hints from the active tab URL.
 * Optimized for TradingView; also Binance / Bybit / OKX / Bitget.
 */

const TV_INTERVAL = {
  "1": "1m",
  "3": "3m",
  "5": "5m",
  "15": "15m",
  "30": "30m",
  "45": "45m",
  "60": "1h",
  "120": "2h",
  "180": "3h",
  "240": "4h",
  D: "1d",
  "1D": "1d",
  W: "1w",
  "1W": "1w",
  M: "1M",
  "1M": "1M",
};

/**
 * @param {string} url
 * @returns {{symbol:string,timeframe:string,exchange:string,source:string,raw:string,confidence:number}}
 */
export function parseChartUrl(url) {
  const empty = {
    symbol: "",
    timeframe: "",
    exchange: "",
    source: "",
    raw: "",
    confidence: 0,
  };
  if (!url || typeof url !== "string") return empty;

  let u;
  try {
    u = new URL(url);
  } catch {
    return empty;
  }

  const host = u.hostname.toLowerCase();
  const path = decodeURIComponent(u.pathname || "");

  if (host.includes("tradingview.")) {
    return parseTradingView(u, path);
  }
  if (host.includes("binance.")) {
    const m = path.match(/\/(?:futures|trade)\/(?:[\w-]+\/)?([A-Za-z0-9]+)(USDT|USDC|BUSD)/i);
    if (m) {
      const pair = `${m[1].toUpperCase()}${m[2].toUpperCase()}`;
      return {
        symbol: pair,
        timeframe: "",
        exchange: path.toLowerCase().includes("futures") ? "binanceusdm" : "",
        source: "binance_url",
        raw: pair,
        confidence: 0.85,
      };
    }
  }
  if (host.includes("bybit.")) {
    const m = path.match(/\/trade\/(?:usdt\/)?([A-Za-z0-9]+)/i);
    if (m) {
      let base = m[1].toUpperCase();
      if (!base.endsWith("USDT")) base = `${base}USDT`;
      return {
        symbol: base,
        timeframe: "",
        exchange: "bybit",
        source: "bybit_url",
        raw: base,
        confidence: 0.8,
      };
    }
  }
  if (host.includes("okx.")) {
    const m = path.match(/([A-Za-z0-9]+)-USDT/i);
    if (m) {
      const pair = `${m[1].toUpperCase()}USDT`;
      return {
        symbol: pair,
        timeframe: "",
        exchange: "okx",
        source: "okx_url",
        raw: pair,
        confidence: 0.8,
      };
    }
  }
  if (host.includes("bitget.")) {
    const m = path.match(/([A-Za-z0-9]+USDT)/i);
    if (m) {
      return {
        symbol: m[1].toUpperCase(),
        timeframe: "",
        exchange: "bitget",
        source: "bitget_url",
        raw: m[1].toUpperCase(),
        confidence: 0.75,
      };
    }
  }

  const gen = path.toUpperCase().match(/\b([A-Z]{2,12})[-_]?USDT\b/);
  if (gen) {
    return {
      symbol: `${gen[1]}USDT`,
      timeframe: "",
      exchange: "",
      source: "generic_url",
      raw: gen[0],
      confidence: 0.55,
    };
  }
  return empty;
}

function parseTradingView(u, path) {
  let raw = "";
  let confidence = 0;
  const symQ = u.searchParams.get("symbol");
  if (symQ) {
    raw = decodeURIComponent(symQ);
    confidence = 0.95;
  }
  if (!raw) {
    const m = path.match(/\/symbols\/([^/]+)/i);
    if (m) {
      raw = m[1];
      confidence = 0.9;
    }
  }
  if (!raw) {
    const m = path.match(/\/chart\/([A-Za-z0-9._:-]{3,40})\/?/i);
    if (m && !/^[a-f0-9]{8,}$/i.test(m[1])) {
      raw = m[1];
      confidence = 0.55;
    }
  }
  if (!raw) {
    const m = u.href.toUpperCase().match(/([A-Z]{2,12})USDT(?:\.P)?/);
    if (m) {
      raw = m[0];
      confidence = 0.5;
    }
  }

  let exchange = "";
  let pair = raw.toUpperCase().replace(/\s+/g, "");
  if (pair.includes(":")) {
    const [ex, rest] = pair.split(":");
    pair = rest;
    const map = {
      BINANCE: "binanceusdm",
      BINANCEUSDM: "binanceusdm",
      BYBIT: "bybit",
      OKX: "okx",
      BITGET: "bitget",
    };
    exchange = map[ex] || "";
  }
  pair = pair.replace(/\.P$/i, "").replace(/PERP/gi, "").replace(/[-_]/g, "");
  if (pair && !pair.endsWith("USDT") && !pair.endsWith("USDC")) {
    // leave as-is; backend normalize_symbol handles bare base
  }

  let timeframe = "";
  const iv = u.searchParams.get("interval") || u.searchParams.get("i");
  if (iv) {
    timeframe = TV_INTERVAL[iv] || TV_INTERVAL[iv.toUpperCase()] || "";
    if (!timeframe && /^\d+[mhdw]$/i.test(iv)) timeframe = iv.toLowerCase();
  }

  return {
    symbol: pair,
    timeframe,
    exchange,
    source: "tradingview",
    raw,
    confidence: pair ? confidence : 0,
  };
}
