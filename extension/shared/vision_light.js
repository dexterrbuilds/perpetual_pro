/**
 * Lightweight client-side chart vision (no OpenCV.js bundle).
 * Detects approximate candle polarity, trend, and horizontal levels from a screenshot.
 * Full OpenCV analysis still runs on the backend.
 */

/**
 * @param {string} dataUrl
 * @returns {Promise<{
 *   candles_approx: number,
 *   bullish_ratio: number,
 *   trend_guess: string,
 *   support_ys: number[],
 *   resistance_ys: number[],
 *   chart_region: object|null,
 *   confidence: number,
 *   notes: string[]
 * }>}
 */
export async function lightVision(dataUrl) {
  const notes = [];
  try {
    const img = await loadImage(dataUrl);
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    if (w < 40 || h < 40) {
      return emptyResult("image too small");
    }

    // Sample chart region: drop top chrome (~12%) and bottom time axis (~15%)
    const x0 = Math.floor(w * 0.04);
    const x1 = Math.floor(w * 0.92);
    const y0 = Math.floor(h * 0.1);
    const y1 = Math.floor(h * 0.82);
    const cw = x1 - x0;
    const ch = y1 - y0;

    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, x0, y0, cw, ch, 0, 0, cw, ch);
    const { data } = ctx.getImageData(0, 0, cw, ch);

    // Column-wise green vs red dominance (TradingView style candles)
    const cols = Math.min(80, Math.floor(cw / 4));
    const colW = Math.floor(cw / cols);
    let bulls = 0;
    let bears = 0;
    const closeYs = [];

    for (let c = 0; c < cols; c++) {
      let g = 0;
      let r = 0;
      let massY = 0;
      let mass = 0;
      const xs = c * colW;
      for (let x = xs; x < xs + colW && x < cw; x += 2) {
        for (let y = 0; y < ch; y += 2) {
          const i = (y * cw + x) * 4;
          const R = data[i];
          const G = data[i + 1];
          const B = data[i + 2];
          // green candle-ish
          if (G > R + 25 && G > B + 10 && G > 60) {
            g++;
            massY += y;
            mass++;
          }
          // red candle-ish
          if (R > G + 25 && R > B + 10 && R > 60) {
            r++;
            massY += y;
            mass++;
          }
        }
      }
      if (g + r < 8) continue;
      if (g > r) bulls++;
      else bears++;
      if (mass > 0) closeYs.push(massY / mass);
    }

    const total = bulls + bears;
    const bullish_ratio = total ? bulls / total : 0.5;
    let trend_guess = "range";
    if (closeYs.length >= 6) {
      const first = avg(closeYs.slice(0, Math.floor(closeYs.length / 3)));
      const last = avg(closeYs.slice(-Math.floor(closeYs.length / 3)));
      // y increases downward → lower y = higher price
      if (first - last > ch * 0.04) trend_guess = "up";
      else if (last - first > ch * 0.04) trend_guess = "down";
      else if (bullish_ratio > 0.58) trend_guess = "up";
      else if (bullish_ratio < 0.42) trend_guess = "down";
    }

    // Horizontal level approx: histogram of "edge-like" rows
    const rowEnergy = new Float64Array(ch);
    for (let y = 1; y < ch - 1; y++) {
      let e = 0;
      for (let x = 0; x < cw; x += 3) {
        const i = (y * cw + x) * 4;
        const i2 = ((y - 1) * cw + x) * 4;
        const lum = data[i] * 0.3 + data[i + 1] * 0.59 + data[i + 2] * 0.11;
        const lum2 = data[i2] * 0.3 + data[i2 + 1] * 0.59 + data[i2 + 2] * 0.11;
        e += Math.abs(lum - lum2);
      }
      rowEnergy[y] = e;
    }
    const peaks = findPeaks(rowEnergy, 6);
    const mid = ch / 2;
    const support_ys = peaks.filter((y) => y > mid).slice(0, 4).map((y) => y + y0);
    const resistance_ys = peaks.filter((y) => y <= mid).slice(0, 4).map((y) => y + y0);

    const candles_approx = total;
    const confidence = Math.min(
      0.85,
      0.2 + Math.min(0.4, total / 60) + (trend_guess !== "range" ? 0.15 : 0) + (peaks.length ? 0.1 : 0)
    );
    notes.push(`cols_scanned=${cols} bull=${bulls} bear=${bears}`);
    notes.push(`trend≈${trend_guess}`);
    if (peaks.length) notes.push(`levels≈${peaks.length}`);

    return {
      candles_approx,
      bullish_ratio: round4(bullish_ratio),
      trend_guess,
      support_ys,
      resistance_ys,
      chart_region: { x: x0, y: y0, w: cw, h: ch },
      confidence: round4(confidence),
      notes,
    };
  } catch (err) {
    return emptyResult(String(err?.message || err));
  }
}

function emptyResult(msg) {
  return {
    candles_approx: 0,
    bullish_ratio: 0.5,
    trend_guess: "unknown",
    support_ys: [],
    resistance_ys: [],
    chart_region: null,
    confidence: 0,
    notes: [msg],
  };
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image load failed"));
    img.src = dataUrl;
  });
}

function avg(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function findPeaks(energy, n) {
  const len = energy.length;
  const mean = avg(Array.from(energy));
  const thr = mean * 1.35;
  const peaks = [];
  for (let y = 2; y < len - 2; y++) {
    if (energy[y] > thr && energy[y] >= energy[y - 1] && energy[y] >= energy[y + 1]) {
      peaks.push({ y, e: energy[y] });
    }
  }
  peaks.sort((a, b) => b.e - a.e);
  // Cluster close peaks
  const out = [];
  for (const p of peaks) {
    if (out.some((y) => Math.abs(y - p.y) < 8)) continue;
    out.push(p.y);
    if (out.length >= n) break;
  }
  return out.sort((a, b) => a - b);
}

function round4(x) {
  return Math.round(Number(x) * 10000) / 10000;
}
