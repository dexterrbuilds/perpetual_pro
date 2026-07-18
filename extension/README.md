# Perpetual Pro — Chrome Extension (Manifest V3)

Capture TradingView / exchange charts and run **pro perpetual futures analysis** against the fixed production API:

**`https://perpetual-pro.onrender.com`**

(No user setting for backend URL — all extension calls use this host.)

## Features

- **Popup** button: *Capture & Analyze Chart*
- **Keyboard shortcut**: `Ctrl+Shift+P` (select area on the active tab)
- **Right-click** on any image or page → analyze / select area
- **Auto full-tab capture** via `chrome.tabs.captureVisibleTab` (default)
- Optional **drag-select** region remains available
- **Client OCR** (bundled Tesseract.js) + **light vision** (candles/trend/levels)
- **TradingView URL** pair extraction as strong symbol fallback
- Fuses **OCR + URL + vision** before calling backend
- Backend re-runs dual OCR (Tesseract multi-PSM + EasyOCR) + OpenCV chart CV
- **Side panel** results: bias, confidence, levels, confluence, patterns, news, risk
- Loading states + clear errors (API wake-up / network)
- Dark-mode UI

## Prerequisites

1. API is the Render deployment: [https://perpetual-pro.onrender.com/health](https://perpetual-pro.onrender.com/health)
2. Free-tier Render apps may cold-start (~30s) on first request — retry if health fails once

## Load unpacked in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select this folder: `perpetual_pro/extension`
5. Pin **Perpetual Pro** to the toolbar

## Usage

| Action | Result |
|--------|--------|
| Popup → **Capture & Analyze Chart** | Full visible tab → OCR + vision + URL → `/analyze` |
| Popup → **Select area** | Optional drag-select chart region |
| `Ctrl+Shift+P` | Full visible tab capture + analyze |
| Right-click image → **Analyze with Perpetual Pro** | Uses that image URL |
| Right-click page → **Select chart area…** | Drag selection |
| **Side panel** | Full-width report view |

Optional **Symbol** / **Timeframe** in popup settings override OCR. If OCR misses the symbol, set it manually (e.g. `BTC`).

## Permissions

- `activeTab` / `tabs` / `scripting` — capture & inject selection UI  
- `contextMenus` — right-click actions  
- `sidePanel` — results panel  
- `storage` — settings + last report  
- Host access to `https://perpetual-pro.onrender.com` only (OCR is fully offline)

## File map

```
extension/
├── manifest.json
├── background.js          # service worker
├── popup.html / popup.js
├── sidepanel.html / sidepanel.js
├── content/select-area.js + .css
├── shared/ocr.js, render.js, styles.css
├── lib/                   # vendored Tesseract.js + eng.traineddata.gz
│   ├── tesseract.min.js
│   ├── worker.min.js
│   ├── tesseract-core-simd-lstm.wasm.js
│   ├── tesseract-core-simd-lstm.wasm
│   └── lang-data/eng.traineddata.gz
├── icons/
└── README.md
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| “Cannot reach … API” | Render free tier may be cold-starting — wait ~30s and retry; check https://perpetual-pro.onrender.com/health |
| Capture fails on Chrome Web Store / chrome:// pages | Use a normal https page (TradingView, etc.) |
| OCR weak | Set Symbol manually; backend still runs full OCR |
| Shortcut conflict | `chrome://extensions/shortcuts` |

## Disclaimer

Not financial advice. Educational / research tooling only.
