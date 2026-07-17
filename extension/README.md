# Perpetual Pro — Chrome Extension (Manifest V3)

Capture TradingView / exchange charts and run **pro perpetual futures analysis** against the local FastAPI backend at `http://localhost:8000`.

## Features

- **Popup** button: *Capture & Analyze Chart*
- **Keyboard shortcut**: `Ctrl+Shift+P` (select area on the active tab)
- **Right-click** on any image or page → analyze / select area
- Capture **visible tab** or **drag-select a region**
- **Light OCR** with **bundled** Tesseract.js under `lib/` (no CDN; symbol / timeframe hints)
- Full analysis via backend `POST /analyze` (OCR + data + indicators + patterns + news + confluence)
- **Side panel** results: bias, confidence, levels, confluence, patterns, news, risk
- Loading states + clear errors (*Backend not running…*)
- Dark-mode UI

## Prerequisites

1. Start the Python API from the `perpetual_pro` project root:

```bash
cd /path/to/perpetual_pro
source .venv/bin/activate
uvicorn main_server:app --reload --port 8000
```

2. Confirm health: open [http://localhost:8000/health](http://localhost:8000/health)

## Load unpacked in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select this folder: `perpetual_pro/extension`
5. Pin **Perpetual Pro** to the toolbar

## Usage

| Action | Result |
|--------|--------|
| Popup → **Capture & Analyze Chart** | Drag-select chart region → OCR → `/analyze` |
| Popup → **Visible tab** | Full viewport capture |
| `Ctrl+Shift+P` | Select area on current tab |
| Right-click image → **Analyze with Perpetual Pro** | Uses that image URL |
| Right-click page → **Select chart area…** | Drag selection |
| **Side panel** | Full-width report view |

Optional **Symbol** / **Timeframe** in popup settings override OCR. If OCR misses the symbol, set it manually (e.g. `BTC`).

## Permissions

- `activeTab` / `tabs` / `scripting` — capture & inject selection UI  
- `contextMenus` — right-click actions  
- `sidePanel` — results panel  
- `storage` — settings + last report  
- Host access to `http://localhost:8000` only (OCR is fully offline)

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
| “Backend not running” | Start uvicorn on port 8000; check firewall |
| Capture fails on Chrome Web Store / chrome:// pages | Use a normal https page (TradingView, etc.) |
| OCR weak | Set Symbol manually; backend still runs full OCR |
| Shortcut conflict | `chrome://extensions/shortcuts` |
| CORS | Not needed for extension → localhost with host_permissions |

## Disclaimer

Not financial advice. Educational / research tooling only.
