# Perpetual Pro — Chrome Extension (Manifest V3)

Capture TradingView / exchange charts and run **pro perpetual futures analysis** against the fixed production API:

**`https://perpetual-pro.onrender.com`**

The API URL is hard-coded — there is **no** user setting for the backend.

## Features

- **Popup** · Capture & Analyze (full visible tab)
- **Keyboard** · `Ctrl+Shift+P`
- **Right-click** · image or page → analyze / select area
- **Select area** · optional drag-select on the chart
- **Client OCR** (bundled Tesseract.js) + light vision + TradingView URL fusion
- **Fresh symbol every capture** · never reuses a previous pair; OCR + URL + optional one-shot override only
- **Side panel** · full scannable report
- **Simulation example** · educational `$100` illustration only (not a wallet); dynamic leverage from the model
- **Suggested hold** · e.g. “4–24 hours” / “Swing – 2–7 days” from timeframe & setup
- Dark, responsive popup UI · clear / refresh reset state cleanly

## Load unpacked

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. **Load unpacked** → select `perpetual_pro/extension`
4. Pin **Perpetual Pro**

## Usage

| Action | Result |
|--------|--------|
| **Capture & Analyze** | Full tab → OCR + vision + URL → `/analyze` |
| **Select area** | Drag-select chart region |
| `Ctrl+Shift+P` | Full tab capture + analyze |
| Right-click image | Analyze that image |
| **Refresh** | Backend health + last report + re-show capture |
| **Clear** | Wipe symbol fields, capture, results, pending state |
| **Side panel** | Full-width report |

Optional **Symbol override** in settings applies **only to the next capture**. Leave blank so every chart resolves from OCR + page URL.

## Simulation (educational)

The extension always requests analysis with a **hidden** simulated capital of **$100** for illustration. The report shows:

> Simulation Example (for illustration only)  
> If you trade this signal with $100 at the suggested Nx leverage:  
> • At TP1 → +$…  
> • At TP2 → +$…

Suggested leverage comes from the backend model (ATR volatility, confidence, funding) — not a user leverage setting.

## Permissions

- `activeTab` / `tabs` / `scripting` — capture & selection UI  
- `contextMenus` — right-click actions  
- `sidePanel` — results panel  
- `storage` — settings + last report + capture preview  
- Host: `https://perpetual-pro.onrender.com` only  

## File map

```
extension/
├── manifest.json
├── background.js
├── popup.html / popup.js
├── sidepanel.html / sidepanel.js
├── content/select-area.js + .css
├── shared/ocr.js, render.js, vision_light.js, url_parse.js, styles.css
├── lib/          # vendored Tesseract.js
└── icons/
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Cannot reach API | Render free tier cold start — wait ~30s, Refresh |
| Wrong symbol | Clear, leave override blank, recapture; or type override for that capture only |
| Capture fails on chrome:// | Use a normal https page (TradingView, etc.) |

## Disclaimer

Not financial advice. Educational / research tooling only.
