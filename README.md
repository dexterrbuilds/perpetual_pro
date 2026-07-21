# perpetual_pro

**Professional-grade crypto perpetual futures analysis CLI** — live market data, derivatives metrics, multi-timeframe confluence, market structure (OB / FVG / BOS / CHoCH), pattern recognition, news sentiment, and optional **screen capture → OCR + computer vision** mode.

Built to feel like a senior prop trader sitting next to you.

> **Not financial advice.** Perpetuals are high-risk. You can lose capital quickly.

---

## Features

### Data Mode (primary)
- Exchanges via **ccxt**: Binance USDM, Bybit, OKX, Bitget
- OHLCV + **funding rate**, **open interest**, **long/short ratio** (when available)
- Multi-timeframe analysis (primary + higher TFs)
- **80+ indicators** via pandas-ta (trend, momentum, volatility, volume, Ichimoku, squeeze, etc.)
- Market structure: Order Blocks, Fair Value Gaps, liquidity pools, BOS / CHoCH, volume profile approx
- Candlestick + classical chart patterns with confidence scores
- RSI / MACD / Stochastic **divergences**
- News + lexicon sentiment (CryptoCompare / CoinGecko; CryptoPanic with token)
- Weighted **confluence engine** (trend, momentum, structure, patterns, derivatives, MTF, volume, news)
- Risk engine: entry zone, SL, TP1–TP3, R:R, position size from account risk %

### Screen Mode (`--screen`)
- Interactive region select, full-screen, fixed region, or image file
- Preprocessing tuned for TradingView / exchange dark charts
- Dual OCR: **Tesseract + EasyOCR**
- CV: candle blobs, trendlines, horizontal S/R, volume bars
- Optional **Ollama** vision model (LLaVA, etc.) if running locally
- Fallback: full live data analysis using detected (or provided) symbol

### Output
- Rich terminal report (bias, setup, confluence table, structure, news, scenarios)
- Save **Markdown + JSON**
- Optional annotated chart image with levels / SL / TPs

### Prop account toolkit
- **Prop risk** (default): **0.5–1% risk per trade**, **max 5x leverage**, flags for high drawdown / wide stop / low R:R
- **LLM confidence** with supporting vs opposing factors in reports
- **Backtest**: `python main.py BTC --backtest --bars 500` (win rate, profit factor, max DD, equity curve)
- **Scheduled scans** at **05:00 / 16:00 / 20:00 WAT** + **Telegram** high-confidence alerts

**Telegram secrets are env-only** (never put tokens in `config.yaml` or commit them):

```bash
# Copy .env.example → .env (gitignored) and fill:
export TELEGRAM_BOT_TOKEN="your-bot-token-from-BotFather"
export TELEGRAM_CHAT_ID="your-chat-id"

python scripts/run_scheduled_scans.py --once   # test one high-conf report now
python scripts/run_scheduled_scans.py          # loop: 05:00 / 16:00 / 20:00 WAT
# or: python main.py --scheduled-scan
```

Streamlit: **Scan & analyze** (manual, always fresh) + **Backtest** tabs.

---

## Project structure

```
perpetual_pro/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
├── .env.example
├── src/
│   ├── cli.py
│   ├── data/          # exchange, multi-tf, news
│   ├── analysis/      # indicators, patterns, structure, confluence, risk
│   ├── vision/        # capture, ocr, chart_detect, preprocess
│   ├── report/        # rich + MD/JSON
│   └── utils/         # config, helpers
└── tests/
```

---

## Requirements

- **Python 3.10+**
- macOS / Linux / Windows (screen capture works best on a desktop session)
- System **Tesseract OCR** binary (for Tesseract engine)

### Install Tesseract (system)

**macOS (Homebrew):**
```bash
brew install tesseract
# Apple Silicon often: /opt/homebrew/bin/tesseract
# Intel: /usr/local/bin/tesseract
```

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

**Windows:**  
Install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and set:
```bash
# .env or config.yaml
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## Installation

```bash
cd perpetual_pro
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional secrets
cp .env.example .env
# edit .env — API keys only needed for private endpoints; public data works without
```

**Note on EasyOCR:** first run downloads recognition models (~100MB). Use `ocr.engine: tesseract` in `config.yaml` if you want to skip EasyOCR.

**Indicators:** production uses **`pandas-ta-classic`** (import name `pandas_ta_classic`).  
If it is missing, the app still boots and uses a pure-pandas fallback suite.

```bash
pip install pandas-ta-classic>=0.4.0
# verify: python scripts/check_ta_backend.py
```

---

## FastAPI server

```bash
# Install API deps (if not already)
pip install fastapi "uvicorn[standard]" python-multipart

# Start server
uvicorn main_server:app --reload --port 8000
# or: python main_server.py
```

Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### `POST /analyze`

Multipart form upload of a chart screenshot. Optional fields override OCR.

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | **Required** PNG/JPEG chart screenshot |
| `symbol` | string | e.g. `BTC` or `BTC/USDT:USDT` (OCR fallback if omitted) |
| `timeframe` | string | e.g. `15m` |
| `exchange` | string | `binanceusdm` / `bybit` / `okx` / `bitget` |
| `higher` | string | e.g. `1h,4h,1d` |
| `balance` | float | Account balance for sizing |
| `risk` | float | Risk % per trade |
| `no_news` | bool | Skip news |
| `dark_theme` | bool | Chart dark theme hint |

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze" \
  -F "image=@./chart.png" \
  -F "symbol=BTC" \
  -F "timeframe=15m" \
  -F "exchange=bybit" | python -m json.tool
```

Response JSON includes `ok`, `bias`, `confidence`, `trade_plan`, `factors`, `patterns`,
`structure`, `news`, `scenarios`, `vision` (OCR/CV), and a disclaimer.

Health: `GET /health`

### Streamlit web app

A private-friendly Streamlit companion app lives in [streamlit_app.py](./streamlit_app.py). It can:
- run single-symbol analysis
- scan a watchlist of symbols and rank setups
- upload a chart image for analysis
- export markdown reports

Run locally:

```bash
streamlit run streamlit_app.py --server.port 8501
```

Set optional env vars for privacy and backend routing:

```bash
export STREAMLIT_PASSWORD="change-me"
export BACKEND_URL="http://127.0.0.1:8000"
```

### Chrome extension

A Manifest V3 extension lives in [`extension/`](./extension/). It calls the fixed production API (hard-coded, no API URL field):

**`https://perpetual-pro.onrender.com`**

1. Chrome → `chrome://extensions` → Developer mode → **Load unpacked** → select `extension/`
2. Use **Capture & Analyze**, `Ctrl+Shift+P`, or right-click a chart image
3. **Clear** resets capture, results, and form fields; **Refresh** reloads status + last report + chart preview
4. Simulation uses a hidden educational **$100** example (not a wallet); report shows suggested hold time and dynamic leverage

See [extension/README.md](./extension/README.md) for details.

---

## Quick start (CLI)

```bash
# Data mode — Bitcoin perps on Binance USDM
python main.py BTC/USDT:USDT

# Short form symbol (normalized to BTC/USDT:USDT)
python main.py BTC --exchange bybit --timeframe 15m --higher 1h,4h,1d

# ETH with custom risk sizing
python main.py ETH --balance 25000 --risk 0.5 --exchange okx

# Screen mode — interactive crop of currently open chart
python main.py --screen

# Screen + force symbol if OCR is weak
python main.py --screen --symbol SOL --exchange binanceusdm

# Fixed region (left,top,width,height)
python main.py --screen --region 80,60,1400,900 --symbol BTC

# Analyze a saved chart image
python main.py --screen --image ./my_chart.png --symbol ETH --timeframe 1h

# List exchanges
python main.py --list-exchanges
```

Reports are written under `./output/` (Markdown + JSON). Captures/annotations under the same tree when screen mode is used.

---

## Configuration

Edit `config.yaml` or override via `.env` / CLI.

| Area | Keys |
|------|------|
| Exchange | `exchange.default`, API key/secret/password |
| Risk | `account_balance`, `risk_per_trade_pct`, `max_leverage`, `min_rr`, ATR multiples |
| Timeframes | `primary`, `higher`, `ohlcv_limit` |
| Confluence weights | `analysis.weights.*` |
| News | `enabled`, `cryptopanic_token`, keyword lists |
| OCR | `engine` (`dual` / `tesseract` / `easyocr`), `tesseract_cmd` |
| Vision | `use_ollama`, `ollama_model`, base URL |
| Output / logging | paths, levels, save flags |

### Environment variables (from `.env.example`)

- `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET` / `EXCHANGE_PASSWORD`
- `PERP_EXCHANGE`
- `CRYPTOPANIC_TOKEN`
- `ACCOUNT_BALANCE` / `RISK_PER_TRADE_PCT`
- `TESSERACT_CMD`
- `OLLAMA_BASE_URL` / `OLLAMA_MODEL`

---

## How the analysis thinks

Confluence is a **weighted sum** of factor scores in `[-1, +1]`:

| Factor | Typical weight | Inputs |
|--------|----------------|--------|
| Trend | 0.18 | EMA stack, slope, vs 200 EMA |
| Momentum | 0.14 | RSI, MACD, stoch + divergences |
| Structure | 0.16 | BOS/CHoCH, OB/FVG proximity, Wyckoff |
| Patterns | 0.12 | Candles + classical geometry |
| Derivatives | 0.12 | Funding extremes, L/S crowding |
| Multi-TF | 0.14 | Higher TF trend alignment |
| Volume | 0.08 | Relative volume, CMF/MFI lean |
| News | 0.06 | Headline sentiment |

Bias thresholds, ADX clarity, and structure conflicts adjust **confidence %**. Trade plan uses ATR + nearby structure for SL/TP and sizes risk as `% of account`.

Scenarios (bullish / base / bearish) are probability-weighted narratives, not guarantees.

---

## Improving accuracy

1. **Use higher-liquidity symbols** (BTC, ETH) — cleaner structure and funding.
2. **Align primary TF with your style** — scalps: 1m–5m; day: 15m–1h; swing: 4h–1d.
3. **Always include HTFs** (`--higher 1h,4h,1d`) and respect multi-TF conflict warnings.
4. **Screen mode:** crop tightly to the candle pane; dark theme setting should match the chart.
5. **Install both OCR engines** for symbol detection; pass `--symbol` when OCR is uncertain.
6. **Ollama + llava** (optional) for qualitative chart commentary:
   ```bash
   ollama pull llava
   ollama serve
   ```
7. **Tune weights** in `config.yaml` to match your playbook (e.g. raise `structure` for SMC-style trading).
8. **API keys** are optional for public OHLCV; some L/S endpoints work better with keys / venue-specific APIs.
9. **Funding extremes** matter more near session funding times — treat crowded positioning as a soft fade signal, not a standalone entry.
10. **Never override risk** to “make the setup work” — if R:R or quality is poor, stand aside.

---

## Tests

```bash
pip install pytest
pytest -q
```

Network-free unit tests cover symbol helpers, patterns, structure, risk, and config load.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Empty OHLCV` / symbol not found | Try `BTC/USDT:USDT`, switch exchange, or check listing |
| Rate limits | Built-in retries; reduce frequency; enable `enable_rate_limit` |
| Tesseract not found | Install binary; set `TESSERACT_CMD` |
| EasyOCR slow first run | Model download; or set `ocr.engine: tesseract` |
| Interactive capture fails | Use `--region` or `--image`; headless servers need `--image` |
| Indicator backend missing | `pip install pandas-ta-classic` — app still runs with pure-pandas fallback |
| News empty | Normal without keys; optional CryptoPanic token |

---

## Disclaimer

This software is provided for **educational and research purposes only**. It does **not** constitute financial, investment, or trading advice. Cryptocurrency perpetual futures involve substantial risk of loss. The authors and contributors accept no liability for any losses incurred. Always do your own research.

---

## License

Use at your own risk. No warranty.
