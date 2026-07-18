# Bundled Tesseract.js (Manifest V3)

Offline OCR assets. **Do not load from CDN.**

## Layout

```
lib/
├── tesseract.min.js                      # UMD API → window.Tesseract
├── worker.min.js                         # Web Worker entry (importScripts-safe)
├── tesseract-core-simd-lstm.wasm.js      # WASM glue
├── tesseract-core-simd-lstm.wasm         # WASM binary (sibling of glue)
└── lang-data/
    └── eng.traineddata.gz                # English pack
```

## MV3 worker rules

| Setting | Value | Why |
|---------|--------|-----|
| `workerPath` | `chrome.runtime.getURL('lib/worker.min.js')` | Real extension worker URL |
| `corePath` | `chrome.runtime.getURL('lib/tesseract-core-simd-lstm.wasm.js')` | Local core, no CDN |
| `langPath` | `chrome.runtime.getURL('lib/lang-data')` | Loads `eng.traineddata.gz` |
| **`workerBlobURL`** | **`false`** | Blob workers **cannot** `importScripts(chrome-extension://…)` |

If `workerBlobURL` is left at the library default (`true`), Chrome fetches the worker into a blob, then the worker fails when it tries to load core/lang from the extension origin.

## CSP

```
script-src 'self' 'wasm-unsafe-eval'
worker-src 'self'
```

No `blob:` workers required when `workerBlobURL: false`.

## Usage (see `shared/ocr.js`)

```js
const worker = await Tesseract.createWorker("eng", 1, {
  workerPath: chrome.runtime.getURL("lib/worker.min.js"),
  corePath: chrome.runtime.getURL("lib/tesseract-core-simd-lstm.wasm.js"),
  langPath: chrome.runtime.getURL("lib/lang-data"),
  workerBlobURL: false,
  gzip: true,
  cacheMethod: "none",
});
```

## Re-vendor

```bash
npm pack tesseract.js@5.1.1
npm pack tesseract.js-core@5.1.1
# copy dist/tesseract.min.js, dist/worker.min.js
# copy tesseract-core-simd-lstm.wasm.js + .wasm
# eng.traineddata.gz → lib/lang-data/
```
