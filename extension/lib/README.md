# Bundled Tesseract.js (offline OCR)

These files are vendored so the extension never loads scripts from a CDN.

| File | Source package | Role |
|------|----------------|------|
| `tesseract.min.js` | `tesseract.js@5.1.1` | Browser API (UMD → `window.Tesseract`) |
| `worker.min.js` | `tesseract.js@5.1.1` | Web Worker entry |
| `tesseract-core-simd-lstm.wasm.js` | `tesseract.js-core@5.1.1` | WASM glue (SIMD + LSTM) |
| `tesseract-core-simd-lstm.wasm` | `tesseract.js-core@5.1.1` | WASM binary (loaded by glue) |
| `lang-data/eng.traineddata.gz` | tessdata 4.0.0 | English language pack |

## Paths used by `shared/ocr.js`

```
workerPath → chrome.runtime.getURL('lib/worker.min.js')
corePath   → chrome.runtime.getURL('lib/tesseract-core-simd-lstm.wasm.js')
langPath   → chrome.runtime.getURL('lib/lang-data')
```

## Re-vendoring (maintainers)

```bash
cd /tmp && npm pack tesseract.js@5.1.1 && npm pack tesseract.js-core@5.1.1
# extract and copy dist/tesseract.min.js, dist/worker.min.js,
# tesseract-core-simd-lstm.wasm(.js) into extension/lib/
# eng.traineddata.gz into extension/lib/lang-data/
```
