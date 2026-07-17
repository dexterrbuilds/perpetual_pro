/**
 * Perpetual Pro — drag-to-select chart region on the page.
 */
(function () {
  if (window.__ppSelectAreaInstalled) return;
  window.__ppSelectAreaInstalled = true;

  let root = null;
  let box = null;
  let start = null;
  let meta = {};

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type === "PP_PING") {
      sendResponse({ ok: true });
      return true;
    }
    if (msg?.type === "PP_START_SELECTION") {
      meta = {
        symbol: msg.symbol || "",
        timeframe: msg.timeframe || "",
        exchange: msg.exchange || "",
      };
      startSelection();
      sendResponse({ ok: true });
      return true;
    }
    if (msg?.type === "PP_CANCEL_SELECTION") {
      teardown();
      sendResponse({ ok: true });
      return true;
    }
    return false;
  });

  function startSelection() {
    teardown();
    root = document.createElement("div");
    root.id = "pp-select-root";

    const hint = document.createElement("div");
    hint.className = "pp-select-hint";
    hint.innerHTML =
      'Perpetual Pro — drag to select chart · <span>Enter</span> confirm · <span>Esc</span> cancel';
    root.appendChild(hint);

    box = document.createElement("div");
    box.className = "pp-select-box";
    box.style.display = "none";
    root.appendChild(box);

    document.documentElement.appendChild(root);

    root.addEventListener("mousedown", onDown, true);
    window.addEventListener("keydown", onKey, true);
  }

  function onDown(e) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    start = { x: e.clientX, y: e.clientY };
    box.style.display = "block";
    updateBox(e.clientX, e.clientY);

    const onMove = (ev) => {
      ev.preventDefault();
      updateBox(ev.clientX, ev.clientY);
    };
    const onUp = (ev) => {
      ev.preventDefault();
      window.removeEventListener("mousemove", onMove, true);
      window.removeEventListener("mouseup", onUp, true);
      const rect = normalizeRect(start.x, start.y, ev.clientX, ev.clientY);
      if (rect.w < 12 || rect.h < 12) {
        teardown();
        chrome.runtime.sendMessage({ type: "AREA_CANCELLED" });
        return;
      }
      const payload = {
        type: "AREA_SELECTED",
        rect,
        dpr: window.devicePixelRatio || 1,
        symbol: meta.symbol,
        timeframe: meta.timeframe,
        exchange: meta.exchange,
      };
      teardown();
      chrome.runtime.sendMessage(payload);
    };
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
  }

  function updateBox(x2, y2) {
    if (!start || !box) return;
    const r = normalizeRect(start.x, start.y, x2, y2);
    box.style.left = r.x + "px";
    box.style.top = r.y + "px";
    box.style.width = r.w + "px";
    box.style.height = r.h + "px";
  }

  function normalizeRect(x1, y1, x2, y2) {
    const x = Math.min(x1, x2);
    const y = Math.min(y1, y2);
    const w = Math.abs(x2 - x1);
    const h = Math.abs(y2 - y1);
    return { x, y, w, h };
  }

  function onKey(e) {
    if (e.key === "Escape") {
      e.preventDefault();
      teardown();
      chrome.runtime.sendMessage({ type: "AREA_CANCELLED" });
    }
  }

  function teardown() {
    window.removeEventListener("keydown", onKey, true);
    if (root && root.parentNode) root.parentNode.removeChild(root);
    root = null;
    box = null;
    start = null;
  }
})();
