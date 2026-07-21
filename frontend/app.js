/**
 * SubVoice – Frontend Application v2
 *
 * Fixes:
 *  - Canvas initialised to container CSS size immediately (not after video metadata)
 *  - ROI coordinates correctly normalised at draw-time, not at mouse-event time
 *  - OCR loop starts only after WS is OPEN
 *  - Voice file browser fetches /voices list from backend
 *  - Debug panel shows WS / OCR / TTS state
 */

/* ════════════════════════════════════════════════════
   CONSTANTS & STATE
   ════════════════════════════════════════════════════ */
const WS_URL          = `ws://${location.host}/ws`;
const OCR_INTERVAL_MS = 1000;
const SPEEDUP_RATE    = 3.0;
const SPEEDUP_THRESH  = 2.0;

const state = {
  stream:      null,
  ws:          null,
  video:       null,
  roi:         null,      // { x,y,w,h } normalised 0..1 relative to canvas display size
  drawingRoi:  false,
  roiStart:    null,      // { x,y } in canvas display-pixel space
  settings: {
    brightness: 0,
    contrast:   100,
    blur:       0,
    sharpen:    0,
    threshold:  "none",
  },
  isCapturing: false,
  audioQueue:  [],
  isPlaying:   false,
  ocrLoopId:   null,
};

/* ════════════════════════════════════════════════════
   DOM REFERENCES
   ════════════════════════════════════════════════════ */
const $  = id => document.getElementById(id);

const canvasArea    = $("canvas-area");
const screenCanvas  = $("screen-canvas");
const roiCanvas     = $("roi-canvas");
const screenCtx     = screenCanvas.getContext("2d");
const roiCtx        = roiCanvas.getContext("2d");
const canvasEmpty   = $("canvas-empty");

const btnShare      = $("btn-share");
const btnStop       = $("btn-stop");
const btnClearRoi   = $("btn-clear-roi");
const btnClearHist  = $("btn-clear-hist");

const statusDot     = $("status-dot");
const statusLabel   = $("status-label");
const gpuLabel      = $("gpu-label");
const voiceLabel    = $("voice-label");
const roiLabel      = $("roi-label");

const ocrPreview    = $("ocr-preview");
const previewEmpty  = $("preview-empty");

const slBrightness  = $("sl-brightness");
const slContrast    = $("sl-contrast");
const slBlur        = $("sl-blur");
const slSharpen     = $("sl-sharpen");
const segThreshold  = $("seg-threshold");
const outBrightness = $("out-brightness");
const outContrast   = $("out-contrast");
const outBlur       = $("out-blur");
const outSharpen    = $("out-sharpen");

const confFill      = $("conf-fill");
const confVal       = $("conf-val");

const subtitleBody  = $("subtitle-body");
const subtitleCard  = $("subtitle-card");

const badgeStatus   = $("badge-status");
const badgeSpeed    = $("badge-speed");
const waveformBars  = $("waveform-bars");
const progressFill  = $("progress-fill");
const timeCur       = $("time-cur");
const timeTot       = $("time-tot");
const queueRow      = $("queue-row");
const queueLabel    = $("queue-label");

const audioEl       = $("audio-el");

const dropZone      = $("drop-zone");
const voiceInput    = $("voice-input");
const dropLabel     = $("drop-label");
const voiceFeedback = $("voice-feedback");
const voiceList     = $("voice-list");

const histScroll    = $("history-scroll");
const dbgText       = $("dbg-text");

/* ════════════════════════════════════════════════════
   WAVEFORM SETUP
   ════════════════════════════════════════════════════ */
(function buildWaveform() {
  for (let i = 0; i < 28; i++) {
    const b = document.createElement("div");
    b.className   = "wbar";
    b.style.height = `${8 + Math.random() * 14}px`;
    waveformBars.appendChild(b);
  }
})();

function setWaveformActive(on) {
  waveformBars.querySelectorAll(".wbar").forEach(b => b.classList.toggle("active", on));
}

/* ════════════════════════════════════════════════════
   DEBUG HELPER
   ════════════════════════════════════════════════════ */
function dbg(msg) {
  console.log("[SubVoice]", msg);
  if (dbgText) dbgText.textContent = msg;
}

/* ════════════════════════════════════════════════════
   CANVAS – initialise to container CSS size
   Always call this before any draw / mouse operation.
   ════════════════════════════════════════════════════ */
function initCanvasSize() {
  const r = canvasArea.getBoundingClientRect();
  const w = Math.max(Math.round(r.width),  640);
  const h = Math.max(Math.round(r.height), 360);
  if (screenCanvas.width !== w || screenCanvas.height !== h) {
    screenCanvas.width  = w;
    screenCanvas.height = h;
    roiCanvas.width     = w;
    roiCanvas.height    = h;
    dbg(`Canvas → ${w}×${h}`);
  }
}

// Keep canvas sized to container on resize
new ResizeObserver(initCanvasSize).observe(canvasArea);

/* ════════════════════════════════════════════════════
   HEALTH CHECK + VOICE LIST
   ════════════════════════════════════════════════════ */
async function fetchHealth() {
  try {
    const d = await fetch("/health").then(r => r.json());
    gpuLabel.textContent = d.gpu_name || (d.cuda_available ? "GPU OK" : "No GPU");
    if (d.voice_filename) {
      voiceLabel.textContent = d.voice_filename;
      voiceLabel.parentElement.style.color = "var(--accent-3)";
    }
    dbg(`Health OK | GPU: ${d.gpu_name} | OCR: ${d.ocr_ready} | TTS: ${d.tts_ready}`);
  } catch (e) {
    gpuLabel.textContent = "Backend offline";
    dbg("Health check failed: " + e.message);
  }
}

async function fetchVoiceList() {
  try {
    const files = await fetch("/voices-list").then(r => r.json());
    renderVoiceList(files);
  } catch (e) {
    dbg("Voice list fetch failed: " + e.message);
  }
}

function renderVoiceList(files) {
  if (!voiceList) return;
  voiceList.innerHTML = "";
  if (!files || files.length === 0) {
    voiceList.innerHTML = '<div class="vl-empty">Brak plików w /voices/</div>';
    return;
  }
  files.forEach(f => {
    const btn = document.createElement("button");
    btn.className   = "vl-btn";
    btn.textContent = f;
    btn.title       = f;
    btn.onclick = () => selectVoiceFromFolder(f, btn);
    voiceList.appendChild(btn);
  });
}

async function selectVoiceFromFolder(filename, btn) {
  voiceFeedback.className   = "voice-feedback";
  voiceFeedback.textContent = "Wczytywanie " + filename + "…";
  try {
    const d = await fetch("/select-voice", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ filename }),
    }).then(r => r.json());

    if (d.error) {
      voiceFeedback.textContent = "Błąd: " + d.error;
      voiceFeedback.className   = "voice-feedback error";
      return;
    }
    voiceFeedback.textContent = "✓ Aktywny: " + filename;
    voiceLabel.textContent    = filename;
    voiceLabel.parentElement.style.color = "var(--accent-3)";

    // Highlight active
    voiceList.querySelectorAll(".vl-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    dbg("Voice selected: " + filename);
  } catch (e) {
    voiceFeedback.textContent = "Błąd: " + e.message;
    voiceFeedback.className   = "voice-feedback error";
  }
}

fetchHealth();
fetchVoiceList();

/* ════════════════════════════════════════════════════
   WEBSOCKET
   ════════════════════════════════════════════════════ */
function connectWS(onOpen) {
  state.ws = new WebSocket(WS_URL);

  state.ws.addEventListener("open", () => {
    setStatus("connected", "Połączono");
    dbg("WS connected");
    if (onOpen) onOpen();
  });

  state.ws.addEventListener("close", () => {
    setStatus("", "Rozłączony");
    dbg("WS closed");
    if (state.isCapturing) {
      // Reconnect after 2s
      setTimeout(() => connectWS(() => startOcrLoop()), 2000);
    }
  });

  state.ws.addEventListener("error", err => {
    setStatus("error", "Błąd WS");
    dbg("WS error: " + JSON.stringify(err));
  });

  state.ws.addEventListener("message", e => {
    try { handleServerMessage(JSON.parse(e.data)); }
    catch (_) {}
  });
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "ocr_result":  updateOcrResult(msg); break;
    case "audio_ready": onAudioReady(msg);    break;
    case "tts_error":
      setBadge("Błąd TTS", "");
      dbg("TTS error: " + msg.message);
      break;
    case "error":
      dbg("Server error: " + msg.message);
      break;
  }
}

function setStatus(dotClass, text) {
  statusDot.className  = "status-dot " + dotClass;
  statusLabel.textContent = text;
}

/* ════════════════════════════════════════════════════
   SCREEN CAPTURE
   ════════════════════════════════════════════════════ */
btnShare.addEventListener("click", async () => {
  try {
    state.stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 5, max: 10 } },
      audio: false,
    });

    const video       = document.createElement("video");
    video.srcObject   = state.stream;
    video.muted       = true;
    video.playsInline = true;
    await video.play();
    state.video       = video;
    state.isCapturing = true;

    // Init canvas to current container size right away
    initCanvasSize();

    canvasEmpty.classList.add("hidden");
    btnShare.disabled = true;
    btnStop.disabled  = false;

    // When video metadata loads, resize canvas to native video resolution
    video.addEventListener("loadedmetadata", () => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw > 0 && vh > 0) {
        screenCanvas.width  = vw;
        screenCanvas.height = vh;
        roiCanvas.width     = vw;
        roiCanvas.height    = vh;
        // Reset ROI as dimensions changed
        clearRoi();
        dbg(`Video loaded ${vw}×${vh}`);
      }
    });

    // Handle stream ending (user stopped sharing in browser)
    state.stream.getVideoTracks()[0].addEventListener("ended", stopCapture);

    // Connect WS; start loop ONLY after WS open
    connectWS(() => startOcrLoop());

  } catch (err) {
    if (err.name !== "NotAllowedError") {
      console.error("Screen capture error:", err);
      dbg("Capture error: " + err.message);
    }
  }
});

btnStop.addEventListener("click", stopCapture);

function stopCapture() {
  state.isCapturing = false;

  clearInterval(state.ocrLoopId);
  state.ocrLoopId = null;

  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }

  screenCtx.clearRect(0, 0, screenCanvas.width, screenCanvas.height);
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  canvasEmpty.classList.remove("hidden");
  btnShare.disabled = false;
  btnStop.disabled  = true;
  setStatus("", "Rozłączony");
}

/* ════════════════════════════════════════════════════
   OCR LOOP – 1 frame per second
   Called only after WS is OPEN.
   ════════════════════════════════════════════════════ */
function startOcrLoop() {
  if (state.ocrLoopId) clearInterval(state.ocrLoopId);

  let frameCount = 0;
  dbg("OCR loop started");

  state.ocrLoopId = setInterval(() => {
    const video = state.video;
    if (!state.isCapturing || !video) { dbg("skip: no video"); return; }
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) { dbg("skip: WS not open ("+state.ws?.readyState+")"); return; }
    if (video.readyState < 2) { dbg("skip: video not ready ("+video.readyState+")"); return; }

    // Sync canvas to video native size
    if (video.videoWidth > 0 && screenCanvas.width !== video.videoWidth) {
      screenCanvas.width  = video.videoWidth;
      screenCanvas.height = video.videoHeight;
      roiCanvas.width     = video.videoWidth;
      roiCanvas.height    = video.videoHeight;
      dbg(`Canvas synced to video: ${video.videoWidth}×${video.videoHeight}`);
    }

    // Draw video frame to canvas
    screenCtx.drawImage(video, 0, 0, screenCanvas.width, screenCanvas.height);
    drawRoiOverlay();

    // Encode as JPEG – toDataURL is synchronous, no Promise chain needed
    let dataUrl;
    try {
      dataUrl = screenCanvas.toDataURL("image/jpeg", 0.75);
    } catch (e) {
      dbg("toDataURL error: " + e.message);
      return;
    }
    const b64 = dataUrl.split(",")[1];
    if (!b64 || b64.length < 50) { dbg("Empty dataURL, skip"); return; }

    frameCount++;
    dbg(`Frame #${frameCount} sent (${Math.round(b64.length/1024)}KB) roi=${state.roi ? 'yes' : 'none'}`);
    setStatus("processing connected", "OCR aktywny");

    state.ws.send(JSON.stringify({
      type:     "frame",
      image:    b64,
      roi:      state.roi,
      settings: { ...state.settings },
    }));

  }, OCR_INTERVAL_MS);
}

/* ════════════════════════════════════════════════════
   ROI SELECTION
   Coordinates stored in NORMALISED (0..1) space,
   always relative to canvas display rect so they
   survive canvas resize.
   ════════════════════════════════════════════════════ */
roiCanvas.addEventListener("mousedown", e => {
  if (!state.isCapturing) return;
  state.drawingRoi = true;
  const r = roiCanvas.getBoundingClientRect();
  // Normalised coords relative to displayed size
  state.roiStart = {
    nx: (e.clientX - r.left) / r.width,
    ny: (e.clientY - r.top)  / r.height,
  };
});

roiCanvas.addEventListener("mousemove", e => {
  if (!state.drawingRoi) return;
  const r  = roiCanvas.getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width;
  const ny = (e.clientY - r.top)  / r.height;
  previewRoi(state.roiStart.nx, state.roiStart.ny, nx - state.roiStart.nx, ny - state.roiStart.ny);
});

roiCanvas.addEventListener("mouseup", e => {
  if (!state.drawingRoi) return;
  state.drawingRoi = false;

  const r  = roiCanvas.getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width;
  const ny = (e.clientY - r.top)  / r.height;

  const x = Math.min(state.roiStart.nx, nx);
  const y = Math.min(state.roiStart.ny, ny);
  const w = Math.abs(nx - state.roiStart.nx);
  const h = Math.abs(ny - state.roiStart.ny);

  // Require a minimum region size (5% width, 2% height)
  if (w > 0.02 && h > 0.01) {
    state.roi = { x, y, w, h };
    roiLabel.textContent = `ROI: ${(x*100).toFixed(0)}% ${(y*100).toFixed(0)}% → ${(w*100).toFixed(0)}×${(h*100).toFixed(0)}%`;
    dbg(`ROI set: x=${x.toFixed(3)} y=${y.toFixed(3)} w=${w.toFixed(3)} h=${h.toFixed(3)}`);
  } else {
    dbg("ROI too small, ignored");
  }
});

// Touch support
roiCanvas.addEventListener("touchstart", e => {
  e.preventDefault();
  roiCanvas.dispatchEvent(new MouseEvent("mousedown", { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY }));
}, { passive: false });
roiCanvas.addEventListener("touchmove",  e => {
  e.preventDefault();
  roiCanvas.dispatchEvent(new MouseEvent("mousemove", { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY }));
}, { passive: false });
roiCanvas.addEventListener("touchend",   e => {
  e.preventDefault();
  roiCanvas.dispatchEvent(new MouseEvent("mouseup", { clientX: e.changedTouches[0].clientX, clientY: e.changedTouches[0].clientY }));
}, { passive: false });

function previewRoi(nx, ny, nw, nh) {
  const cw = roiCanvas.width;
  const ch = roiCanvas.height;
  const x  = Math.min(nx, nx + nw) * cw;
  const y  = Math.min(ny, ny + nh) * ch;
  const w  = Math.abs(nw) * cw;
  const h  = Math.abs(nh) * ch;

  roiCtx.clearRect(0, 0, cw, ch);
  roiCtx.fillStyle = "rgba(0,0,0,0.4)";
  roiCtx.fillRect(0, 0, cw, ch);
  roiCtx.clearRect(x, y, w, h);
  roiCtx.strokeStyle  = "#818cf8";
  roiCtx.lineWidth    = 2;
  roiCtx.setLineDash([8, 4]);
  roiCtx.strokeRect(x, y, w, h);
  roiCtx.setLineDash([]);
  // Corner labels
  roiCtx.fillStyle  = "#818cf8";
  roiCtx.font       = "12px Inter,sans-serif";
  roiCtx.fillText(`${Math.round(w)}×${Math.round(h)}px`, x + 4, y + 14);
}

function drawRoiOverlay() {
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  if (!state.roi) return;

  const cw = roiCanvas.width;
  const ch = roiCanvas.height;
  const { x, y, w, h } = state.roi;
  const px = x * cw, py = y * ch, pw = w * cw, ph = h * ch;

  roiCtx.fillStyle = "rgba(0,0,0,0.35)";
  roiCtx.fillRect(0, 0, cw, ch);
  roiCtx.clearRect(px, py, pw, ph);
  roiCtx.strokeStyle  = "rgba(129,140,248,0.85)";
  roiCtx.lineWidth    = 2;
  roiCtx.strokeRect(px, py, pw, ph);
}

function clearRoi() {
  state.roi = null;
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  roiLabel.textContent = "Przeciągnij myszą by zaznaczyć region OCR";
}

btnClearRoi.addEventListener("click", clearRoi);

/* ════════════════════════════════════════════════════
   OCR RESULT HANDLER
   ════════════════════════════════════════════════════ */
function updateOcrResult(msg) {
  const { text, confidence, preview, changed } = msg;

  // Confidence bar
  confFill.style.width = (confidence || 0) + "%";
  confVal.textContent  = (confidence || 0).toFixed(0) + "%";

  // ALWAYS show current OCR text so user can see what's detected
  const display = text.trim() || "(brak tekstu)";
  if (text.trim()) {
    // Show raw text in subtitle area regardless of change threshold
    if (subtitleBody.dataset.raw !== display) {
      subtitleBody.dataset.raw = display;
      // Only animate when it truly changed for TTS
      if (changed) {
        subtitleBody.classList.remove("flash");
        void subtitleBody.offsetWidth;
        subtitleBody.classList.add("flash");
        subtitleCard.classList.add("active");
        setTimeout(() => subtitleCard.classList.remove("active"), 2000);
        addHistory(text);
        dbg(`TTS triggered: "${text.slice(0, 50)}"`);
      }
      subtitleBody.textContent = text;
    }
  } else {
    subtitleBody.dataset.raw = "";
  }

  // Preview image
  if (preview) {
    ocrPreview.src           = "data:image/jpeg;base64," + preview;
    ocrPreview.style.display = "block";
    if (previewEmpty) previewEmpty.style.display = "none";
  }

  dbg(`OCR: conf=${(confidence||0).toFixed(0)}% changed=${changed} text="${(text||'').slice(0,40)}"`);
}

/* ════════════════════════════════════════════════════
   AUDIO PLAYER
   ════════════════════════════════════════════════════ */
function onAudioReady({ url, text }) {
  dbg(`Audio ready: ${url}`);
  if (!state.isPlaying) {
    playAudio(url);
  } else {
    state.audioQueue.push({ url, text });
    updateQueueBadge();
    const remaining = isNaN(audioEl.duration) ? 0 : (audioEl.duration - audioEl.currentTime);
    if (remaining > SPEEDUP_THRESH) {
      audioEl.playbackRate = SPEEDUP_RATE;
      badgeSpeed.hidden    = false;
      setBadge("Przyspieszanie", "playing");
    }
  }
}

function playAudio(url) {
  state.isPlaying      = true;
  audioEl.playbackRate = 1.0;
  audioEl.src          = url;
  audioEl.load();
  audioEl.play().catch(e => { console.warn("[Audio]", e); onAudioEnded(); });
  setWaveformActive(true);
  setBadge("Odtwarzanie", "playing");
  badgeSpeed.hidden = true;
}

function onAudioEnded() {
  state.isPlaying = false;
  setWaveformActive(false);
  badgeSpeed.hidden = true;
  if (state.audioQueue.length > 0) {
    const next = state.audioQueue.shift();
    updateQueueBadge();
    playAudio(next.url);
  } else {
    setBadge("Gotowy", "");
    progressFill.style.width = "0%";
    timeCur.textContent = "0:00";
    timeTot.textContent = "0:00";
  }
}

audioEl.addEventListener("ended",       onAudioEnded);
audioEl.addEventListener("error",  ()=> onAudioEnded());
audioEl.addEventListener("timeupdate", () => {
  const cur = audioEl.currentTime;
  const dur = isNaN(audioEl.duration) ? 0 : audioEl.duration;
  if (dur > 0) {
    progressFill.style.width = (cur / dur * 100) + "%";
    timeCur.textContent = fmtTime(cur);
    timeTot.textContent = fmtTime(dur);
  }
});

function fmtTime(s) {
  return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}`;
}

function setBadge(text, type) {
  badgeStatus.textContent = text;
  badgeStatus.className   = "badge badge-status " + (type || "");
}

function updateQueueBadge() {
  const n = state.audioQueue.length;
  queueLabel.textContent = n === 0 ? "Kolejka: pusta" : `Kolejka: ${n} audio`;
  queueRow.classList.toggle("has-items", n > 0);
}

/* ════════════════════════════════════════════════════
   SLIDERS
   ════════════════════════════════════════════════════ */
function bindSlider(el, out, key, fmt) {
  const update = () => {
    state.settings[key] = parseFloat(el.value);
    out.textContent     = fmt(el.value);
  };
  el.addEventListener("input", update);
  update();
}

bindSlider(slBrightness, outBrightness, "brightness", v => (+v > 0 ? "+" : "") + v);
bindSlider(slContrast,   outContrast,   "contrast",   v => (v/100).toFixed(1) + "×");
bindSlider(slBlur,       outBlur,       "blur",       v => v);
bindSlider(slSharpen,    outSharpen,    "sharpen",    v => v);

segThreshold.querySelectorAll(".seg-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    segThreshold.querySelectorAll(".seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.settings.threshold = btn.dataset.val;
  });
});

/* ════════════════════════════════════════════════════
   VOICE UPLOAD (drag & drop / click)
   ════════════════════════════════════════════════════ */
dropZone.addEventListener("click",    ()   => voiceInput.click());
dropZone.addEventListener("keydown",  e    => (e.key === "Enter" || e.key === " ") && voiceInput.click());
dropZone.addEventListener("dragover", e    => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave",()   => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop",     e    => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) uploadVoice(e.dataTransfer.files[0]);
});
voiceInput.addEventListener("change", () => voiceInput.files[0] && uploadVoice(voiceInput.files[0]));

async function uploadVoice(file) {
  dropLabel.textContent       = file.name;
  voiceFeedback.textContent   = "Wysyłanie…";
  voiceFeedback.className     = "voice-feedback";

  const fd = new FormData();
  fd.append("file", file);

  try {
    const d = await fetch("/upload-voice", { method: "POST", body: fd }).then(r => r.json());
    if (d.error) {
      voiceFeedback.textContent = "Błąd: " + d.error;
      voiceFeedback.className   = "voice-feedback error";
      return;
    }
    voiceFeedback.textContent = "✓ Wczytano: " + d.filename;
    voiceLabel.textContent    = d.filename;
    voiceLabel.parentElement.style.color = "var(--accent-3)";
    // Refresh file list
    fetchVoiceList();
    dbg("Voice uploaded: " + d.filename);
  } catch (e) {
    voiceFeedback.textContent = "Błąd: " + e.message;
    voiceFeedback.className   = "voice-feedback error";
  }
}

/* ════════════════════════════════════════════════════
   HISTORY
   ════════════════════════════════════════════════════ */
function addHistory(text) {
  const empty = histScroll.querySelector(".history-empty");
  if (empty) empty.remove();

  const item = document.createElement("div");
  item.className = "history-item";
  item.innerHTML = `
    <div class="history-item-text">${escHtml(text)}</div>
    <div class="history-item-time">${new Date().toLocaleTimeString("pl-PL")}</div>`;
  histScroll.insertBefore(item, histScroll.firstChild);

  while (histScroll.children.length > 50) histScroll.removeChild(histScroll.lastChild);
}

btnClearHist.addEventListener("click", () => {
  histScroll.innerHTML = '<div class="history-empty">Historia jest pusta</div>';
});

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

/* ════════════════════════════════════════════════════
   TTS TEST CARD
   ════════════════════════════════════════════════════ */
const ttsTestInput  = $("tts-test-input");
const ttsTestBtn    = $("btn-tts-test");
const ttsTestStatus = $("tts-test-status");
const ttsCharCur    = $("tts-char-cur");

// Enable button when there's text; show char count
ttsTestInput.addEventListener("input", () => {
  const len = ttsTestInput.value.length;
  ttsCharCur.textContent = len;
  ttsTestBtn.disabled    = len === 0;
});

// Ctrl+Enter submits
ttsTestInput.addEventListener("keydown", e => {
  if (e.ctrlKey && e.key === "Enter") ttsTestBtn.click();
});

ttsTestBtn.addEventListener("click", async () => {
  const text = ttsTestInput.value.trim();
  if (!text) return;

  // UI: loading state
  ttsTestBtn.disabled    = true;
  ttsTestStatus.className = "tts-test-status loading";
  ttsTestStatus.textContent = "⏳ Generowanie audio…";

  try {
    const res  = await fetch("/tts-test", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();

    if (data.error) {
      ttsTestStatus.className   = "tts-test-status error";
      ttsTestStatus.textContent = "✗ " + data.error;
      return;
    }

    ttsTestStatus.className   = "tts-test-status ok";
    ttsTestStatus.textContent = "✓ Gotowe – odtwarzanie…";

    // Play via the shared audio player (queues if busy)
    onAudioReady({ url: data.url, text: data.text });

  } catch (e) {
    ttsTestStatus.className   = "tts-test-status error";
    ttsTestStatus.textContent = "✗ Błąd połączenia: " + e.message;
  } finally {
    ttsTestBtn.disabled = ttsTestInput.value.length === 0;
  }
});

/* ════════════════════════════════════════════════════
   INIT
   ════════════════════════════════════════════════════ */
setStatus("", "Rozłączony");
updateQueueBadge();
initCanvasSize();
