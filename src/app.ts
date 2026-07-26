import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface CaptureTargetInfo {
  id: string;
  title: string;
  is_window: boolean;
}

interface VoiceInfo {
  key: string;
  name: string;
  is_downloaded: boolean;
}

interface TtsDownloadProgressPayload {
  voice_key: string;
  voice_name: string;
  downloaded_bytes: number;
  total_bytes: number;
  pct: number;
  status: string;
  error_msg?: string;
}

interface WordInfo {
  raw: string;
  corrected: string;
  confidence: number;
  is_corrected: boolean;
}

interface SubtitleHistoryEntry {
  id: number;
  timestamp: string;
  raw_text: string;
  corrected_text: string;
  confidence: number;
  words: WordInfo[];
  capture_time_ms: number;
  filter_time_ms: number;
  ocr_time_ms: number;
  tts_time_ms: number;
}

interface TelemetryData {
  scan_interval_ms: number;
  is_paused: boolean;
  audio_queue_len: number;
  current_voice: string;
  base_speed: number;
  effective_speed: number;
  volume: number;
  ocr_loaded: boolean;
  tts_loaded: boolean;
  capture_time_ms: number;
  filter_time_ms: number;
  ocr_time_ms: number;
  tts_time_ms: number;
}

document.addEventListener("DOMContentLoaded", async () => {
  // Elements
  const targetSelect = document.getElementById("target-select") as HTMLSelectElement;
  const btnRefreshTargets = document.getElementById("btn-refresh-targets") as HTMLButtonElement;
  const btnPause = document.getElementById("btn-pause") as HTMLButtonElement;
  const pauseText = document.getElementById("pause-text") as HTMLSpanElement;

  const badgeOcr = document.getElementById("badge-ocr") as HTMLSpanElement;
  const badgeTts = document.getElementById("badge-tts") as HTMLSpanElement;

  const sliderInterval = document.getElementById("slider-interval") as HTMLInputElement;
  const valInterval = document.getElementById("val-interval") as HTMLSpanElement;

  const sliderFrameDiffThreshold = document.getElementById("slider-frame-diff-threshold") as HTMLInputElement;
  const valFrameDiffThreshold = document.getElementById("val-frame-diff-threshold") as HTMLSpanElement;

  const chkOcrFilters = document.querySelectorAll<HTMLInputElement>(".chk-ocr-filter");

  const sliderCropCutTop = document.getElementById("slider-crop-cut-top") as HTMLInputElement;
  const valCropCutTop = document.getElementById("val-crop-cut-top") as HTMLSpanElement;

  const sliderCropCutBottom = document.getElementById("slider-crop-cut-bottom") as HTMLInputElement;
  const valCropCutBottom = document.getElementById("val-crop-cut-bottom") as HTMLSpanElement;

  const sliderCropCutLeft = document.getElementById("slider-crop-cut-left") as HTMLInputElement;
  const valCropCutLeft = document.getElementById("val-crop-cut-left") as HTMLSpanElement;

  const sliderCropCutRight = document.getElementById("slider-crop-cut-right") as HTMLInputElement;
  const valCropCutRight = document.getElementById("val-crop-cut-right") as HTMLSpanElement;

  const boxOverlayTop = document.getElementById("box-overlay-top") as HTMLDivElement;
  const boxValTop = document.getElementById("box-val-top") as HTMLElement;
  const boxOverlayBottom = document.getElementById("box-overlay-bottom") as HTMLDivElement;
  const boxValBottom = document.getElementById("box-val-bottom") as HTMLElement;
  const boxOverlayLeft = document.getElementById("box-overlay-left") as HTMLDivElement;
  const boxValLeft = document.getElementById("box-val-left") as HTMLElement;
  const boxOverlayRight = document.getElementById("box-overlay-right") as HTMLDivElement;
  const boxValRight = document.getElementById("box-val-right") as HTMLElement;
  const boxActiveKadr = document.getElementById("box-active-kadr") as HTMLDivElement;
  const boxActiveDims = document.getElementById("box-active-dims") as HTMLSpanElement;
  const presetBtns = document.querySelectorAll<HTMLButtonElement>(".preset-btn");

  const selectVoice = document.getElementById("select-voice") as HTMLSelectElement;
  const ttsDownloadStatus = document.getElementById("tts-download-status") as HTMLDivElement;
  const ttsStatusText = document.getElementById("tts-status-text") as HTMLSpanElement;
  const ttsStatusPct = document.getElementById("tts-status-pct") as HTMLSpanElement;
  const ttsProgressBarFill = document.getElementById("tts-progress-bar-fill") as HTMLDivElement;
  const sliderSpeed = document.getElementById("slider-speed") as HTMLInputElement;
  const valSpeed = document.getElementById("val-speed") as HTMLSpanElement;

  const sliderVolume = document.getElementById("slider-volume") as HTMLInputElement;
  const valVolume = document.getElementById("val-volume") as HTMLSpanElement;

  const sliderAutocorrectThreshold = document.getElementById("slider-autocorrect-threshold") as HTMLInputElement;
  const valAutocorrectThreshold = document.getElementById("val-autocorrect-threshold") as HTMLSpanElement;

  const sliderWordRejectThreshold = document.getElementById("slider-word-reject-threshold") as HTMLInputElement;
  const valWordRejectThreshold = document.getElementById("val-word-reject-threshold") as HTMLSpanElement;

  const imgScanPreview = document.getElementById("img-scan-preview") as HTMLImageElement;
  const previewCropInfo = document.getElementById("preview-crop-info") as HTMLSpanElement;
  const activeTimestamp = document.getElementById("active-timestamp") as HTMLSpanElement;
  const activeConfidence = document.getElementById("active-confidence") as HTMLSpanElement;
  const textCorrected = document.getElementById("text-corrected") as HTMLDivElement;
  const textRaw = document.getElementById("text-raw") as HTMLSpanElement;

  const metricCapTime = document.getElementById("metric-cap-time") as HTMLSpanElement;
  const metricFilterTime = document.getElementById("metric-filter-time") as HTMLSpanElement;
  const metricOcrTime = document.getElementById("metric-ocr-time") as HTMLSpanElement;
  const metricTtsTime = document.getElementById("metric-tts-time") as HTMLSpanElement;
  const metricSpeed = document.getElementById("metric-speed") as HTMLSpanElement;

  const historyTbody = document.getElementById("history-tbody") as HTMLTableSectionElement;
  const historyCount = document.getElementById("history-count") as HTMLSpanElement;

  const chkTogglePreview = document.getElementById("chk-toggle-preview") as HTMLInputElement;
  const previewDisabledPlaceholder = document.getElementById("preview-disabled-placeholder") as HTMLDivElement;
  const liveTag = document.getElementById("live-tag") as HTMLSpanElement;

  let isWindowFocused = true;

  function updatePreviewState() {
    const isEnabled = isWindowFocused && (chkTogglePreview ? chkTogglePreview.checked : true);
    invoke("set_preview_enabled", { enabled: isEnabled });

    if (previewDisabledPlaceholder) {
      if (isEnabled) {
        previewDisabledPlaceholder.style.display = "none";
        if (imgScanPreview) imgScanPreview.style.opacity = "1";
        if (liveTag) liveTag.style.display = "inline-block";
      } else {
        previewDisabledPlaceholder.style.display = "flex";
        if (imgScanPreview) imgScanPreview.style.opacity = "0.15";
        if (liveTag) liveTag.style.display = "none";
      }
    }
  }

  if (chkTogglePreview) {
    chkTogglePreview.addEventListener("change", updatePreviewState);
  }

  if (sliderAutocorrectThreshold && valAutocorrectThreshold) {
    sliderAutocorrectThreshold.addEventListener("input", () => {
      const val = parseInt(sliderAutocorrectThreshold.value, 10);
      valAutocorrectThreshold.textContent = `${val} %`;
      invoke("set_autocorrect_threshold", { thresholdPct: val });
    });
  }

  if (sliderWordRejectThreshold && valWordRejectThreshold) {
    sliderWordRejectThreshold.addEventListener("input", () => {
      const val = parseInt(sliderWordRejectThreshold.value, 10);
      valWordRejectThreshold.textContent = `${val} %`;
      invoke("set_word_reject_threshold", { thresholdPct: val });
    });
  }

  window.addEventListener("focus", () => {
    isWindowFocused = true;
    updatePreviewState();
  });

  window.addEventListener("blur", () => {
    isWindowFocused = false;
    updatePreviewState();
  });

  let historyItemsCount = 0;

  // 1. Fetch available capture targets
  async function refreshTargets() {
    try {
      const targets = await invoke<CaptureTargetInfo[]>("get_targets");
      targetSelect.innerHTML = "";

      if (targets.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "Brak dostępnych okien/ekranów";
        targetSelect.appendChild(opt);
        return;
      }

      targets.forEach((target) => {
        const opt = document.createElement("option");
        opt.value = target.id;
        opt.textContent = target.title;
        targetSelect.appendChild(opt);
      });

      if (targets.length > 0) {
        await invoke("set_target", { targetId: targets[0].id });
      }
    } catch (e) {
      console.error("Failed to load targets:", e);
    }
  }

  btnRefreshTargets.addEventListener("click", refreshTargets);
  await refreshTargets();

  targetSelect.addEventListener("change", async () => {
    await invoke("set_target", { targetId: targetSelect.value });
  });

  // 2. Crop Sliders & Box Model Handling
  function updateCropParams() {
    let cutTop = parseInt(sliderCropCutTop.value, 10) || 0;
    let cutBottom = parseInt(sliderCropCutBottom.value, 10) || 0;
    let cutLeft = parseInt(sliderCropCutLeft.value, 10) || 0;
    let cutRight = parseInt(sliderCropCutRight.value, 10) || 0;

    // Enforce max 95% total vertical / horizontal cuts so at least 5% active area remains
    if (cutTop + cutBottom > 95) {
      cutBottom = Math.max(0, 95 - cutTop);
      if (cutBottom === 0) cutTop = 95;
      sliderCropCutBottom.value = cutBottom.toString();
      sliderCropCutTop.value = cutTop.toString();
    }

    if (cutLeft + cutRight > 95) {
      cutRight = Math.max(0, 95 - cutLeft);
      if (cutRight === 0) cutLeft = 95;
      sliderCropCutRight.value = cutRight.toString();
      sliderCropCutLeft.value = cutLeft.toString();
    }

    const activeWidthPct = 100 - cutLeft - cutRight;
    const activeHeightPct = 100 - cutTop - cutBottom;

    // Update slider badges
    if (valCropCutTop) valCropCutTop.textContent = `${cutTop} %`;
    if (valCropCutBottom) valCropCutBottom.textContent = `${cutBottom} %`;
    if (valCropCutLeft) valCropCutLeft.textContent = `${cutLeft} %`;
    if (valCropCutRight) valCropCutRight.textContent = `${cutRight} %`;

    // Update Box Model visual overlays & active kadr box
    if (boxOverlayTop) boxOverlayTop.style.height = `${cutTop}%`;
    if (boxValTop) boxValTop.textContent = `${cutTop}%`;

    if (boxOverlayBottom) boxOverlayBottom.style.height = `${cutBottom}%`;
    if (boxValBottom) boxValBottom.textContent = `${cutBottom}%`;

    if (boxOverlayLeft) boxOverlayLeft.style.width = `${cutLeft}%`;
    if (boxValLeft) boxValLeft.textContent = `${cutLeft}%`;

    if (boxOverlayRight) boxOverlayRight.style.width = `${cutRight}%`;
    if (boxValRight) boxValRight.textContent = `${cutRight}%`;

    if (boxActiveKadr) {
      boxActiveKadr.style.top = `${cutTop}%`;
      boxActiveKadr.style.bottom = `${cutBottom}%`;
      boxActiveKadr.style.left = `${cutLeft}%`;
      boxActiveKadr.style.right = `${cutRight}%`;
    }

    if (boxActiveDims) {
      boxActiveDims.textContent = `${activeWidthPct}% × ${activeHeightPct}%`;
    }

    if (previewCropInfo) {
      previewCropInfo.textContent = `Obszar: ${activeWidthPct}% × ${activeHeightPct}% (G:${cutTop}% D:${cutBottom}%)`;
    }

    // Convert cut margins to Rust crop params (top_pct, height_pct, left_pct, width_pct)
    const topPct = cutTop / 100.0;
    const heightPct = activeHeightPct / 100.0;
    const leftPct = cutLeft / 100.0;
    const widthPct = activeWidthPct / 100.0;

    invoke("set_crop", {
      topPct,
      heightPct,
      leftPct,
      widthPct,
    });
  }

  if (sliderCropCutTop) sliderCropCutTop.addEventListener("input", updateCropParams);
  if (sliderCropCutBottom) sliderCropCutBottom.addEventListener("input", updateCropParams);
  if (sliderCropCutLeft) sliderCropCutLeft.addEventListener("input", updateCropParams);
  if (sliderCropCutRight) sliderCropCutRight.addEventListener("input", updateCropParams);

  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const top = btn.dataset.top;
      const bottom = btn.dataset.bottom;
      const left = btn.dataset.left;
      const right = btn.dataset.right;

      if (top !== undefined && sliderCropCutTop) sliderCropCutTop.value = top;
      if (bottom !== undefined && sliderCropCutBottom) sliderCropCutBottom.value = bottom;
      if (left !== undefined && sliderCropCutLeft) sliderCropCutLeft.value = left;
      if (right !== undefined && sliderCropCutRight) sliderCropCutRight.value = right;

      updateCropParams();
    });
  });

  // Initial update of crop Box Model visual state
  updateCropParams();

  // 3. Scan Interval & Frame Diff Threshold & OCR Multi-Filter Checkboxes
  sliderInterval.addEventListener("input", () => {
    const val = parseInt(sliderInterval.value, 10);
    valInterval.textContent = `${val} ms`;
    invoke("set_scan_interval", { intervalMs: val });
  });

  if (sliderFrameDiffThreshold) {
    sliderFrameDiffThreshold.addEventListener("input", () => {
      const val = parseInt(sliderFrameDiffThreshold.value, 10);
      let label = `${val} bit`;
      if (val === 0) label = "0 (Wyłączone - Skanuj każdą)";
      else if (val === 1) label = "1 (Ultra czuły - Czarny ekran)";
      else if (val === 2) label = "2 (Domyślnie wysoka)";
      else if (val >= 5) label = `${val} (Niska czułość)`;

      if (valFrameDiffThreshold) valFrameDiffThreshold.textContent = label;
      invoke("set_frame_diff_threshold", { threshold: val });
    });
  }

  function updateActiveFilters() {
    const activeFilters: string[] = [];
    chkOcrFilters.forEach((chk) => {
      if (chk.checked) {
        activeFilters.push(chk.value);
      }
    });
    invoke("set_ocr_filters", { filters: activeFilters });
  }

  chkOcrFilters.forEach((chk) => {
    chk.addEventListener("change", updateActiveFilters);
  });

  // 4. TTS Controls & Voice Downloading Status
  async function refreshVoicesList() {
    try {
      const voices = await invoke<VoiceInfo[]>("get_tts_voices");
      const currentSelected = selectVoice ? selectVoice.value : "piper_jarvis";
      if (selectVoice) {
        selectVoice.innerHTML = "";
        voices.forEach((v) => {
          const opt = document.createElement("option");
          opt.value = v.key;
          opt.textContent = v.is_downloaded
            ? `✅ ${v.name}`
            : `☁️ ${v.name} (Do pobrania)`;
          if (v.key === currentSelected) {
            opt.selected = true;
          }
          selectVoice.appendChild(opt);
        });
      }
    } catch (e) {
      console.error("Failed to load TTS voices info:", e);
    }
  }

  await refreshVoicesList();

  selectVoice.addEventListener("change", () => {
    invoke("set_tts_voice", { voiceKey: selectVoice.value });
  });

  await listen<TtsDownloadProgressPayload>("tts_download_progress", (event) => {
    const payload = event.payload;
    if (!ttsDownloadStatus) return;

    if (payload.status === "downloading") {
      ttsDownloadStatus.style.display = "flex";
      const downloadedMB = (payload.downloaded_bytes / (1024 * 1024)).toFixed(1);
      const totalMB = payload.total_bytes > 0 ? (payload.total_bytes / (1024 * 1024)).toFixed(1) : "?";
      if (ttsStatusText) ttsStatusText.textContent = `☁️ Pobieranie: ${downloadedMB} / ${totalMB} MB`;
      if (ttsStatusPct) ttsStatusPct.textContent = `${Math.round(payload.pct)} %`;
      if (ttsProgressBarFill) ttsProgressBarFill.style.width = `${payload.pct}%`;
    } else if (payload.status === "extracting") {
      ttsDownloadStatus.style.display = "flex";
      if (ttsStatusText) ttsStatusText.textContent = "📦 Rozpakowywanie archiwum modelu...";
      if (ttsStatusPct) ttsStatusPct.textContent = "99 %";
      if (ttsProgressBarFill) ttsProgressBarFill.style.width = "99%";
    } else if (payload.status === "loading_into_memory") {
      ttsDownloadStatus.style.display = "flex";
      if (ttsStatusText) ttsStatusText.textContent = "⚡ Inicjalizacja w pamięci GPU/CPU...";
      if (ttsStatusPct) ttsStatusPct.textContent = "100 %";
      if (ttsProgressBarFill) ttsProgressBarFill.style.width = "100%";
    } else if (payload.status === "ready") {
      ttsDownloadStatus.style.display = "flex";
      if (ttsStatusText) ttsStatusText.textContent = "✅ Głos gotowy do użycia!";
      if (ttsStatusPct) ttsStatusPct.textContent = "100 %";
      if (ttsProgressBarFill) ttsProgressBarFill.style.width = "100%";
      setTimeout(() => {
        if (ttsDownloadStatus) ttsDownloadStatus.style.display = "none";
      }, 2500);
      refreshVoicesList();
    } else if (payload.status === "error") {
      ttsDownloadStatus.style.display = "flex";
      if (ttsStatusText) ttsStatusText.textContent = `❌ Błąd: ${payload.error_msg || "Nieudane pobieranie"}`;
      if (ttsStatusPct) ttsStatusPct.textContent = "ERR";
    }
  });

  sliderSpeed.addEventListener("input", () => {
    const speed = parseFloat(sliderSpeed.value);
    valSpeed.textContent = `${speed.toFixed(1)} x`;
    invoke("set_tts_speed", { speed });
  });

  sliderVolume.addEventListener("input", () => {
    const volPct = parseInt(sliderVolume.value, 10);
    if (volPct > 100) {
      valVolume.textContent = `${volPct}% (🔊 Wzmocnienie ${(volPct / 100).toFixed(1)}x)`;
    } else {
      valVolume.textContent = `${volPct} %`;
    }
    invoke("set_tts_volume", { volume: volPct / 100.0 });
  });

  // 5. Pause / Start Button
  btnPause.addEventListener("click", async () => {
    const isNowPaused = await invoke<boolean>("toggle_pause");
    if (isNowPaused) {
      btnPause.classList.add("paused");
      pauseText.textContent = "▶️ START SKANOWANIA";
    } else {
      btnPause.classList.remove("paused");
      pauseText.textContent = "⏸️ PAUZA SKANOWANIA";
    }
  });

  // 6. Tauri Event Listeners
  await listen<{ preview_base64: string }>("scan_preview", (event) => {
    if (event.payload && event.payload.preview_base64) {
      imgScanPreview.src = event.payload.preview_base64;
    }
  });

  await listen<TelemetryData>("telemetry_update", (event) => {
    const data = event.payload;

    if (metricCapTime) metricCapTime.textContent = `${data.capture_time_ms.toFixed(1)} ms`;
    if (metricFilterTime) metricFilterTime.textContent = `${data.filter_time_ms.toFixed(1)} ms`;
    if (metricOcrTime) metricOcrTime.textContent = `${data.ocr_time_ms.toFixed(1)} ms`;
    if (metricTtsTime) metricTtsTime.textContent = `${data.tts_time_ms.toFixed(1)} ms`;

    metricSpeed.textContent = `${data.effective_speed.toFixed(1)}x (${data.audio_queue_len} q)`;

    if (data.audio_queue_len > 1) {
      metricSpeed.style.color = "#f43f5e";
    } else {
      metricSpeed.style.color = "#06b6d4";
    }

    if (data.ocr_loaded) {
      badgeOcr.textContent = "OCR: GOTOWY";
      badgeOcr.classList.remove("loading");
      badgeOcr.classList.add("ready");
    } else {
      badgeOcr.textContent = "OCR: Ładowanie...";
      badgeOcr.classList.remove("ready");
      badgeOcr.classList.add("loading");
    }

    if (data.tts_loaded) {
      badgeTts.textContent = "TTS: GOTOWY";
      badgeTts.classList.remove("loading");
      badgeTts.classList.add("ready");
    } else {
      badgeTts.textContent = "TTS: Ładowanie...";
      badgeTts.classList.remove("ready");
      badgeTts.classList.add("loading");
    }
  });

  await listen<SubtitleHistoryEntry>("ocr_result", (event) => {
    const entry = event.payload;

    const confPct = Math.round(entry.confidence * 1000) / 10;

    activeTimestamp.textContent = entry.timestamp;
    textRaw.textContent = entry.raw_text;

    // Render word chips with confidence sub-label in active subtitle box
    if (entry.words && entry.words.length > 0) {
      textCorrected.innerHTML = `
        <div class="active-word-chips">
          ${entry.words
            .map((w) => {
              const wConfPct = Math.round(w.confidence * 100);
              const statusClass = w.is_corrected
                ? "corrected"
                : w.confidence >= 0.95
                ? "high-conf"
                : "low-conf";
              return `
                <span class="word-chip ${statusClass}">
                  <span class="word-text">${escapeHtml(w.corrected)}</span>
                  <sub class="word-sub-conf">${wConfPct}%</sub>
                </span>
              `;
            })
            .join("")}
        </div>
      `;
    } else {
      textCorrected.textContent = entry.corrected_text;
    }

    if (activeConfidence) {
      activeConfidence.textContent = `Pewność: ${confPct}%`;
      if (confPct >= 95) {
        activeConfidence.style.background = "rgba(16, 185, 129, 0.15)";
        activeConfidence.style.color = "#6ee7b7";
        activeConfidence.style.borderColor = "rgba(16, 185, 129, 0.3)";
      } else {
        activeConfidence.style.background = "rgba(245, 158, 11, 0.15)";
        activeConfidence.style.color = "#fcd34d";
        activeConfidence.style.borderColor = "rgba(245, 158, 11, 0.3)";
      }
    }

    if (metricCapTime) metricCapTime.textContent = `${entry.capture_time_ms.toFixed(1)} ms`;
    if (metricFilterTime) metricFilterTime.textContent = `${entry.filter_time_ms.toFixed(1)} ms`;
    if (metricOcrTime) metricOcrTime.textContent = `${entry.ocr_time_ms.toFixed(1)} ms`;
    if (metricTtsTime) metricTtsTime.textContent = `${entry.tts_time_ms.toFixed(1)} ms`;

    const emptyRow = historyTbody.querySelector(".empty-row");
    if (emptyRow) {
      emptyRow.remove();
    }

    // Format subtitle text for history row (showing strikethrough for low-confidence corrected words)
    let formattedSubtitleHtml = "";
    if (entry.words && entry.words.length > 0) {
      formattedSubtitleHtml = entry.words
        .map((w) => {
          const wConfPct = Math.round(w.confidence * 100);
          if (w.is_corrected) {
            return `<del class="strikethrough-raw">${escapeHtml(w.raw)}</del><ins class="corrected-val">${escapeHtml(w.corrected)}</ins> <small class="word-conf-tag">(${wConfPct}%)</small>`;
          } else {
            return escapeHtml(w.corrected);
          }
        })
        .join(" ");
    } else {
      formattedSubtitleHtml = escapeHtml(entry.corrected_text);
    }

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="time-cell">${entry.timestamp}</td>
      <td class="corr-text">${formattedSubtitleHtml}</td>
      <td class="stats-tag">${Math.round(entry.capture_time_ms)}m/${Math.round(entry.ocr_time_ms)}m</td>
    `;

    historyTbody.insertBefore(tr, historyTbody.firstChild);
    historyItemsCount += 1;
    historyCount.textContent = `${historyItemsCount} wpisów`;

    // Cap DOM nodes at 100 elements to prevent UI lag and memory bloat over long sessions
    if (historyTbody.children.length > 100) {
      historyTbody.lastElementChild?.remove();
    }
  });
});

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
