import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface CaptureTargetInfo {
  id: string;
  title: string;
  is_window: boolean;
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

  const chkOcrFilters = document.querySelectorAll<HTMLInputElement>(".chk-ocr-filter");

  const sliderCropTop = document.getElementById("slider-crop-top") as HTMLInputElement;
  const valCropTop = document.getElementById("val-crop-top") as HTMLSpanElement;

  const sliderCropHeight = document.getElementById("slider-crop-height") as HTMLInputElement;
  const valCropHeight = document.getElementById("val-crop-height") as HTMLSpanElement;

  const sliderCropLeft = document.getElementById("slider-crop-left") as HTMLInputElement;
  const valCropLeft = document.getElementById("val-crop-left") as HTMLSpanElement;

  const sliderCropWidth = document.getElementById("slider-crop-width") as HTMLInputElement;
  const valCropWidth = document.getElementById("val-crop-width") as HTMLSpanElement;

  const selectVoice = document.getElementById("select-voice") as HTMLSelectElement;
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

  // 2. Crop Sliders Handling
  function updateCropParams() {
    const topPct = parseFloat(sliderCropTop.value) / 100.0;
    const heightPct = parseFloat(sliderCropHeight.value) / 100.0;
    const leftPct = parseFloat(sliderCropLeft.value) / 100.0;
    const widthPct = parseFloat(sliderCropWidth.value) / 100.0;

    valCropTop.textContent = `${sliderCropTop.value} %`;
    valCropHeight.textContent = `${sliderCropHeight.value} %`;
    valCropLeft.textContent = `${sliderCropLeft.value} %`;
    valCropWidth.textContent = `${sliderCropWidth.value} %`;

    previewCropInfo.textContent = `Obszar: Top ${sliderCropTop.value}%, H ${sliderCropHeight.value}%`;

    invoke("set_crop", {
      topPct,
      heightPct,
      leftPct,
      widthPct,
    });
  }

  sliderCropTop.addEventListener("input", updateCropParams);
  sliderCropHeight.addEventListener("input", updateCropParams);
  sliderCropLeft.addEventListener("input", updateCropParams);
  sliderCropWidth.addEventListener("input", updateCropParams);

  // 3. Scan Interval & OCR Multi-Filter Checkboxes
  sliderInterval.addEventListener("input", () => {
    const val = parseInt(sliderInterval.value, 10);
    valInterval.textContent = `${val} ms`;
    invoke("set_scan_interval", { intervalMs: val });
  });

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

  // 4. TTS Controls
  selectVoice.addEventListener("change", () => {
    invoke("set_tts_voice", { voiceKey: selectVoice.value });
  });

  sliderSpeed.addEventListener("input", () => {
    const speed = parseFloat(sliderSpeed.value);
    valSpeed.textContent = `${speed.toFixed(1)} x`;
    invoke("set_tts_speed", { speed });
  });

  sliderVolume.addEventListener("input", () => {
    const volPct = parseInt(sliderVolume.value, 10);
    valVolume.textContent = `${volPct} %`;
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
