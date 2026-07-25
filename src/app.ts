import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface CaptureTargetInfo {
  id: string;
  title: string;
  is_window: boolean;
}

interface SubtitleHistoryEntry {
  id: number;
  timestamp: string;
  raw_text: string;
  corrected_text: string;
  confidence: number;
  capture_time_ms: number;
  ocr_time_ms: number;
  tts_time_ms: number;
  boxes_passed: number;
  boxes_filtered: number;
  status: string;
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
  ocr_time_ms: number;
  tts_time_ms: number;
  engine_status: string;
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

  const imgScanPreview = document.getElementById("img-scan-preview") as HTMLImageElement;
  const previewCropInfo = document.getElementById("preview-crop-info") as HTMLSpanElement;
  const activeTimestamp = document.getElementById("active-timestamp") as HTMLSpanElement;
  const textCorrected = document.getElementById("text-corrected") as HTMLDivElement;
  const textRaw = document.getElementById("text-raw") as HTMLSpanElement;

  const metricCapTime = document.getElementById("metric-cap-time") as HTMLSpanElement;
  const metricOcrTime = document.getElementById("metric-ocr-time") as HTMLSpanElement;
  const metricTtsTime = document.getElementById("metric-tts-time") as HTMLSpanElement;
  const metricSpeed = document.getElementById("metric-speed") as HTMLSpanElement;

  const historyTbody = document.getElementById("history-tbody") as HTMLTableSectionElement;
  const historyCount = document.getElementById("history-count") as HTMLSpanElement;

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

    metricCapTime.textContent = `${data.capture_time_ms.toFixed(1)} ms`;
    metricOcrTime.textContent = `${data.ocr_time_ms.toFixed(1)} ms`;
    metricTtsTime.textContent = `${data.tts_time_ms.toFixed(1)} ms`;

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

    activeTimestamp.textContent = entry.timestamp;
    textCorrected.textContent = entry.corrected_text;
    textRaw.textContent = entry.raw_text;

    metricCapTime.textContent = `${entry.capture_time_ms.toFixed(1)} ms`;
    metricOcrTime.textContent = `${entry.ocr_time_ms.toFixed(1)} ms`;
    metricTtsTime.textContent = `${entry.tts_time_ms.toFixed(1)} ms`;

    const emptyRow = historyTbody.querySelector(".empty-row");
    if (emptyRow) {
      emptyRow.remove();
    }

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="time-cell">${entry.timestamp}</td>
      <td class="raw-text">${escapeHtml(entry.raw_text)}</td>
      <td class="corr-text">${escapeHtml(entry.corrected_text)}</td>
      <td class="stats-tag">${Math.round(entry.capture_time_ms)}m/${Math.round(entry.ocr_time_ms)}m</td>
    `;

    historyTbody.insertBefore(tr, historyTbody.firstChild);
    historyItemsCount += 1;
    historyCount.textContent = `${historyItemsCount} wpisów`;
  });
});

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
