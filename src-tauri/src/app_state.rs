//! Central application state and background OCR-TTS scanning loop thread.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Emitter};

use crate::audio_player::{AudioItem, AudioPlayer};
use crate::diff_filter::{has_image_changed, has_significant_change, FrameHash};
use crate::ocr_engine::OcrEngine;
use crate::tts_engine::TtsEngine;
use crate::window_capture::{capture_and_crop, CropRegion};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordInfo {
    pub raw: String,
    pub corrected: String,
    pub confidence: f32,
    pub is_corrected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleHistoryEntry {
    pub id: u64,
    pub timestamp: String,
    pub raw_text: String,
    pub corrected_text: String,
    pub confidence: f32,
    pub words: Vec<WordInfo>,
    pub capture_time_ms: f64,
    pub filter_time_ms: f64,
    pub ocr_time_ms: f64,
    pub tts_time_ms: f64,
    pub boxes_passed: usize,
    pub boxes_filtered: usize,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryData {
    pub scan_interval_ms: u64,
    pub is_paused: bool,
    pub audio_queue_len: usize,
    pub current_voice: String,
    pub base_speed: f32,
    pub effective_speed: f32,
    pub volume: f32,
    pub ocr_loaded: bool,
    pub tts_loaded: bool,
    pub capture_time_ms: f64,
    pub filter_time_ms: f64,
    pub ocr_time_ms: f64,
    pub tts_time_ms: f64,
    pub engine_status: String,
}

pub struct AppState {
    pub is_paused: Arc<AtomicBool>,
    pub is_preview_enabled: Arc<AtomicBool>,
    pub scan_interval_ms: Arc<AtomicU64>,
    pub autocorrect_threshold: Arc<Mutex<f32>>,
    pub word_reject_threshold: Arc<Mutex<f32>>,
    pub target_id: Arc<Mutex<String>>,
    pub crop_region: Arc<Mutex<CropRegion>>,
    pub ocr_filters: Arc<Mutex<Vec<String>>>,
    pub tts_voice: Arc<Mutex<String>>,
    pub tts_speed: Arc<Mutex<f32>>,
    pub tts_volume: Arc<Mutex<f32>>,
    pub last_spoken_text: Arc<Mutex<String>>,

    pub ocr_loaded: Arc<AtomicBool>,
    pub tts_loaded: Arc<AtomicBool>,
    pub ocr_engine: Arc<Mutex<Option<OcrEngine>>>,
    pub tts_engine: Arc<Mutex<Option<TtsEngine>>>,
    pub audio_player: Arc<Mutex<Option<AudioPlayer>>>,
    pub engine_status: Arc<Mutex<String>>,
}

impl AppState {
    pub fn new() -> Self {
        let is_paused = Arc::new(AtomicBool::new(true));
        let is_preview_enabled = Arc::new(AtomicBool::new(true));
        let scan_interval_ms = Arc::new(AtomicU64::new(300));
        let autocorrect_threshold = Arc::new(Mutex::new(0.95f32));
        let word_reject_threshold = Arc::new(Mutex::new(0.40f32));
        let target_id = Arc::new(Mutex::new("mon_0".to_string()));
        let crop_region = Arc::new(Mutex::new(CropRegion::default()));
        let ocr_filters = Arc::new(Mutex::new(vec!["padding20".to_string(), "contrast".to_string()]));
        let tts_voice = Arc::new(Mutex::new("piper_jarvis".to_string()));
        let tts_speed = Arc::new(Mutex::new(1.0f32));
        let tts_volume = Arc::new(Mutex::new(1.0f32));
        let last_spoken_text = Arc::new(Mutex::new(String::new()));
        let engine_status = Arc::new(Mutex::new("Inicjalizacja silników...".to_string()));

        let ocr_loaded = Arc::new(AtomicBool::new(false));
        let tts_loaded = Arc::new(AtomicBool::new(false));

        let audio_player = AudioPlayer::new().ok();

        Self {
            is_paused,
            is_preview_enabled,
            scan_interval_ms,
            autocorrect_threshold,
            word_reject_threshold,
            target_id,
            crop_region,
            ocr_filters,
            tts_voice,
            tts_speed,
            tts_volume,
            last_spoken_text,
            ocr_loaded,
            tts_loaded,
            ocr_engine: Arc::new(Mutex::new(None)),
            tts_engine: Arc::new(Mutex::new(None)),
            audio_player: Arc::new(Mutex::new(audio_player)),
            engine_status,
        }
    }

    pub fn start_worker_thread(self: &Arc<Self>, app_handle: AppHandle) {
        let state_for_init = Arc::clone(self);

        // 1. Initialize OCR & TTS engines asynchronously in background
        thread::spawn(move || {
            println!("[Worker] Async loading OCR engine...");
            match OcrEngine::new() {
                Ok(ocr) => {
                    *state_for_init.ocr_engine.lock() = Some(ocr);
                    state_for_init.ocr_loaded.store(true, Ordering::Relaxed);
                    println!("[Worker] ✓ OCR engine loaded.");
                }
                Err(e) => {
                    eprintln!("[Worker] ❌ OCR load error: {:?}", e);
                    *state_for_init.engine_status.lock() = format!("Błąd OCR: {:?}", e);
                }
            }

            println!("[Worker] Async loading TTS engine...");
            match TtsEngine::new("piper_jarvis") {
                Ok(tts) => {
                    *state_for_init.tts_engine.lock() = Some(tts);
                    state_for_init.tts_loaded.store(true, Ordering::Relaxed);
                    println!("[Worker] ✓ TTS engine loaded.");
                    *state_for_init.engine_status.lock() = "Model OCR & TTS Załadowany".to_string();
                }
                Err(e) => {
                    eprintln!("[Worker] ❌ TTS load error: {:?}", e);
                }
            }
        });

        // 2. Frame Capture & Subtitle Processing Loop
        let state = Arc::clone(self);
        thread::spawn(move || {
            let mut entry_counter = 0u64;
            let mut last_cap_time = 0.0f64;
            let mut last_filter_time = 0.0f64;
            let mut last_ocr_time = 0.0f64;
            let mut last_tts_time = 0.0f64;

            let mut last_frame_hash: Option<FrameHash> = None;

            loop {
                let interval = state.scan_interval_ms.load(Ordering::Relaxed).max(100);
                let paused = state.is_paused.load(Ordering::Relaxed);
                let is_preview_active = state.is_preview_enabled.load(Ordering::Relaxed);
                let is_ocr_ready = state.ocr_loaded.load(Ordering::Relaxed);
                let is_tts_ready = state.tts_loaded.load(Ordering::Relaxed);

                let target = state.target_id.lock().clone();
                let crop = state.crop_region.lock().clone();
                let active_filters = state.ocr_filters.lock().clone();

                // Capture frame, crop, apply active multi-filters & emit live preview Data URI if enabled
                if let Ok(frame_res) = capture_and_crop(&target, &crop, &active_filters, is_preview_active) {
                    last_cap_time = frame_res.duration_ms;
                    last_filter_time = frame_res.filter_duration_ms;

                    if is_preview_active && !frame_res.preview_base64.is_empty() {
                        let _ = app_handle.emit("scan_preview", serde_json::json!({
                            "preview_base64": frame_res.preview_base64
                        }));
                    }

                    if !paused && is_ocr_ready {
                        let (img_changed, new_hash) = has_image_changed(last_frame_hash, &frame_res.filtered_image);
                        if img_changed {
                            last_frame_hash = Some(new_hash);
                            let ocr_guard = state.ocr_engine.lock();
                            if let Some(ref ocr) = *ocr_guard {
                                let threshold = *state.autocorrect_threshold.lock();
                                let reject_threshold = *state.word_reject_threshold.lock();
                                if let Ok(ocr_res) = ocr.process_image(&frame_res.filtered_image, threshold, reject_threshold) {
                                    last_ocr_time = ocr_res.duration_ms;
                                    let now_str = chrono_now_str();

                                    if !ocr_res.corrected_text.is_empty() {

                                    let mut last_txt = state.last_spoken_text.lock();
                                    
                                    if has_significant_change(&last_txt, &ocr_res.corrected_text) {
                                        *last_txt = ocr_res.corrected_text.clone();
                                        let text_to_speak = ocr_res.corrected_text.clone();
                                        drop(last_txt);

                                        let qlen = state.get_audio_queue_len();
                                        let b_speed = *state.tts_speed.lock();
                                        let eff_speed = if qlen > 1 { (b_speed * 2.5).min(3.5) } else { b_speed };

                                        let tts_guard = state.tts_engine.lock();

                                        if is_tts_ready {
                                            if let Some(ref tts) = *tts_guard {
                                                if let Ok(audio) = tts.generate_speech(&text_to_speak, eff_speed) {
                                                    last_tts_time = audio.duration_ms;

                                                    let player_guard = state.audio_player.lock();
                                                    if let Some(ref player) = *player_guard {
                                                        let _ = player.enqueue(AudioItem {
                                                            text: text_to_speak.clone(),
                                                            samples: audio.samples,
                                                            sample_rate: audio.sample_rate,
                                                        });
                                                    }
                                                }
                                            }
                                        }

                                        entry_counter += 1;
                                        let entry = SubtitleHistoryEntry {
                                            id: entry_counter,
                                            timestamp: now_str,
                                            raw_text: ocr_res.raw_text,
                                            corrected_text: ocr_res.corrected_text,
                                            confidence: ocr_res.confidence,
                                            words: ocr_res.words,
                                            capture_time_ms: last_cap_time,
                                            filter_time_ms: last_filter_time,
                                            ocr_time_ms: last_ocr_time,
                                            tts_time_ms: last_tts_time,
                                            boxes_passed: ocr_res.boxes_passed,
                                            boxes_filtered: ocr_res.boxes_filtered,
                                            status: "SPOKEN".to_string(),
                                        };

                                        let _ = app_handle.emit("ocr_result", entry);
                                    }
                                }
                            }
                        }
                    }
                }
            }

                let qlen = state.get_audio_queue_len();
                let b_speed = *state.tts_speed.lock();
                let eff_speed = if qlen > 1 { (b_speed * 2.5).min(3.5) } else { b_speed };

                let current_status = if paused {
                    "WSTRZYMANE (Kliknij START)".to_string()
                } else if !is_ocr_ready || !is_tts_ready {
                    "Ładowanie modeli...".to_string()
                } else {
                    "SKANOWANIE AKTYWNE".to_string()
                };

                let _ = app_handle.emit("telemetry_update", TelemetryData {
                    scan_interval_ms: interval,
                    is_paused: paused,
                    audio_queue_len: qlen,
                    current_voice: state.tts_voice.lock().clone(),
                    base_speed: b_speed,
                    effective_speed: eff_speed,
                    volume: *state.tts_volume.lock(),
                    ocr_loaded: is_ocr_ready,
                    tts_loaded: is_tts_ready,
                    capture_time_ms: last_cap_time,
                    filter_time_ms: last_filter_time,
                    ocr_time_ms: last_ocr_time,
                    tts_time_ms: last_tts_time,
                    engine_status: current_status,
                });

                thread::sleep(Duration::from_millis(interval));
            }
        });
    }

    pub fn get_audio_queue_len(&self) -> usize {
        let player = self.audio_player.lock();
        if let Some(ref p) = *player {
            p.get_queue_len()
        } else {
            0
        }
    }
}

fn chrono_now_str() -> String {
    use std::time::SystemTime;
    if let Ok(dur) = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        let secs = dur.as_secs() % 86400;
        let hours = (secs / 3600 + 2) % 24; // CEST UTC+2
        let mins = (secs % 3600) / 60;
        let s = secs % 60;
        format!("{:02}:{:02}:{:02}", hours, mins, s)
    } else {
        "00:00:00".to_string()
    }
}
