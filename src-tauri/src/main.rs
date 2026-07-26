// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app_state;
mod audio_player;
mod box_filter;
mod diff_filter;
mod ocr_engine;
mod tts_engine;
mod window_capture;

use app_state::AppState;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::State;
use window_capture::{get_available_targets, CaptureTargetInfo};

#[tauri::command]
fn get_targets() -> Result<Vec<CaptureTargetInfo>, String> {
    get_available_targets().map_err(|e| e.to_string())
}

#[tauri::command]
fn set_target(state: State<'_, Arc<AppState>>, target_id: String) -> Result<(), String> {
    *state.target_id.lock() = target_id;
    Ok(())
}

#[tauri::command]
fn set_crop(
    state: State<'_, Arc<AppState>>,
    top_pct: f32,
    height_pct: f32,
    left_pct: f32,
    width_pct: f32,
) -> Result<(), String> {
    let mut crop = state.crop_region.lock();
    crop.top_pct = top_pct;
    crop.height_pct = height_pct;
    crop.left_pct = left_pct;
    crop.width_pct = width_pct;
    Ok(())
}

#[tauri::command]
fn set_ocr_filters(state: State<'_, Arc<AppState>>, filters: Vec<String>) -> Result<(), String> {
    *state.ocr_filters.lock() = filters;
    Ok(())
}

#[tauri::command]
fn set_scan_interval(state: State<'_, Arc<AppState>>, interval_ms: u16) -> Result<(), String> {
    let interval_ms = interval_ms.clamp(100, 1000);
    state.scan_interval_ms.store(interval_ms, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
fn get_tts_voices() -> Result<Vec<tts_engine::VoiceInfo>, String> {
    Ok(tts_engine::get_voices_info())
}

#[tauri::command]
fn set_tts_voice(app_handle: tauri::AppHandle, state: State<'_, Arc<AppState>>, voice_key: String) -> Result<(), String> {
    *state.tts_voice.lock() = voice_key.clone();
    let state_clone = Arc::clone(&state);
    let handle_clone = app_handle.clone();
    std::thread::spawn(move || {
        let voice_name = tts_engine::VOICES
            .iter()
            .find(|v| v.key == voice_key)
            .map(|v| v.name)
            .unwrap_or(&voice_key);

        let model_dir = tts_engine::get_models_base_dir().join(&voice_key);
        let is_already_downloaded = tts_engine::is_voice_downloaded(&model_dir);

        if !is_already_downloaded {
            use tauri::Emitter;
            let _ = handle_clone.emit(
                "tts_download_progress",
                tts_engine::TtsDownloadProgressPayload {
                    voice_key: voice_key.clone(),
                    voice_name: voice_name.to_string(),
                    downloaded_bytes: 0,
                    total_bytes: 0,
                    pct: 0.0,
                    status: "downloading".to_string(),
                    error_msg: None,
                },
            );
        }

        let tts_guard = state_clone.tts_engine.lock();
        if let Some(ref tts) = *tts_guard {
            if let Err(e) = tts.load_voice(&voice_key, Some(&handle_clone)) {
                eprintln!("Failed to load voice {}: {}", voice_key, e);
                use tauri::Emitter;
                let _ = handle_clone.emit(
                    "tts_download_progress",
                    tts_engine::TtsDownloadProgressPayload {
                        voice_key: voice_key.clone(),
                        voice_name: voice_name.to_string(),
                        downloaded_bytes: 0,
                        total_bytes: 0,
                        pct: 0.0,
                        status: "error".to_string(),
                        error_msg: Some(e.to_string()),
                    },
                );
            }
        }
    });
    Ok(())
}

#[tauri::command]
fn set_tts_speed(state: State<'_, Arc<AppState>>, speed: f32) -> Result<(), String> {
    let speed = speed.clamp(0.5, 3.0);
    *state.tts_speed.lock() = speed;
    Ok(())
}

#[tauri::command]
fn set_tts_volume(state: State<'_, Arc<AppState>>, volume: f32) -> Result<(), String> {
    let volume = volume.clamp(0.0, 1.0);
    *state.tts_volume.lock() = volume;
    let player_guard = state.audio_player.lock();
    if let Some(ref player) = *player_guard {
        player.set_volume(volume);
    }
    Ok(())
}

#[tauri::command]
fn toggle_pause(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    let current = state.is_paused.load(Ordering::Relaxed);
    let new_val = !current;
    state.is_paused.store(new_val, Ordering::Relaxed);

    let player_guard = state.audio_player.lock();
    if let Some(ref player) = *player_guard {
        player.set_paused(new_val);
    }

    Ok(new_val)
}

#[tauri::command]
fn set_preview_enabled(state: State<'_, Arc<AppState>>, enabled: bool) -> Result<(), String> {
    state.is_preview_enabled.store(enabled, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
fn set_autocorrect_threshold(state: State<'_, Arc<AppState>>, threshold_pct: f32) -> Result<(), String> {
    let threshold = (threshold_pct / 100.0).clamp(0.0, 1.0);
    *state.autocorrect_threshold.lock() = threshold;
    Ok(())
}

#[tauri::command]
fn set_word_reject_threshold(state: State<'_, Arc<AppState>>, threshold_pct: f32) -> Result<(), String> {
    let threshold = (threshold_pct / 100.0).clamp(0.0, 1.0);
    *state.word_reject_threshold.lock() = threshold;
    Ok(())
}

#[tauri::command]
fn set_frame_diff_threshold(state: State<'_, Arc<AppState>>, threshold: u8) -> Result<(), String> {
    let threshold = threshold.clamp(0, 20);
    *state.frame_diff_threshold.lock() = threshold;
    Ok(())
}

fn main() {
    let app_state = Arc::new(AppState::new());
    let state_for_worker = Arc::clone(&app_state);

    tauri::Builder::default()
        .manage(app_state)
        .setup(move |app| {
            state_for_worker.start_worker_thread(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_targets,
            set_target,
            set_crop,
            set_ocr_filters,
            set_scan_interval,
            set_preview_enabled,
            set_autocorrect_threshold,
            set_word_reject_threshold,
            set_frame_diff_threshold,
            get_tts_voices,
            set_tts_voice,
            set_tts_speed,
            set_tts_volume,
            toggle_pause
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
