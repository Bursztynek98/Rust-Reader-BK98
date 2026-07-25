// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app_state;
mod audio_player;
mod box_filter;
mod diff_filter;
mod ocr_engine;
mod tts_engine;
mod window_capture;

use app_state::{AppState, SubtitleHistoryEntry};
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
fn set_scan_interval(state: State<'_, Arc<AppState>>, interval_ms: u64) -> Result<(), String> {
    let interval_ms = interval_ms.clamp(100, 1000);
    state.scan_interval_ms.store(interval_ms, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
fn set_tts_voice(state: State<'_, Arc<AppState>>, voice_key: String) -> Result<(), String> {
    *state.tts_voice.lock() = voice_key.clone();
    let state_clone = Arc::clone(&state);
    std::thread::spawn(move || {
        let tts_guard = state_clone.tts_engine.lock();
        if let Some(ref tts) = *tts_guard {
            let _ = tts.load_voice(&voice_key);
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
fn get_history(state: State<'_, Arc<AppState>>) -> Result<Vec<SubtitleHistoryEntry>, String> {
    Ok(state.history.lock().clone())
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
            set_tts_voice,
            set_tts_speed,
            set_tts_volume,
            toggle_pause,
            get_history
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
