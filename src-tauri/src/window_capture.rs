//! Window & Monitor screen capture using pure Rust xcap & imageops (Sub-millisecond cropping, zero OpenCV DLLs).

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{imageops, DynamicImage, ImageFormat, RgbaImage};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::time::Instant;
use xcap::{Monitor, Window};

use crate::ocr_engine::apply_ocr_filters;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureTargetInfo {
    pub id: String,
    pub title: String,
    pub is_window: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CropRegion {
    pub top_pct: f32,    // 0.0 to 1.0 (default 0.80 for bottom 20%)
    pub height_pct: f32, // 0.0 to 1.0 (default 0.20)
    pub left_pct: f32,   // 0.0 to 1.0 (default 0.00)
    pub width_pct: f32,  // 0.0 to 1.0 (default 1.00)
}

impl Default for CropRegion {
    fn default() -> Self {
        Self {
            top_pct: 0.80,
            height_pct: 0.20,
            left_pct: 0.00,
            width_pct: 1.00,
        }
    }
}

pub struct FrameCaptureResult {
    pub filtered_image: DynamicImage,
    pub preview_base64: String,
    pub duration_ms: f64,
}

/// Enumerate available system windows and display monitors.
pub fn get_available_targets() -> Result<Vec<CaptureTargetInfo>> {
    let mut targets = Vec::new();

    // 1. Monitors
    if let Ok(monitors) = Monitor::all() {
        for (idx, mon) in monitors.iter().enumerate() {
            let name = mon.name();
            let display_name = if name.is_empty() {
                format!("Ekran {}", idx + 1)
            } else {
                name.to_string()
            };
            targets.push(CaptureTargetInfo {
                id: format!("mon_{}", idx),
                title: format!("[Ekran] {}", display_name),
                is_window: false,
            });
        }
    }

    // 2. Windows
    if let Ok(windows) = Window::all() {
        for win in windows {
            let title = win.title();
            if !title.trim().is_empty() && !title.contains("Rust Reader") {
                let app_name = win.app_name();
                let display_title = if !app_name.is_empty() {
                    format!("[Okno] {} ({})", title, app_name)
                } else {
                    format!("[Okno] {}", title)
                };
                targets.push(CaptureTargetInfo {
                    id: format!("win_{}", win.id()),
                    title: display_title,
                    is_window: true,
                });
            }
        }
    }

    Ok(targets)
}

/// Capture frame from target, crop, apply active multi-filters (including 20px padding) & generate live preview.
pub fn capture_and_crop(
    target_id: &str,
    crop: &CropRegion,
    filters: &[String],
    generate_preview: bool,
) -> Result<FrameCaptureResult> {
    let cap_start = Instant::now();

    let rgba = capture_raw_target(target_id)?;
    let img_w = rgba.width();
    let img_h = rgba.height();

    if img_w == 0 || img_h == 0 {
        return Err(anyhow!("Empty frame captured"));
    }

    // Compute crop pixel bounds
    let x = ((img_w as f32 * crop.left_pct.clamp(0.0, 1.0)) as u32).min(img_w - 1);
    let y = ((img_h as f32 * crop.top_pct.clamp(0.0, 1.0)) as u32).min(img_h - 1);
    let w = ((img_w as f32 * crop.width_pct.clamp(0.01, 1.0)) as u32).min(img_w - x).max(10);
    let h = ((img_h as f32 * crop.height_pct.clamp(0.01, 1.0)) as u32).min(img_h - y).max(10);

    // Fast pure Rust crop
    let cropped_sub = imageops::crop_imm(&rgba, x, y, w, h);
    let cropped_dyn = DynamicImage::ImageRgba8(cropped_sub.to_image());

    // Apply active multi-filter chain (padding, contrast, binarize, sharpen, upscale)
    let filtered_dyn = apply_ocr_filters(&cropped_dyn, filters);

    let preview_base64 = if generate_preview {
        let preview_thumb = if filtered_dyn.width() > 520 {
            let th_h = (520 * filtered_dyn.height()) / filtered_dyn.width();
            filtered_dyn.resize(520, th_h.max(10), imageops::FilterType::Nearest)
        } else {
            filtered_dyn.clone()
        };

        let mut jpeg_bytes = Vec::new();
        let mut cursor = Cursor::new(&mut jpeg_bytes);
        let _ = preview_thumb.write_to(&mut cursor, ImageFormat::Jpeg);

        format!("data:image/jpeg;base64,{}", STANDARD.encode(&jpeg_bytes))
    } else {
        String::new()
    };

    let duration_ms = cap_start.elapsed().as_secs_f64() * 1000.0;

    Ok(FrameCaptureResult {
        filtered_image: filtered_dyn,
        preview_base64,
        duration_ms,
    })
}

fn capture_raw_target(target_id: &str) -> Result<RgbaImage> {
    let monitors = Monitor::all().map_err(|e| anyhow!("Failed listing monitors: {:?}", e))?;

    if target_id.starts_with("mon_") {
        let idx: usize = target_id.trim_start_matches("mon_").parse().unwrap_or(0);
        let mon = monitors
            .get(idx)
            .or_else(|| monitors.first())
            .ok_or_else(|| anyhow!("No monitors available"))?;
        let img = mon.capture_image().map_err(|e| anyhow!("Capture monitor error: {:?}", e))?;
        Ok(img)
    } else if target_id.starts_with("win_") {
        let win_id_str = target_id.trim_start_matches("win_");
        if let Ok(windows) = Window::all() {
            for win in windows {
                if win.id().to_string() == win_id_str {
                    if let Ok(img) = win.capture_image() {
                        return Ok(img);
                    }
                }
            }
        }
        let mon = monitors.first().ok_or_else(|| anyhow!("No monitors available"))?;
        let img = mon.capture_image().map_err(|e| anyhow!("Capture primary monitor fallback error: {:?}", e))?;
        Ok(img)
    } else {
        let mon = monitors.first().ok_or_else(|| anyhow!("No monitors available"))?;
        let img = mon.capture_image().map_err(|e| anyhow!("Capture primary monitor error: {:?}", e))?;
        Ok(img)
    }
}
