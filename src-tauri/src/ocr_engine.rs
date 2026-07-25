//! OCR engine wrapper using oar-ocr and SymSpell spell correction with ultra-fast SIMD/memcpy multi-filter preprocessing.

use anyhow::{anyhow, Result};
use image::{imageops, DynamicImage, GrayImage, RgbaImage};
use oar_ocr::core::config::onnx::{OrtExecutionProvider, OrtSessionConfig};
use oar_ocr::prelude::*;
use symspell::{SymSpell, UnicodeiStringStrategy, Verbosity};
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use crate::app_state::WordInfo;
use crate::box_filter::{is_clean_text, is_valid_box_size, sanitize_text};

pub struct OcrResult {
    pub raw_text: String,
    pub corrected_text: String,
    pub confidence: f32,
    pub words: Vec<WordInfo>,
    pub duration_ms: f64,
    pub boxes_passed: usize,
    pub boxes_filtered: usize,
}

pub struct OcrEngine {
    ocr: OAROCR,
    symspell: Mutex<SymSpell<UnicodeiStringStrategy>>,
}

impl OcrEngine {
    pub fn new() -> Result<Self> {
        let dict_path = locate_or_create_dictionary("pl_50k.txt");

        let mut symspell = SymSpell::<UnicodeiStringStrategy>::default();
        if Path::new(&dict_path).exists() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                symspell.load_dictionary(&dict_path, 0, 1, " ");
            }));
        }

        let ort_config = OrtSessionConfig::new().with_execution_providers(vec![
            OrtExecutionProvider::TensorRT {
                device_id: Some(0),
                max_workspace_size: None,
                min_subgraph_size: None,
                fp16_enable: None,
                timing_cache: None,
                timing_cache_path: None,
                force_timing_cache: None,
                engine_cache: None,
                engine_cache_path: None,
                dump_ep_context_model: None,
                ep_context_file_path: None,
            },
            OrtExecutionProvider::CUDA {
                device_id: Some(0),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                cudnn_conv_use_max_workspace: None,
            },
        ]);

        let ocr = OAROCRBuilder::new(
            "pp-ocrv6_tiny_det.onnx",
            "pp-ocrv6_small_rec.onnx",
            "ppocrv6_dict.txt",
        )
        .ort_session(ort_config)
        .build()
        .map_err(|e| anyhow!("Failed to initialize OCR engine: {:?}", e))?;

        Ok(Self {
            ocr,
            symspell: Mutex::new(symspell),
        })
    }

    pub fn process_image(
        &self,
        img: &DynamicImage,
        autocorrect_threshold: f32,
        word_reject_threshold: f32,
    ) -> Result<OcrResult> {
        let start = Instant::now();
        let frame_w = img.width();
        let frame_h = img.height();

        let rgb_img = img.to_rgb8();

        let predictions = self
            .ocr
            .predict(vec![rgb_img])
            .map_err(|e| anyhow!("OCR prediction failed: {:?}", e))?;

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut raw_parts = Vec::new();
        let mut corrected_parts = Vec::new();
        let mut all_words = Vec::new();
        let mut total_conf = 0.0f32;
        let mut count = 0usize;
        let mut boxes_passed = 0usize;
        let mut boxes_filtered = 0usize;

        if let Some(res) = predictions.first() {
            for region in &res.text_regions {
                if let Some((raw_text, confidence)) = region.text_with_confidence() {
                    // Reject low-confidence OCR regions completely (e.g. confidence < 40%)
                    if confidence < word_reject_threshold {
                        boxes_filtered += 1;
                        continue;
                    }
                    let mut box_w = 50u32;
                    let mut box_h = 20u32;

                    if let Some(ref poly) = region.dt_poly {
                        let pts = &poly.points;
                        let min_x = pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
                        let max_x = pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
                        let min_y = pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
                        let max_y = pts.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);

                        box_w = (max_x - min_x).max(0.0) as u32;
                        box_h = (max_y - min_y).max(0.0) as u32;
                    }

                    // Apply box size filter
                    if !is_valid_box_size(box_w, box_h, frame_w, frame_h) {
                        boxes_filtered += 1;
                        continue;
                    }

                    // Strip invalid characters (Chinese, CJK, Emojis, weird OCR noise)
                    let clean_raw = sanitize_text(&raw_text);

                    // Apply text repetition & standalone dot/comma filter
                    if !is_clean_text(&clean_raw) {
                        boxes_filtered += 1;
                        continue;
                    }

                    boxes_passed += 1;

                    let mut line_corrected_words = Vec::new();
                    for word in clean_raw.split_whitespace() {
                        let raw_word = word.to_string();
                        let (corr_word, is_corr) = if confidence >= autocorrect_threshold {
                            (raw_word.clone(), false)
                        } else {
                            if word.len() > 2 {
                                let sym = self.symspell.lock().unwrap();
                                let suggestions = sym.lookup(word, Verbosity::Top, 2);
                                if let Some(sug) = suggestions.first() {
                                    if sug.term != raw_word {
                                        (sug.term.clone(), true)
                                    } else {
                                        (raw_word.clone(), false)
                                    }
                                } else {
                                    (raw_word.clone(), false)
                                }
                            } else {
                                (raw_word.clone(), false)
                            }
                        };

                        line_corrected_words.push(corr_word.clone());
                        all_words.push(WordInfo {
                            raw: raw_word,
                            corrected: corr_word,
                            confidence,
                            is_corrected: is_corr,
                        });
                    }

                    let corrected_line = line_corrected_words.join(" ");

                    raw_parts.push(clean_raw);
                    corrected_parts.push(corrected_line);
                    total_conf += confidence;
                    count += 1;
                }
            }
        }

        let raw_text = raw_parts.join(" ");
        let corrected_text = corrected_parts.join(" ");
        let avg_conf = if count > 0 { total_conf / count as f32 } else { 0.0 };

        Ok(OcrResult {
            raw_text,
            corrected_text,
            confidence: avg_conf,
            words: all_words,
            duration_ms,
            boxes_passed,
            boxes_filtered,
        })
    }
}

/// Ultra-fast multi-filter preprocessing pipeline using memcpy and LookUp Tables (< 1ms execution)
pub fn apply_ocr_filters(img: &DynamicImage, filters: &[String]) -> DynamicImage {
    let mut current = img.clone();

    // 1. Add 20px Padding Border (Ultra-fast row memcpy)
    if filters.iter().any(|f| f == "padding20") {
        current = add_border_padding_fast(&current, 20);
    }

    // 2. Fast 1.5x Nearest-Neighbor Upscaling for small fonts
    if filters.iter().any(|f| f == "upscale") {
        if current.height() < 160 {
            let new_w = current.width() * 3 / 2;
            let new_h = current.height() * 3 / 2;
            current = current.resize(new_w, new_h, imageops::FilterType::Nearest);
        }
    }

    // 3. Fast Linear Contrast Stretching via 256-element LookUp Table (LUT)
    if filters.iter().any(|f| f == "contrast") {
        let gray = current.to_luma8();
        let stretched = stretch_contrast_lut(&gray);
        current = DynamicImage::ImageLuma8(stretched);
    }

    // 4. Smooth Unsharp Masking (Soft edge sharpening without distortion or ringing halos)
    if filters.iter().any(|f| f == "sharpen") {
        let gray = current.to_luma8();
        let sharpened = sharpen_fast_smooth(&gray);
        current = DynamicImage::ImageLuma8(sharpened);
    }

    // 5. Fast Direct Binarization (Black & White)
    if filters.iter().any(|f| f == "binarize") {
        let gray = current.to_luma8();
        let binarized = binarize_fast(&gray, 140);
        current = DynamicImage::ImageLuma8(binarized);
    }

    current
}

/// Ultra-fast 20px padding border using raw std::slice::copy_from_slice (memcpy)
fn add_border_padding_fast(img: &DynamicImage, padding_px: u32) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (orig_w, orig_h) = rgba.dimensions();
    let new_w = orig_w + (padding_px * 2);
    let new_h = orig_h + (padding_px * 2);

    let row_bytes = (orig_w * 4) as usize;
    let new_stride = (new_w * 4) as usize;
    let pad_bytes = (padding_px * 4) as usize;

    // Fill background with dark RGBA(12, 12, 16, 255)
    let mut new_buf = vec![12u8; (new_w * new_h * 4) as usize];
    for i in (3..new_buf.len()).step_by(4) {
        new_buf[i] = 255;
    }

    let src_raw = rgba.as_raw();
    for y in 0..orig_h as usize {
        let src_offset = y * row_bytes;
        let dst_offset = (y + padding_px as usize) * new_stride + pad_bytes;
        new_buf[dst_offset..dst_offset + row_bytes].copy_from_slice(&src_raw[src_offset..src_offset + row_bytes]);
    }

    DynamicImage::ImageRgba8(RgbaImage::from_raw(new_w, new_h, new_buf).unwrap())
}

/// Linear contrast stretch using a 256-element LookUp Table (< 0.1ms)
fn stretch_contrast_lut(gray: &GrayImage) -> GrayImage {
    let pixels = gray.as_raw();
    if pixels.is_empty() { return gray.clone(); }

    let mut min_val = 255u8;
    let mut max_val = 0u8;

    for &v in pixels {
        if v < min_val { min_val = v; }
        if v > max_val { max_val = v; }
    }

    if max_val <= min_val {
        return gray.clone();
    }

    let range = (max_val - min_val) as f32;
    let mut lut = [0u8; 256];
    for i in 0..=255 {
        if i < min_val {
            lut[i as usize] = 0;
        } else if i > max_val {
            lut[i as usize] = 255;
        } else {
            lut[i as usize] = ((i as f32 - min_val as f32) / range * 255.0).clamp(0.0, 255.0) as u8;
        }
    }

    let mut out_buf = vec![0u8; pixels.len()];
    for (i, &p) in pixels.iter().enumerate() {
        out_buf[i] = lut[p as usize];
    }

    GrayImage::from_raw(gray.width(), gray.height(), out_buf).unwrap()
}

/// Smooth Unsharp Masking (Soft edge enhancement without noise distortion/ringing halos)
fn sharpen_fast_smooth(gray: &GrayImage) -> GrayImage {
    let (w, h) = gray.dimensions();
    if w < 3 || h < 3 { return gray.clone(); }

    let src = gray.as_raw();
    let mut dst = src.clone(); // Copy original image so borders stay clean!
    let w_usize = w as usize;

    for y in 1..(h as usize - 1) {
        for x in 1..(w_usize - 1) {
            let idx = y * w_usize + x;
            let center = src[idx] as i32;

            // Smooth 3x3 9-point box blur
            let sum_3x3 = src[idx - w_usize - 1] as i32 + src[idx - w_usize] as i32 + src[idx - w_usize + 1] as i32
                        + src[idx - 1] as i32 + center + src[idx + 1] as i32
                        + src[idx + w_usize - 1] as i32 + src[idx + w_usize] as i32 + src[idx + w_usize + 1] as i32;
            let blur = sum_3x3 / 9;

            // Soft Unsharp Masking: original + 0.6 * (original - blur)
            let detail = center - blur;
            let val = center + (detail * 6 / 10);
            dst[idx] = val.clamp(0, 255) as u8;
        }
    }

    GrayImage::from_raw(w, h, dst).unwrap()
}

/// Fast single-pass binary thresholding
fn binarize_fast(gray: &GrayImage, threshold: u8) -> GrayImage {
    let pixels = gray.as_raw();
    let mut out_buf = vec![0u8; pixels.len()];

    for (i, &p) in pixels.iter().enumerate() {
        out_buf[i] = if p > threshold { 255 } else { 0 };
    }

    GrayImage::from_raw(gray.width(), gray.height(), out_buf).unwrap()
}

fn locate_or_create_dictionary(filename: &str) -> String {
    if Path::new(filename).exists() && is_valid_dict_file(filename) {
        return filename.to_string();
    }

    let parent_path = format!("../{}", filename);
    if Path::new(&parent_path).exists() && is_valid_dict_file(&parent_path) {
        return parent_path;
    }

    let sample = "witaj 1000\nświat 1000\njestem 1000\ntak 1000\nnie 1000\ndobrze 1000\nbardzo 1000\nsprawdzić 1000\noglądasz 1000\nfilm 1000\nanime 1000\ngra 1000\npliki 1000\nsystem 1000\ntekst 1000\nczytaj 1000\ngrać 1000\nrazem 1000\nwszystko 1000\ndziękuję 1000\nprzepraszam 1000\ndo 1000\nwidzenia 1000\ncześć 1000\n";
    let _ = fs::write(filename, sample);
    filename.to_string()
}

fn is_valid_dict_file(filepath: &str) -> bool {
    if let Ok(content) = fs::read_to_string(filepath) {
        if let Some(first_line) = content.lines().next() {
            let parts: Vec<&str> = first_line.split_whitespace().collect();
            if parts.len() >= 2 && parts[1].parse::<u64>().is_ok() {
                return true;
            }
        }
    }
    false
}
