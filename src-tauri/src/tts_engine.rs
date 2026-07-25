//! TTS Engine wrapper using sherpa-onnx for Polish speech synthesis.

use anyhow::{anyhow, Context, Result};
use bzip2::read::BzDecoder;
use reqwest::blocking::Client;
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsSupertonicModelConfig, OfflineTtsVitsModelConfig,
};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use tar::Archive;

pub struct VoiceDefinition {
    pub key: &'static str,
    pub name: &'static str,
    pub url: &'static str,
    pub is_supertonic: bool,
}

pub const VOICES: &[VoiceDefinition] = &[
    VoiceDefinition {
        key: "piper_zenski",
        name: "Polski głos żeński (Piper VITS)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-pl_PL-zenski_wg_glos-medium.tar.bz2",
        is_supertonic: false,
    },
    VoiceDefinition {
        key: "piper_meski",
        name: "Polski głos męski (Piper VITS)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-pl_PL-meski_wg_glos-medium.tar.bz2",
        is_supertonic: false,
    },
    VoiceDefinition {
        key: "piper_jarvis",
        name: "Polski głos Jarvis (Piper VITS)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-pl_PL-jarvis_wg_glos-medium.tar.bz2",
        is_supertonic: false,
    },
    VoiceDefinition {
        key: "supertonic",
        name: "Wielojęzyczny Supertonic 3 (Polski)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
        is_supertonic: true,
    },
];

pub struct GeneratedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub duration_ms: f64,
}

pub struct TtsEngine {
    active_voice_key: Mutex<String>,
    tts_instance: Mutex<Option<OfflineTts>>,
}

impl TtsEngine {
    pub fn new(default_voice: &str) -> Result<Self> {
        let voice_key = if VOICES.iter().any(|v| v.key == default_voice) {
            default_voice.to_string()
        } else {
            "piper_jarvis".to_string()
        };

        let engine = Self {
            active_voice_key: Mutex::new(voice_key.clone()),
            tts_instance: Mutex::new(None),
        };

        // Try loading initial voice
        let _ = engine.load_voice(&voice_key);
        Ok(engine)
    }

    pub fn load_voice(&self, voice_key: &str) -> Result<()> {
        let voice_def = VOICES
            .iter()
            .find(|v| v.key == voice_key)
            .ok_or_else(|| anyhow!("Unknown voice key: {}", voice_key))?;

        let models_root = PathBuf::from("models").join(voice_def.key);
        ensure_voice_model_downloaded(voice_def, &models_root)?;

        let mut model_config = OfflineTtsModelConfig {
            num_threads: 4,
            debug: false,
            ..Default::default()
        };

        if voice_def.is_supertonic {
            model_config.supertonic = find_supertonic_files(&models_root)?;
        } else {
            model_config.vits = find_vits_piper_files(&models_root)?;
        }

        // Try CUDA first, then CPU fallback
        let mut gpu_config = model_config.clone();
        gpu_config.provider = Some("cuda".to_string());
        let tts = if let Some(t) = OfflineTts::create(&OfflineTtsConfig {
            model: gpu_config,
            ..Default::default()
        }) {
            t
        } else {
            model_config.provider = Some("cpu".to_string());
            OfflineTts::create(&OfflineTtsConfig {
                model: model_config,
                ..Default::default()
            })
            .ok_or_else(|| anyhow!("Failed to create OfflineTts engine on CPU"))?
        };

        let mut lock = self.tts_instance.lock().unwrap();
        *lock = Some(tts);
        *self.active_voice_key.lock().unwrap() = voice_key.to_string();
        Ok(())
    }

    pub fn generate_speech(&self, text: &str, speed: f32) -> Result<GeneratedAudio> {
        let lock = self.tts_instance.lock().unwrap();
        let tts = lock
            .as_ref()
            .ok_or_else(|| anyhow!("TTS engine is not initialized"))?;

        let current_voice = self.active_voice_key.lock().unwrap().clone();
        let is_supertonic = current_voice == "supertonic";
        let mut extra_map = HashMap::new();
        if is_supertonic {
            extra_map.insert("lang".to_string(), serde_json::Value::String("pl".to_string()));
            extra_map.insert("language".to_string(), serde_json::Value::String("pl".to_string()));
        }

        let gen_config = GenerationConfig {
            speed,
            silence_scale: 0.2,
            extra: if is_supertonic { Some(extra_map) } else { None },
            ..Default::default()
        };

        let start = Instant::now();
        let audio = tts
            .generate_with_config(text, &gen_config, None::<fn(&[f32], f32) -> bool>)
            .ok_or_else(|| anyhow!("TTS audio synthesis failed"))?;
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(GeneratedAudio {
            samples: audio.samples().to_vec(),
            sample_rate: audio.sample_rate() as u32,
            duration_ms,
        })
    }
}

fn ensure_voice_model_downloaded(voice_def: &VoiceDefinition, model_dir: &Path) -> Result<()> {
    if model_dir.exists() && fs::read_dir(model_dir)?.next().is_some() {
        return Ok(());
    }

    println!("Downloading voice model: {} ...", voice_def.name);
    let client = Client::builder()
        .user_agent("Sherpa-ONNX-Rust-Downloader")
        .build()?;

    let mut response = client.get(voice_def.url).send().context("HTTP download failed")?;
    if !response.status().is_success() {
        return Err(anyhow!("HTTP error status: {}", response.status()));
    }

    let mut downloaded_bytes: Vec<u8> = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        downloaded_bytes.extend_from_slice(&buffer[..bytes_read]);
    }

    let bz = BzDecoder::new(&downloaded_bytes[..]);
    let mut archive = Archive::new(bz);

    fs::create_dir_all(model_dir)?;
    archive.unpack(model_dir).context("Unpacking archive failed")?;
    Ok(())
}

fn find_supertonic_files(base_dir: &Path) -> Result<OfflineTtsSupertonicModelConfig> {
    let mut duration_predictor = None;
    let mut text_encoder = None;
    let mut vector_estimator = None;
    let mut vocoder = None;
    let mut tts_json = None;
    let mut unicode_indexer = None;
    let mut voice_style = None;

    let mut files = Vec::new();
    search_dir_recursive(base_dir, &mut files, &mut Vec::new())?;

    for path in files {
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        match file_name {
            name if name.contains("duration_predictor") => duration_predictor = Some(path.to_string_lossy().to_string()),
            name if name.contains("text_encoder") => text_encoder = Some(path.to_string_lossy().to_string()),
            name if name.contains("vector_estimator") => vector_estimator = Some(path.to_string_lossy().to_string()),
            name if name.contains("vocoder") => vocoder = Some(path.to_string_lossy().to_string()),
            "tts.json" => tts_json = Some(path.to_string_lossy().to_string()),
            name if name.contains("unicode_indexer") => unicode_indexer = Some(path.to_string_lossy().to_string()),
            "voice.bin" => voice_style = Some(path.to_string_lossy().to_string()),
            _ => {}
        }
    }

    Ok(OfflineTtsSupertonicModelConfig {
        duration_predictor,
        text_encoder,
        vector_estimator,
        vocoder,
        tts_json,
        unicode_indexer,
        voice_style,
    })
}

fn find_vits_piper_files(base_dir: &Path) -> Result<OfflineTtsVitsModelConfig> {
    let mut model_path = None;
    let mut tokens_path = None;
    let mut data_dir_path = None;

    let mut files = Vec::new();
    let mut dirs = Vec::new();
    search_dir_recursive(base_dir, &mut files, &mut dirs)?;

    for path in files {
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.ends_with(".onnx") && !file_name.contains("json") {
            model_path = Some(path.to_string_lossy().to_string());
        } else if file_name == "tokens.txt" {
            tokens_path = Some(path.to_string_lossy().to_string());
        }
    }

    for dir in dirs {
        let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if dir_name == "espeak-ng-data" {
            data_dir_path = Some(dir.to_string_lossy().to_string());
        }
    }

    Ok(OfflineTtsVitsModelConfig {
        model: model_path,
        tokens: tokens_path,
        data_dir: data_dir_path,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        length_scale: 1.0,
        dict_dir: None,
        lexicon: None,
    })
}

fn search_dir_recursive(dir: &Path, files: &mut Vec<PathBuf>, dirs: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path.clone());
                search_dir_recursive(&path, files, dirs)?;
            } else {
                files.push(path);
            }
        }
    }
    Ok(())
}
