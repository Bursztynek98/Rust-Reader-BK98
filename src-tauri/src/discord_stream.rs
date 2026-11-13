//! Local companion HTTP & WebSocket server for streaming TTS audio to a browser tab
//! for Discord window screen-share capture.

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use hound::{WavSpec, WavWriter};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

#[derive(Clone)]
pub enum StreamEvent {
    Audio(Arc<Vec<u8>>),
    Volume(f32),
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordStreamStatus {
    pub is_enabled: bool,
    pub mute_local: bool,
    pub connected_clients: usize,
    pub port: u16,
    pub url: String,
}

pub struct DiscordStreamManager {
    is_enabled: Arc<AtomicBool>,
    mute_local: Arc<AtomicBool>,
    connected_clients: Arc<AtomicUsize>,
    port: Arc<AtomicU16>,
    volume: Arc<parking_lot::Mutex<f32>>,
    event_tx: broadcast::Sender<StreamEvent>,
}

impl DiscordStreamManager {
    pub fn new(default_port: u16) -> Self {
        let (event_tx, _) = broadcast::channel::<StreamEvent>(64);
        let is_enabled = Arc::new(AtomicBool::new(true));
        let mute_local = Arc::new(AtomicBool::new(true));
        let connected_clients = Arc::new(AtomicUsize::new(0));
        let port = Arc::new(AtomicU16::new(default_port));
        let volume = Arc::new(parking_lot::Mutex::new(1.0f32));

        let mgr = Self {
            is_enabled: Arc::clone(&is_enabled),
            mute_local: Arc::clone(&mute_local),
            connected_clients: Arc::clone(&connected_clients),
            port: Arc::clone(&port),
            volume: Arc::clone(&volume),
            event_tx: event_tx.clone(),
        };

        // Start server in background thread with Tokio runtime
        let is_enabled_cl = Arc::clone(&is_enabled);
        let conn_clients_cl = Arc::clone(&connected_clients);
        let port_cl = Arc::clone(&port);
        let volume_cl = Arc::clone(&volume);
        let event_tx_cl = event_tx;

        std::thread::Builder::new()
            .name("discord-stream-server".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[DiscordStream] Failed to build Tokio runtime: {:?}", e);
                        return;
                    }
                };

                rt.block_on(async move {
                    if let Err(e) = run_server(port_cl, conn_clients_cl, is_enabled_cl, volume_cl, event_tx_cl).await {
                        eprintln!("[DiscordStream] Server error: {:?}", e);
                    }
                });
            })
            .expect("Failed to spawn discord-stream-server thread");

        mgr
    }

    pub fn is_enabled(&self) -> bool {
        self.is_enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, val: bool) {
        self.is_enabled.store(val, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub fn get_volume(&self) -> f32 {
        *self.volume.lock()
    }

    pub fn set_volume(&self, val: f32) {
        let val = val.clamp(0.0, 3.0);
        *self.volume.lock() = val;
        let _ = self.event_tx.send(StreamEvent::Volume(val));
    }

    pub fn should_mute_local(&self) -> bool {
        self.is_enabled()
            && self.mute_local.load(Ordering::Relaxed)
            && self.connected_clients.load(Ordering::Relaxed) > 0
    }

    pub fn set_mute_local(&self, val: bool) {
        self.mute_local.store(val, Ordering::Relaxed);
    }

    pub fn get_connected_clients(&self) -> usize {
        self.connected_clients.load(Ordering::Relaxed)
    }

    pub fn get_port(&self) -> u16 {
        self.port.load(Ordering::Relaxed)
    }

    pub fn get_status(&self) -> DiscordStreamStatus {
        let port = self.get_port();
        DiscordStreamStatus {
            is_enabled: self.is_enabled(),
            mute_local: self.mute_local.load(Ordering::Relaxed),
            connected_clients: self.get_connected_clients(),
            port,
            url: format!("http://localhost:{}", port),
        }
    }

    pub fn broadcast_audio(&self, samples: &[f32], sample_rate: u32) -> bool {
        if !self.is_enabled() || self.get_connected_clients() == 0 {
            return false;
        }

        match samples_to_wav(samples, sample_rate) {
            Ok(wav_bytes) => {
                let arc_data = Arc::new(wav_bytes);
                let _ = self.event_tx.send(StreamEvent::Audio(arc_data));
                true
            }
            Err(e) => {
                eprintln!("[DiscordStream] Failed to encode WAV: {:?}", e);
                false
            }
        }
    }

    pub fn skip(&self) {
        let _ = self.event_tx.send(StreamEvent::Skip);
    }

    pub fn open_browser(&self) {
        let url = format!("http://localhost:{}", self.get_port());
        #[cfg(target_os = "windows")]
        {
            use std::ffi::OsStr;
            use std::os::windows::ffi::OsStrExt;

            let wide_url: Vec<u16> = OsStr::new(&url).encode_wide().chain(std::iter::once(0)).collect();
            let wide_op: Vec<u16> = OsStr::new("open").encode_wide().chain(std::iter::once(0)).collect();

            #[link(name = "shell32")]
            extern "system" {
                fn ShellExecuteW(
                    hwnd: *mut std::ffi::c_void,
                    operation: *const u16,
                    file: *const u16,
                    parameters: *const u16,
                    directory: *const u16,
                    show_cmd: i32,
                ) -> isize;
            }

            unsafe {
                ShellExecuteW(
                    std::ptr::null_mut(),
                    wide_op.as_ptr(),
                    wide_url.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1, // SW_SHOWNORMAL
                );
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = std::process::Command::new("xdg-open")
                .arg(&url)
                .spawn();
        }
    }
}

pub fn samples_to_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    {
        let mut writer = WavWriter::new(&mut cursor, spec)?;
        for &sample in samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(sample_i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

#[derive(Clone)]
struct AppStateWs {
    connected_clients: Arc<AtomicUsize>,
    is_enabled: Arc<AtomicBool>,
    volume: Arc<parking_lot::Mutex<f32>>,
    event_tx: broadcast::Sender<StreamEvent>,
}

async fn run_server(
    port_holder: Arc<AtomicU16>,
    connected_clients: Arc<AtomicUsize>,
    is_enabled: Arc<AtomicBool>,
    volume: Arc<parking_lot::Mutex<f32>>,
    event_tx: broadcast::Sender<StreamEvent>,
) -> Result<()> {
    let base_port = port_holder.load(Ordering::Relaxed);
    let mut listener: Option<TcpListener> = None;

    for offset in 0..10 {
        let try_port = base_port + offset;
        let addr = SocketAddr::from(([127, 0, 0, 1], try_port));
        if let Ok(l) = TcpListener::bind(addr).await {
            println!("[DiscordStream] ✓ Listening on http://{}", addr);
            listener = Some(l);
            port_holder.store(try_port, Ordering::Relaxed);
            break;
        }
    }

    let listener = match listener {
        Some(l) => l,
        None => {
            return Err(anyhow::anyhow!(
                "Failed to bind companion web server to any port in range {}-{}",
                base_port,
                base_port + 10
            ));
        }
    };

    let state = AppStateWs {
        connected_clients,
        is_enabled,
        volume,
        event_tx,
    };

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/ws", get(ws_handler))
        .with_state(state);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn index_handler() -> impl IntoResponse {
    Html(COMPANION_HTML)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppStateWs>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppStateWs) {
    state.connected_clients.fetch_add(1, Ordering::Relaxed);
    println!(
        "[DiscordStream] Client connected. Total active tabs: {}",
        state.connected_clients.load(Ordering::Relaxed)
    );

    // Send initial volume immediately to the newly connected browser tab
    let initial_vol = *state.volume.lock();
    let init_json = serde_json::json!({
        "type": "volume",
        "volume": initial_vol
    });
    let _ = socket.send(Message::Text(init_json.to_string())).await;

    let mut rx = state.event_tx.subscribe();

    loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Ok(StreamEvent::Audio(wav_data)) => {
                        if state.is_enabled.load(Ordering::Relaxed) {
                            if socket.send(Message::Binary((*wav_data).clone())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Ok(StreamEvent::Volume(vol)) => {
                        let json = serde_json::json!({ "type": "volume", "volume": vol });
                        if socket.send(Message::Text(json.to_string())).await.is_err() {
                            break;
                        }
                    }
                    Ok(StreamEvent::Skip) => {
                        let json = serde_json::json!({ "type": "skip" });
                        if socket.send(Message::Text(json.to_string())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(missed)) => {
                        eprintln!("[DiscordStream] Client lagged, missed {} audio frames", missed);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Ping(_))) => {
                        if socket.send(Message::Pong(vec![])).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        break;
                    }
                    Some(Err(_)) => {
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    state.connected_clients.fetch_sub(1, Ordering::Relaxed);
    println!(
        "[DiscordStream] Client disconnected. Active tabs: {}",
        state.connected_clients.load(Ordering::Relaxed)
    );
}

const COMPANION_HTML: &str = r#"<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Rust Reader — Audio</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0b0d14;
      color: #f3f4f6;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      user-select: none;
      padding: 20px;
    }
    .card {
      background: rgba(18, 22, 34, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 16px;
      padding: 28px 36px;
      text-align: center;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      max-width: 420px;
      transition: border-color 0.2s;
    }
    .status-dot {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #f59e0b;
      display: inline-block;
      margin-right: 8px;
      vertical-align: middle;
      box-shadow: 0 0 10px #f59e0b;
    }
    .status-dot.active {
      background: #10b981;
      box-shadow: 0 0 12px #10b981;
    }
    .title {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .subtitle {
      font-size: 13px;
      color: #9ca3af;
      line-height: 1.4;
    }
  </style>
</head>
<body>
  <div class="card" id="card">
    <div class="title">
      <span class="status-dot active" id="dot"></span>
      <span id="title">Audio aktywne dla Discorda</span>
    </div>
    <div class="subtitle" id="subtitle">
      Karta odtwarza głos lektora w tle dla streamu na Discordzie.
    </div>
    <div id="vol-info" style="margin-top: 12px; font-size: 13px; font-weight: 500; color: #9ca3af;">
      Głośność: 100%
    </div>
  </div>

  <script>
    let audioCtx = null;
    let gainNode = null;
    let compressor = null;
    let currentVolume = 1.0;
    let audioQueue = [];
    let isPlaying = false;
    let currentSource = null;
    let ws = null;
    let keepAliveOsc = null;

    const dot = document.getElementById('dot');
    const title = document.getElementById('title');
    const subtitle = document.getElementById('subtitle');
    const volInfo = document.getElementById('vol-info');

    function updateVolumeDisplay(vol) {
      if (!volInfo) return;
      const pct = Math.round(vol * 100);
      if (pct > 100) {
        volInfo.textContent = `Głośność: ${pct}% (🔊 Wzmocnienie ${(vol).toFixed(1)}x)`;
        volInfo.style.color = '#38bdf8';
      } else {
        volInfo.textContent = `Głośność: ${pct}%`;
        volInfo.style.color = '#9ca3af';
      }
    }

    function initAudio() {
      if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContextClass();
        try {
          gainNode = audioCtx.createGain();
          gainNode.gain.setValueAtTime(currentVolume, audioCtx.currentTime);

          compressor = audioCtx.createDynamicsCompressor();
          compressor.threshold.setValueAtTime(-3, audioCtx.currentTime);
          compressor.knee.setValueAtTime(6, audioCtx.currentTime);
          compressor.ratio.setValueAtTime(12, audioCtx.currentTime);
          compressor.attack.setValueAtTime(0.003, audioCtx.currentTime);
          compressor.release.setValueAtTime(0.15, audioCtx.currentTime);

          gainNode.connect(compressor);
          compressor.connect(audioCtx.destination);

          keepAliveOsc = audioCtx.createOscillator();
          const silentGain = audioCtx.createGain();
          silentGain.gain.value = 0.00001;
          keepAliveOsc.connect(silentGain);
          silentGain.connect(audioCtx.destination);
          keepAliveOsc.start();
        } catch (e) {}
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume().catch(() => {});
      }
    }

    // Initialize immediately on page load
    initAudio();

    // Auto-resume on any window interaction (focus, click, keydown)
    window.addEventListener('focus', initAudio);
    document.addEventListener('click', initAudio);
    document.addEventListener('keydown', initAudio);

    function connectWs() {
      const loc = window.location;
      const wsUrl = (loc.protocol === 'https:' ? 'wss://' : 'ws://') + loc.host + '/ws';
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        dot.className = 'status-dot active';
        title.innerText = 'Audio aktywne dla Discorda';
      };

      ws.onclose = () => {
        dot.className = 'status-dot';
        title.innerText = 'Rozłączono (ponawianie...)';
        setTimeout(connectWs, 2000);
      };

      ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'skip') {
              audioQueue = [];
              if (currentSource) {
                try { currentSource.stop(); } catch(e){}
                currentSource = null;
              }
              isPlaying = false;
            } else if (data.type === 'volume' && typeof data.volume === 'number') {
              currentVolume = data.volume;
              if (gainNode && audioCtx) {
                gainNode.gain.setValueAtTime(currentVolume, audioCtx.currentTime);
              }
              updateVolumeDisplay(currentVolume);
            }
          } catch (e) {}
        } else if (event.data instanceof ArrayBuffer) {
          initAudio();
          if (!audioCtx) return;
          try {
            const buf = await audioCtx.decodeAudioData(event.data);
            audioQueue.push(buf);
            playNext();
          } catch (e) {}
        }
      };
    }

    function playNext() {
      if (isPlaying || audioQueue.length === 0 || !audioCtx) return;
      isPlaying = true;
      const buf = audioQueue.shift();
      currentSource = audioCtx.createBufferSource();
      currentSource.buffer = buf;
      if (gainNode) {
        currentSource.connect(gainNode);
      } else {
        currentSource.connect(audioCtx.destination);
      }
      currentSource.onended = () => {
        isPlaying = false;
        currentSource = null;
        playNext();
      };
      currentSource.start(0);
    }

    connectWs();
  </script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_to_wav_encoding() {
        let sample_rate = 22050;
        let samples: Vec<f32> = (0..2205)
            .map(|i| (i as f32 * 0.05).sin() * 0.5)
            .collect();

        let wav_bytes = samples_to_wav(&samples, sample_rate).expect("Encoding should succeed");
        assert!(!wav_bytes.is_empty());
        assert_eq!(&wav_bytes[0..4], b"RIFF");
        assert_eq!(&wav_bytes[8..12], b"WAVE");

        // Verify with hound reader
        let cursor = Cursor::new(wav_bytes);
        let reader = hound::WavReader::new(cursor).expect("Hound should read the generated WAV");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, sample_rate);
        assert_eq!(reader.spec().bits_per_sample, 16);
    }

    #[test]
    fn test_stream_manager_initial_state() {
        let mgr = DiscordStreamManager::new(4848);
        assert!(mgr.is_enabled());
        assert_eq!(mgr.get_connected_clients(), 0);
        // With 0 clients, should_mute_local must be false
        assert!(!mgr.should_mute_local());
    }
}

