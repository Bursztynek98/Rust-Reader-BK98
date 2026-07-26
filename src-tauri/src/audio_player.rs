//! Background audio playback controller using rodio.
//! Keeps rodio playback speed fixed at 1.0x to preserve natural vocal pitch (zero chipmunk effect).

use anyhow::{anyhow, Result};
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread;

pub struct AudioItem {
    #[allow(dead_code)]
    pub text: String,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct AudioPlayer {
    sender: Sender<AudioItem>,
    volume: Arc<parking_lot::Mutex<f32>>,
    is_paused: Arc<AtomicBool>,
    queue_len: Arc<parking_lot::Mutex<usize>>,
}

impl AudioPlayer {
    pub fn new() -> Result<Self> {
        let (tx, rx) = channel::<AudioItem>();
        let volume = Arc::new(parking_lot::Mutex::new(1.0f32));
        let is_paused = Arc::new(AtomicBool::new(true));
        let queue_len = Arc::new(parking_lot::Mutex::new(0usize));

        let vol_clone = Arc::clone(&volume);
        let paused_clone = Arc::clone(&is_paused);
        let qlen_clone = Arc::clone(&queue_len);

        thread::spawn(move || {
            let _stream_handle: Option<(OutputStream, OutputStreamHandle)> = OutputStream::try_default().ok();
            if let Some((_stream, handle)) = _stream_handle {
                let sink = Sink::try_new(&handle).unwrap();
                let mut pending_queue: Vec<AudioItem> = Vec::new();

                loop {
                    // Efficiently block thread when queue & sink are empty, waking up immediately on new audio
                    if pending_queue.is_empty() && sink.empty() {
                        if let Ok(item) = rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            pending_queue.push(item);
                        }
                    }

                    // Receive any remaining queued items from channel
                    while let Ok(item) = rx.try_recv() {
                        pending_queue.push(item);
                    }

                    // Cap pending audio queue to max 10 items to prevent PCM buffer RAM accumulation
                    if pending_queue.len() > 10 {
                        let to_remove = pending_queue.len() - 10;
                        pending_queue.drain(..to_remove);
                    }

                    // Update queue length metric
                    {
                        let mut q = qlen_clone.lock();
                        *q = pending_queue.len() + sink.len();
                    }

                    // If paused, pause sink without destroying it
                    if paused_clone.load(Ordering::Relaxed) {
                        if !sink.is_paused() {
                            sink.pause();
                        }
                        thread::sleep(std::time::Duration::from_millis(50));
                        continue;
                    } else {
                        if sink.is_paused() {
                            sink.play();
                        }
                    }

                    // Apply current volume
                    let vol = *vol_clone.lock();
                    sink.set_volume(vol);

                    // Fixed 1.0x playback speed in rodio to ensure pitch is never distorted
                    sink.set_speed(1.0);

                    // Play next item if sink has space
                    if sink.len() < 2 && !pending_queue.is_empty() {
                        let next_item = pending_queue.remove(0);
                        let source = SamplesBuffer::new(1, next_item.sample_rate, next_item.samples);
                        sink.append(source);
                    }

                    thread::sleep(std::time::Duration::from_millis(15));
                }
            }
        });

        Ok(Self {
            sender: tx,
            volume,
            is_paused,
            queue_len,
        })
    }

    pub fn enqueue(&self, item: AudioItem) -> Result<()> {
        self.sender
            .send(item)
            .map_err(|e| anyhow!("Failed to enqueue audio item: {:?}", e))
    }

    pub fn set_volume(&self, vol: f32) {
        *self.volume.lock() = vol.clamp(0.0, 1.0);
    }

    pub fn set_paused(&self, paused: bool) {
        self.is_paused.store(paused, Ordering::Relaxed);
    }

    pub fn get_queue_len(&self) -> usize {
        *self.queue_len.lock()
    }
}
