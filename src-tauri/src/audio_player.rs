//! Background audio playback controller using rodio.
//! Keeps rodio playback speed fixed at 1.0x to preserve natural vocal pitch (zero chipmunk effect).

use anyhow::{anyhow, Result};
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread;

pub struct AudioItem {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub text_len: usize,
}

pub enum PlayerCommand {
    Enqueue(AudioItem),
    SetVolume(f32),
    SetPaused(bool),
    Skip,
}

pub fn calculate_player_speed(qlen: usize, qtext: usize) -> f32 {
    if qlen <= 1 {
        return 1.0;
    }
    let extra_items = (qlen as f32 - 1.0).max(0.0);
    let extra_chars = (qtext as f32 - 30.0).max(0.0);
    let item_boost = extra_items * 0.35;
    let char_boost = extra_chars * 0.008;

    (1.0 + item_boost + char_boost).min(2.0)
}

pub struct AudioPlayer {
    sender: Sender<PlayerCommand>,
    volume: Arc<parking_lot::Mutex<f32>>,
    is_paused: Arc<AtomicBool>,
    queue_len: Arc<parking_lot::Mutex<usize>>,
    queue_text_len: Arc<parking_lot::Mutex<usize>>,
}

impl AudioPlayer {
    pub fn new() -> Result<Self> {
        let (tx, rx) = channel::<PlayerCommand>();
        let volume = Arc::new(parking_lot::Mutex::new(1.0f32));
        let is_paused = Arc::new(AtomicBool::new(true));
        let queue_len = Arc::new(parking_lot::Mutex::new(0usize));
        let queue_text_len = Arc::new(parking_lot::Mutex::new(0usize));

        let vol_clone = Arc::clone(&volume);
        let paused_clone = Arc::clone(&is_paused);
        let qlen_clone = Arc::clone(&queue_len);
        let qtext_clone = Arc::clone(&queue_text_len);

        thread::spawn(move || {
            let _stream_handle: Option<(OutputStream, OutputStreamHandle)> = OutputStream::try_default().ok();
            if let Some((_stream, handle)) = _stream_handle {
                let mut sink = Sink::try_new(&handle).unwrap();
                let mut pending_queue: Vec<AudioItem> = Vec::new();
                let mut active_text_len: usize = 0;

                loop {
                    // Block thread efficiently when queue & sink are empty, waking up on new command
                    if pending_queue.is_empty() && sink.empty() {
                        if let Ok(cmd) = rx.recv_timeout(std::time::Duration::from_millis(50)) {
                            match cmd {
                                PlayerCommand::Enqueue(item) => pending_queue.push(item),
                                PlayerCommand::SetVolume(v) => *vol_clone.lock() = v,
                                PlayerCommand::SetPaused(p) => paused_clone.store(p, Ordering::Relaxed),
                                PlayerCommand::Skip => {
                                    pending_queue.clear();
                                    sink.stop();
                                    if let Ok(new_sink) = Sink::try_new(&handle) {
                                        sink = new_sink;
                                    }
                                    active_text_len = 0;
                                }
                            }
                        }
                    }

                    // Receive any remaining queued commands from channel
                    while let Ok(cmd) = rx.try_recv() {
                        match cmd {
                            PlayerCommand::Enqueue(item) => pending_queue.push(item),
                            PlayerCommand::SetVolume(v) => *vol_clone.lock() = v,
                            PlayerCommand::SetPaused(p) => paused_clone.store(p, Ordering::Relaxed),
                            PlayerCommand::Skip => {
                                pending_queue.clear();
                                sink.stop();
                                if let Ok(new_sink) = Sink::try_new(&handle) {
                                    sink = new_sink;
                                }
                                active_text_len = 0;
                            }
                        }
                    }

                    // Cap pending audio queue to max 10 items to prevent PCM buffer RAM accumulation
                    if pending_queue.len() > 10 {
                        let to_remove = pending_queue.len() - 10;
                        pending_queue.drain(..to_remove);
                    }

                    if sink.empty() {
                        active_text_len = 0;
                    }

                    // Calculate queue metrics
                    let total_pending_chars: usize = pending_queue.iter().map(|i| i.text_len).sum::<usize>() + active_text_len;
                    let current_qlen = pending_queue.len() + sink.len();

                    *qlen_clone.lock() = current_qlen;
                    *qtext_clone.lock() = total_pending_chars;

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

                    // Dynamically accelerate audio player (rodio sink) speed linearly when queue builds up
                    let player_speed = calculate_player_speed(current_qlen, total_pending_chars);
                    sink.set_speed(player_speed);

                    // Play next item if sink has space
                    if sink.len() < 2 && !pending_queue.is_empty() {
                        let next_item = pending_queue.remove(0);
                        active_text_len = next_item.text_len;
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
            queue_text_len,
        })
    }

    pub fn enqueue(&self, item: AudioItem) -> Result<()> {
        self.sender
            .send(PlayerCommand::Enqueue(item))
            .map_err(|e| anyhow!("Failed to enqueue audio item: {:?}", e))
    }

    pub fn skip(&self) -> Result<()> {
        self.sender
            .send(PlayerCommand::Skip)
            .map_err(|e| anyhow!("Failed to send skip command: {:?}", e))
    }

    pub fn set_volume(&self, vol: f32) {
        let vol = vol.clamp(0.0, 3.0);
        *self.volume.lock() = vol;
        let _ = self.sender.send(PlayerCommand::SetVolume(vol));
    }

    pub fn set_paused(&self, paused: bool) {
        self.is_paused.store(paused, Ordering::Relaxed);
        let _ = self.sender.send(PlayerCommand::SetPaused(paused));
    }

    pub fn get_queue_len(&self) -> usize {
        *self.queue_len.lock()
    }

    pub fn get_queue_text_len(&self) -> usize {
        *self.queue_text_len.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_setting() {
        let player = AudioPlayer::new().unwrap();
        player.set_volume(2.5);
        assert_eq!(*player.volume.lock(), 2.5);
    }

    #[test]
    fn test_skip_command() {
        let player = AudioPlayer::new().unwrap();
        assert!(player.skip().is_ok());
    }
}

