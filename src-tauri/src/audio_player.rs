//! Background audio playback controller using rodio.
//! Keeps rodio playback speed fixed at 1.0x to preserve natural vocal pitch (zero chipmunk effect).

use anyhow::{anyhow, Result};
use rodio::cpal::traits::{DeviceTrait, HostTrait};
use rodio::{buffer::SamplesBuffer, cpal, OutputStream, OutputStreamHandle, Sink};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread;

pub struct AudioItem {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub text_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

pub enum PlayerCommand {
    Enqueue(AudioItem),
    SetVolume(f32),
    SetPaused(bool),
    SetDevice(Option<String>),
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

pub fn get_output_devices() -> Vec<AudioDeviceInfo> {
    let host = cpal::default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());

    let mut list = Vec::new();
    if let Ok(devices) = host.output_devices() {
        for dev in devices {
            if let Ok(name) = dev.name() {
                let is_default = default_name.as_ref().map(|d| d == &name).unwrap_or(false);
                list.push(AudioDeviceInfo { name, is_default });
            }
        }
    }
    list.sort_by(|a, b| a.name.cmp(&b.name));
    list.dedup_by(|a, b| a.name == b.name);
    list
}

fn open_stream_for_device(name: Option<&str>) -> Option<(OutputStream, OutputStreamHandle)> {
    if let Some(target) = name {
        if !target.is_empty() && target != "default" {
            let host = cpal::default_host();
            if let Ok(mut devs) = host.output_devices() {
                if let Some(dev) = devs.find(|d| d.name().map(|n| n == target).unwrap_or(false)) {
                    if let Ok(stream) = OutputStream::try_from_device(&dev) {
                        return Some(stream);
                    }
                }
            }
        }
    }
    OutputStream::try_default().ok()
}

pub struct AudioPlayer {
    sender: Sender<PlayerCommand>,
    volume: Arc<parking_lot::Mutex<f32>>,
    is_paused: Arc<AtomicBool>,
    queue_len: Arc<parking_lot::Mutex<usize>>,
    queue_text_len: Arc<parking_lot::Mutex<usize>>,
    #[allow(dead_code)]
    current_device: Arc<parking_lot::Mutex<Option<String>>>,
}

impl AudioPlayer {
    pub fn new() -> Result<Self> {
        let (tx, rx) = channel::<PlayerCommand>();
        let volume = Arc::new(parking_lot::Mutex::new(1.0f32));
        let is_paused = Arc::new(AtomicBool::new(true));
        let queue_len = Arc::new(parking_lot::Mutex::new(0usize));
        let queue_text_len = Arc::new(parking_lot::Mutex::new(0usize));
        let current_device = Arc::new(parking_lot::Mutex::new(None::<String>));

        let vol_clone = Arc::clone(&volume);
        let paused_clone = Arc::clone(&is_paused);
        let qlen_clone = Arc::clone(&queue_len);
        let qtext_clone = Arc::clone(&queue_text_len);
        let dev_clone = Arc::clone(&current_device);

        thread::spawn(move || {
            let mut stream_pair = open_stream_for_device(None);
            let mut sink = stream_pair.as_ref().and_then(|(_, h)| Sink::try_new(h).ok());
            let mut pending_queue: Vec<AudioItem> = Vec::new();
            let mut active_text_len: usize = 0;

            let recreate_sink = |stream_pair: &Option<(OutputStream, OutputStreamHandle)>, vol: f32, paused: bool| -> Option<Sink> {
                if let Some((_, handle)) = stream_pair {
                    if let Ok(new_sink) = Sink::try_new(handle) {
                        new_sink.set_volume(vol);
                        if paused {
                            new_sink.pause();
                        }
                        return Some(new_sink);
                    }
                }
                None
            };

            let handle_command = |cmd: PlayerCommand,
                                  pending_queue: &mut Vec<AudioItem>,
                                  stream_pair: &mut Option<(OutputStream, OutputStreamHandle)>,
                                  sink: &mut Option<Sink>,
                                  active_text_len: &mut usize| {
                match cmd {
                    PlayerCommand::Enqueue(item) => pending_queue.push(item),
                    PlayerCommand::SetVolume(v) => {
                        *vol_clone.lock() = v;
                        if let Some(ref s) = sink {
                            s.set_volume(v);
                        }
                    }
                    PlayerCommand::SetPaused(p) => {
                        paused_clone.store(p, Ordering::Relaxed);
                        if let Some(ref s) = sink {
                            if p {
                                s.pause();
                            } else {
                                s.play();
                            }
                        }
                    }
                    PlayerCommand::SetDevice(new_dev) => {
                        *dev_clone.lock() = new_dev.clone();
                        if let Some(ref s) = sink {
                            s.stop();
                        }
                        *sink = None;
                        *stream_pair = None;

                        *stream_pair = open_stream_for_device(new_dev.as_deref());
                        let vol = *vol_clone.lock();
                        let paused = paused_clone.load(Ordering::Relaxed);
                        *sink = recreate_sink(stream_pair, vol, paused);
                        *active_text_len = 0;
                    }
                    PlayerCommand::Skip => {
                        pending_queue.clear();
                        if let Some(ref s) = sink {
                            s.stop();
                        }
                        let vol = *vol_clone.lock();
                        let paused = paused_clone.load(Ordering::Relaxed);
                        *sink = recreate_sink(stream_pair, vol, paused);
                        *active_text_len = 0;
                    }
                }
            };

            loop {
                // Block thread efficiently when queue & sink are empty, waking up on new command
                let is_sink_empty = sink.as_ref().map(|s| s.empty()).unwrap_or(true);
                if pending_queue.is_empty() && is_sink_empty {
                    if let Ok(cmd) = rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        handle_command(cmd, &mut pending_queue, &mut stream_pair, &mut sink, &mut active_text_len);
                    }
                }

                // Receive any remaining queued commands from channel
                while let Ok(cmd) = rx.try_recv() {
                    handle_command(cmd, &mut pending_queue, &mut stream_pair, &mut sink, &mut active_text_len);
                }

                // Cap pending audio queue to max 10 items to prevent PCM buffer RAM accumulation
                if pending_queue.len() > 10 {
                    let to_remove = pending_queue.len() - 10;
                    pending_queue.drain(..to_remove);
                }

                if sink.as_ref().map(|s| s.empty()).unwrap_or(true) {
                    active_text_len = 0;
                }

                // Calculate queue metrics
                let sink_len = sink.as_ref().map(|s| s.len()).unwrap_or(0);
                let total_pending_chars: usize = pending_queue.iter().map(|i| i.text_len).sum::<usize>() + active_text_len;
                let current_qlen = pending_queue.len() + sink_len;

                *qlen_clone.lock() = current_qlen;
                *qtext_clone.lock() = total_pending_chars;

                if let Some(ref s) = sink {
                    // If paused, pause sink without destroying it
                    if paused_clone.load(Ordering::Relaxed) {
                        if !s.is_paused() {
                            s.pause();
                        }
                        thread::sleep(std::time::Duration::from_millis(50));
                        continue;
                    } else {
                        if s.is_paused() {
                            s.play();
                        }
                    }

                    // Apply current volume
                    let vol = *vol_clone.lock();
                    s.set_volume(vol);

                    // Dynamically accelerate audio player (rodio sink) speed linearly when queue builds up
                    let player_speed = calculate_player_speed(current_qlen, total_pending_chars);
                    s.set_speed(player_speed);

                    // Play next item if sink has space
                    if s.len() < 2 && !pending_queue.is_empty() {
                        let next_item = pending_queue.remove(0);
                        active_text_len = next_item.text_len;
                        let source = SamplesBuffer::new(1, next_item.sample_rate, next_item.samples);
                        s.append(source);
                    }
                } else if stream_pair.is_none() {
                    let cur_dev = dev_clone.lock().clone();
                    stream_pair = open_stream_for_device(cur_dev.as_deref());
                    let vol = *vol_clone.lock();
                    let paused = paused_clone.load(Ordering::Relaxed);
                    sink = recreate_sink(&stream_pair, vol, paused);
                }

                thread::sleep(std::time::Duration::from_millis(15));
            }
        });

        Ok(Self {
            sender: tx,
            volume,
            is_paused,
            queue_len,
            queue_text_len,
            current_device,
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

    pub fn set_device(&self, device_name: Option<String>) -> Result<()> {
        self.sender
            .send(PlayerCommand::SetDevice(device_name))
            .map_err(|e| anyhow!("Failed to set audio device: {:?}", e))
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

    #[allow(dead_code)]
    pub fn get_current_device(&self) -> Option<String> {
        self.current_device.lock().clone()
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

    #[test]
    fn test_set_device_command() {
        let player = AudioPlayer::new().unwrap();
        assert!(player.set_device(None).is_ok());
    }
}

