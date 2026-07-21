"""
TTS Worker – OmniVoice GPU pipeline.
Model weights are loaded ONCE at startup. Only the reference audio path
changes when the user uploads a new voice file, enabling fast voice swapping.
"""
import asyncio
from pathlib import Path


class TTSWorker:
    """OmniVoice singleton – GPU inference with persistent model weights."""

    def __init__(self):
        self.model = None
        self.ref_audio_path: str | None = None
        self._voice_lock = asyncio.Lock()

    # ──────────────────────────────────────────────────────────────────────
    async def load_model(self):
        """Load OmniVoice model asynchronously (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self):
        import traceback
        # Verify torch GPU is available before trying to load
        try:
            import torch
            print(f"[TTS] torch version: {torch.__version__}")
            print(f"[TTS] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[TTS] GPU: {torch.cuda.get_device_name(0)}")
                print(f"[TTS] CUDA version (runtime): {torch.version.cuda}")
        except Exception as te:
            print(f"[TTS] torch check failed: {te}")

        try:
            from omnivoice import OmniVoice
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cuda:0",
            )
            print("[TTS] OmniVoice model loaded on GPU (CUDA:0).")
        except Exception as e:
            print(f"[TTS] WARNING: Failed to load OmniVoice: {e}")
            print("[TTS] Full traceback:")
            traceback.print_exc()
            print("[TTS] TTS will be disabled until the model is available.")
            self.model = None

    # ──────────────────────────────────────────────────────────────────────
    async def set_voice(self, path: str):
        """
        Set a new reference voice audio file.
        Converts MP3/OGG/FLAC → WAV automatically.
        Does NOT reload model weights.
        """
        async with self._voice_lock:
            loop = asyncio.get_event_loop()
            wav_path = await loop.run_in_executor(None, self._to_wav, path)
            self.ref_audio_path = wav_path
            print(f"[TTS] Voice reference updated → {wav_path}")

    def _to_wav(self, path: str) -> str:
        """Convert audio file to 22050 Hz mono WAV for OmniVoice."""
        p = Path(path)
        if p.suffix.lower() == ".wav":
            return path

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(p))
            audio = audio.set_frame_rate(22050).set_channels(1)
            wav_path = str(p.with_suffix(".wav"))
            audio.export(wav_path, format="wav")
            print(f"[TTS] Converted {p.name} → {Path(wav_path).name}")
            return wav_path
        except Exception as e:
            print(f"[TTS] Conversion error: {e}, using original path")
            return path

    # ──────────────────────────────────────────────────────────────────────
    def generate_sync(self, text: str, output_path: str) -> bool:
        """
        Generate audio for `text` using the loaded model and reference voice.
        Saves result as WAV to `output_path`.
        Blocking – intended to be called via loop.run_in_executor().
        """
        if self.model is None:
            print("[TTS] Model not loaded – skipping generation.")
            return False

        if self.ref_audio_path is None:
            print("[TTS] No reference voice – upload a voice file first.")
            return False

        try:
            result = self.model.generate(
                text=text,
                ref_audio=self.ref_audio_path,
            )

            # OmniVoice returns (audio_array, sample_rate) or just array
            if isinstance(result, tuple):
                audio_data, sr = result
            else:
                audio_data = result
                sr = 22050

            import soundfile as sf
            import numpy as np
            if hasattr(audio_data, "cpu"):
                audio_data = audio_data.cpu().numpy()
            audio_data = np.asarray(audio_data, dtype=np.float32)
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()

            sf.write(output_path, audio_data, sr)
            print(f"[TTS] Generated: {Path(output_path).name} ({len(audio_data)/sr:.1f}s)")
            return True

        except Exception as e:
            print(f"[TTS] Generation error: {e}")
            return False
