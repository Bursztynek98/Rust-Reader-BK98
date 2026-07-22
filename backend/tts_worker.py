"""
TTS Worker – Official PiperVoice Python API with GPU/CUDA acceleration.
Model is loaded ONCE in GPU VRAM and synthesized via voice.synthesize_wav.
"""
import asyncio
import os
import wave
import urllib.request
from pathlib import Path

DEFAULT_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkzargh/medium/pl_PL-darkzargh-medium.onnx"
DEFAULT_JSON_URL  = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkzargh/medium/pl_PL-darkzargh-medium.onnx.json"


class TTSWorker:
    """Piper TTS singleton using official PiperVoice Python API."""

    def __init__(self):
        self.model = None
        self.current_model_path: str | None = None
        self._voice_lock = asyncio.Lock()

    # ──────────────────────────────────────────────────────────────────────
    async def load_model(self):
        """Load default Piper TTS voice model into GPU memory ONCE at startup."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_default_model_sync)

    def _load_default_model_sync(self):
        voices_dir = Path("/app/voices")
        voices_dir.mkdir(exist_ok=True, parents=True)

        onnx_files = list(voices_dir.glob("*.onnx"))
        if onnx_files:
            target_onnx = onnx_files[0]
        else:
            target_onnx = voices_dir / "pl_PL-darkzargh-medium.onnx"
            target_json = voices_dir / "pl_PL-darkzargh-medium.onnx.json"
            if not target_onnx.exists():
                print("[TTS] Downloading default Piper Polish voice model...")
                try:
                    urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(target_onnx))
                    urllib.request.urlretrieve(DEFAULT_JSON_URL, str(target_json))
                    print("[TTS] Default Piper model downloaded successfully.")
                except Exception as e:
                    print(f"[TTS] Download failed: {e}")

        if target_onnx.exists():
            self._load_piper_voice_sync(str(target_onnx))

    def _load_piper_voice_sync(self, model_path: str):
        try:
            from piper import PiperVoice

            onnx_path = str(Path(model_path).with_suffix(".onnx")) if not model_path.endswith(".onnx") else model_path
            print(f"[TTS] Loading PiperVoice into GPU VRAM (use_cuda=True): {onnx_path}")

            # Load ONNX model ONCE on GPU VRAM using official API
            try:
                self.model = PiperVoice.load(onnx_path, use_cuda=True)
                print("[TTS] PiperVoice loaded successfully on GPU (CUDA)!")
            except Exception as cuda_err:
                print(f"[TTS] CUDA load warning ({cuda_err}), loading on CPU...")
                self.model = PiperVoice.load(onnx_path, use_cuda=False)
                print("[TTS] PiperVoice loaded on CPU.")

            self.current_model_path = onnx_path

        except Exception as e:
            print(f"[TTS] Error loading PiperVoice ({model_path}): {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    # ──────────────────────────────────────────────────────────────────────
    async def set_voice(self, path: str):
        """Switch Piper ONNX model in GPU VRAM."""
        async with self._voice_lock:
            p = Path(path)
            if p.suffix.lower() in (".onnx", ".json"):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_piper_voice_sync, str(p))

    # ──────────────────────────────────────────────────────────────────────
    def generate_sync(
        self,
        text: str,
        output_path: str,
        num_step: int = 16,
        speed: float = 1.0,
    ) -> bool:
        """
        Synthesize text using the loaded PiperVoice model in GPU memory.
        """
        if self.model is None:
            print("[TTS] PiperVoice model not loaded – skipping generation.")
            return False

        try:
            from piper.config import SynthesisConfig

            length_scale = 1.0 / max(0.2, float(speed))
            syn_config = SynthesisConfig(length_scale=length_scale)

            with wave.open(output_path, "wb") as wav_file:
                self.model.synthesize_wav(text, wav_file, syn_config=syn_config)

            print(f"[TTS] PiperVoice generated: {Path(output_path).name} (speed={speed}x)")
            return True

        except Exception as e:
            print(f"[TTS] PiperVoice generation error: {e}")
            import traceback
            traceback.print_exc()
            return False
