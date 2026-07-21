"""
TTS Worker – OmniVoice GPU pipeline z cache'owaniem promptu głosu.
Wagi modelu oraz embedding głosu są przechowywane w pamięci VRAM.
Głos referencyjny jest enkodowany TYLKO RAZ po jego wybraniu/wgraniu.
"""
import asyncio
import threading
from pathlib import Path


class TTSWorker:
    """OmniVoice singleton – GPU inference z trwałym embeddingiem głosu."""

    def __init__(self):
        self.model = None
        self.ref_audio_path: str | None = None
        self.cached_prompt = None
        self._voice_lock = asyncio.Lock()
        self._gpu_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────
    async def load_model(self):
        """Load OmniVoice model asynchronously (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self):
        import traceback
        device = "cpu"
        try:
            import torch
            print(f"[TTS] torch version: {torch.__version__}")
            if torch.cuda.is_available():
                device = "cuda:0"
                print(f"[TTS] GPU: {torch.cuda.get_device_name(0)}")
                print(f"[TTS] CUDA runtime: {torch.version.cuda}")
            else:
                print("[TTS] CUDA not available – falling back to CPU.")
        except Exception as te:
            print(f"[TTS] torch check failed: {te}")

        if device == "cpu":
            print("[TTS] WARNING: OmniVoice is designed for GPU inference. Performance may be slow on CPU.")
            raise RuntimeError("OmniVoice requires a CUDA-enabled GPU for optimal performance.")
        try:
            from omnivoice import OmniVoice
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map=device,
            )
            print(f"[TTS] OmniVoice model loaded successfully on {device}.")
        except Exception as e:
            print(f"[TTS] WARNING: Failed to load OmniVoice: {e}")
            print("[TTS] Full traceback:")
            traceback.print_exc()
            self.model = None

    # ──────────────────────────────────────────────────────────────────────
    async def set_voice(self, path: str):
        """
        Ustawia nowy głos referencyjny.
        Konwertuje audio do WAV 22050Hz Mono i PRE-ENKODUJE embedding głosu RAZ.
        """
        async with self._voice_lock:
            loop = asyncio.get_event_loop()
            wav_path = await loop.run_in_executor(None, self._to_wav, path)
            self.ref_audio_path = wav_path

            # Pre-kompilacja / wyciągnięcie promptu głosu do VRAM
            if self.model is not None:
                await loop.run_in_executor(None, self._cache_voice_prompt_sync)

            print(f"[TTS] Voice reference & embedding ready → {wav_path}")

    def _cache_voice_prompt_sync(self):
        """Pre-enkodowanie próbki głosu raz przy jej wyborze."""
        if not self.ref_audio_path or self.model is None:
            return

        with self._gpu_lock:
            try:
                # W zależnosci od wersji OmniVoice, prompt tworzony jest z ref_audio
                if hasattr(self.model, "create_voice_clone_prompt"):
                    self.cached_prompt = self.model.create_voice_clone_prompt(ref_audio=self.ref_audio_path)
                    print("[TTS] Pre-computed voice prompt cached via create_voice_clone_prompt()")
                elif hasattr(self.model, "encode_prompt"):
                    self.cached_prompt = self.model.encode_prompt(ref_audio=self.ref_audio_path)
                    print("[TTS] Pre-computed voice prompt cached via encode_prompt()")
                else:
                    self.cached_prompt = None
                    print("[TTS] Standard ref_audio mode (no create_prompt method found)")
            except Exception as e:
                print(f"[TTS] Prompt caching warning: {e}, falling back to ref_audio path")
                self.cached_prompt = None

    def _to_wav(self, path: str) -> str:
        """Zawsze konwertuje plik audio do formatu 22050 Hz Mono WAV."""
        p = Path(path)
        wav_path = str(p.with_name(f"{p.stem}_mono22k.wav"))

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(p))
            # Wymuszamy 22050 Hz oraz 1 kanał (mono) dla OmniVoice
            audio = audio.set_frame_rate(22050).set_channels(1)
            audio.export(wav_path, format="wav")
            print(f"[TTS] Converted {p.name} → {Path(wav_path).name} (22050Hz Mono)")
            return wav_path
        except Exception as e:
            print(f"[TTS] Conversion error: {e}, using original path")
            return path

    # ──────────────────────────────────────────────────────────────────────
    def generate_sync(
        self,
        text: str,
        output_path: str,
        num_step: int = 16,
        speed: float = 1.0,
    ) -> bool:
        """
        Generuje audio dla tekstu. Używa zapamiętanego w VRAM promptu głosu.
        """
        if self.model is None:
            print("[TTS] Model nie jest załadowany – pomijam generowanie.")
            return False

        if self.ref_audio_path is None:
            print("[TTS] Brak głosu referencyjnego – wgraj najpierw plik audio.")
            return False

        with self._gpu_lock:
            try:
                # Jeśli wygenerowano wcześniej prompt głosu, przekazujemy go bezpośrednio
                if self.cached_prompt is not None:
                    result = self.model.generate(
                        text=text,
                        voice_clone_prompt=self.cached_prompt,
                        num_step=int(num_step),  # diffusion steps (or 16 for faster inference)
                        speed=float(speed),     # speed factor (>1.0 faster, <1.0 slower)
                    )
                else:
                    result = self.model.generate(
                        text=text,
                        ref_audio=self.ref_audio_path,
                        num_step=int(num_step),  # diffusion steps (or 16 for faster inference)
                        speed=float(speed),     # speed factor (>1.0 faster, <1.0 slower)
                    )

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

                # Zapewniamy poprawny kształt macierzy dla soundfile
                if audio_data.ndim > 1:
                    audio_data = audio_data.squeeze()
                if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]:
                    audio_data = audio_data.T

                sf.write(output_path, audio_data, sr)
                print(f"[TTS] Wygenerowano: {Path(output_path).name} ({len(audio_data)/sr:.1f}s)")
                return True

            except Exception as e:
                print(f"[TTS] Błąd generowania audio: {e}")
                return False
