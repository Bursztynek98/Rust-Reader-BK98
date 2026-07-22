"""
Gemma / AI LLM Worker – Local open AI model for OCR text correction.
Fixes Polish diacritics, typos, and OCR artifacts before passing text to TTS.
Strictly constrained prompt & zero-conversation output filter.
"""
import asyncio
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Open, un-gated, ultra-fast model for Polish text correction (< 1GB VRAM, ~5ms response)
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")


class GemmaWorker:
    """Singleton worker for local AI LLM OCR text correction."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.is_ready = False
        self._lock = asyncio.Lock()

    # ──────────────────────────────────────────────────────────────────────
    async def load_model(self):
        """Load AI model asynchronously onto CUDA GPU."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            print(f"[AI-Correction] Loading AI model ({MODEL_NAME}) on {device.upper()} ({gpu_name})...")

            token = os.getenv("HF_TOKEN", None)
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                token=token,
            )
            self.model.eval()
            self.is_ready = True

            actual_device = next(self.model.parameters()).device
            print(f"[AI-Correction] AI model loaded successfully on target device: {actual_device} ({gpu_name})!")

        except Exception as e:
            print(f"[AI-Correction] Model load notice ({e}). Using heuristic OCR fixer.")
            self.is_ready = False

    # ──────────────────────────────────────────────────────────────────────
    def correct_text_sync(self, text: str) -> str:
        """
        Correct raw OCR text into clean Polish grammar and diacritics.
        Enforces strict output format with zero conversational prefix.
        """
        if not text or not text.strip():
            return ""

        raw_clean = text.strip()

        if not self.is_ready or self.model is None or self.tokenizer is None:
            return self._heuristic_fix(raw_clean)

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Jesteś precyzyjnym algorytmem korekty błędów OCR dla polskich napisów filmowych. "
                        "Twoim JEDYNYM zadaniem jest zwrócić poprawiony tekst (dodanie polskich znaków ą, ę, ć, ł, ń, ó, ś, ż, ź, naprawienie literówek). "
                        "BEZWZGLĘDNIE ZAKAZANE jest dodawanie komentarzy, przedmów, wyjaśnień lub cudzysłowów. "
                        "Odpowiedz WYŁĄCZNIE poprawionym tekstem."
                    )
                },
                {"role": "user", "content": "O, to nie rnoie bye rnoziwe"},
                {"role": "assistant", "content": "O, to nie może być możliwe"},
                {"role": "user", "content": raw_clean}
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    temperature=0.01,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_len:]
            corrected = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Clean any conversational intro phrases if small model ever produces them
            intro_patterns = [
                r"^(oto|poprawiony|poprawiona|wersja|tekst|korekta|odpowiedź)[:\s-]+",
                r"^poprawiony tekst[:\s-]+",
                r"^tekst po korekcie[:\s-]+",
            ]
            for pat in intro_patterns:
                corrected = re.sub(pat, "", corrected, flags=re.IGNORECASE).strip()

            corrected = corrected.strip('"\'\n ')
            print(f"[AI-Correction] Corrected: '{raw_clean}' → '{corrected}'")
            return corrected if corrected else raw_clean

        except Exception as e:
            print(f"[AI-Correction] Correction exception: {e}")
            return self._heuristic_fix(raw_clean)

    def _heuristic_fix(self, text: str) -> str:
        """Quick heuristic fix for common OCR typos."""
        fixed = text
        replacements = {
            "rnoie": "może",
            "rnoziwe": "możliwe",
            "rn": "m",
            "cl": "d",
            "vv": "w",
        }
        for k, v in replacements.items():
            fixed = re.sub(r'\b' + re.escape(k) + r'\b', v, fixed)
        return fixed
