"""
Vision Worker – Multimodal Vision AI (Qwen/Qwen2-VL-2B-Instruct) for cropped subtitle image OCR.
Reads text directly from cropped image pixels using vision-language intelligence.
"""
import asyncio
import io
import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")


class VisionWorker:
    """Singleton worker for local Qwen2-VL Vision AI OCR."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.is_ready = False
        self._lock = asyncio.Lock()

    # ──────────────────────────────────────────────────────────────────────
    async def load_model(self):
        """Load Qwen2-VL model asynchronously onto CUDA GPU."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            print(f"[Vision-AI] Loading Qwen2-VL model ({MODEL_NAME}) on {device.upper()} ({gpu_name})...")

            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            try:
                # Try loading directly from local disk cache (instant offline load, no network check)
                self.processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=True,
                )
            except Exception:
                # Download model on first run if not cached locally
                self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                )

            self.model.eval()
            self.is_ready = True

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            actual_device = next(self.model.parameters()).device
            print(f"[Vision-AI] Qwen2-VL model loaded successfully on target device: {actual_device} ({gpu_name})!")

        except Exception as e:
            print(f"[Vision-AI] Qwen2-VL model load notice ({e}).")
            self.is_ready = False

    # ──────────────────────────────────────────────────────────────────────
    def process_cropped_image_sync(self, image_bytes: bytes, ocr_suggestion: str = "") -> str:
        """
        Extract & correct text from cropped subtitle ROI image using Qwen2-VL visual pixels.
        """
        if not image_bytes or not self.is_ready or self.model is None or self.processor is None:
            return ocr_suggestion

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            prompt = (
                "OCR this image, return only the text in Polish, do not add any comments or explanations"
                # "Sugerowany odczyt z OCR: '" + ocr_suggestion + "'. "
                # "Zwróć TYLKO I WYŁĄCZNIE sam odczytany tekst po polsku bez żadnych komentarzy."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.4,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            input_len = inputs.input_ids.shape[1]
            generated_tokens = generated_ids[0][input_len:]
            output_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            output_text = output_text.strip('"\'\n ')

            print(f"[Vision-AI] Qwen2-VL result: '{output_text}' (PaddleOCR suggestion: '{ocr_suggestion}')")
            return output_text if output_text else ocr_suggestion

        except Exception as e:
            print(f"[Vision-AI] Vision process error: {e}")
            return ocr_suggestion
