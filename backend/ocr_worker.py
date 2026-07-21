"""
OCR Worker – PaddleOCR GPU pipeline with OpenCV preprocessing.
Model is loaded once as a singleton and reused for all frames.
"""
import base64
import numpy as np
import cv2
from paddleocr import PaddleOCR

from backend.diff_utils import _normalize


class OCRWorker:
    """PaddleOCR singleton with configurable image preprocessing."""

    def __init__(self):
        # Load model once – GPU inference
        self.ocr = PaddleOCR(
            use_gpu=True,
            lang="pl",               # Latin charset – best for most subtitles
            use_angle_cls=True,      # detect rotated text
            show_log=False,
            # det_db_thresh=0.3,
            # det_db_box_thresh=0.5,
            # rec_batch_num=6,
        )
        print("[OCR] PaddleOCR initialized on GPU.")

    # ──────────────────────────────────────────────────────────────────────
    def _upscale(img: np.ndarray, factor: float = 2.5) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * factor), int(h * factor)),
                        interpolation=cv2.INTER_LANCZOS4)
    
    # ──────────────────────────────────────────────────────────────────────
    def preprocess(self, img: np.ndarray, settings: dict) -> np.ndarray:
        """
        Apply user-controlled preprocessing pipeline:
          1. Brightness / Contrast
          2. Gaussian blur (noise reduction)
          3. Sharpening
          4. Optional binarisation (none / otsu / adaptive)
        """
        brightness = float(settings.get("brightness", 0))        # -100 .. +100
        contrast   = float(settings.get("contrast", 100)) / 100  # 50..300 → 0.5..3.0
        blur       = int(settings.get("blur", 0))                 # 0..5
        sharpen    = int(settings.get("sharpen", 0))              # 0..10
        threshold  = settings.get("threshold", "none")            # none|otsu|adaptive

        # Brightness + Contrast  (f(x) = α·x + β)
        img_out = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

        # Gaussian blur (denoise)
        if blur > 0:
            k = blur * 2 + 1
            img_out = cv2.GaussianBlur(img_out, (k, k), 0)

        # Unsharp mask sharpening
        if sharpen > 0:
            strength = sharpen / 10.0          # 0.1 .. 1.0
            blurred  = cv2.GaussianBlur(img_out, (0, 0), 3)
            img_out  = cv2.addWeighted(img_out, 1 + strength, blurred, -strength, 0)

        # Binarisation
        if threshold in ("otsu", "adaptive"):
            gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY) if img_out.ndim == 3 else img_out
            if threshold == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                binary = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                    blockSize=15, C=8
                )
            img_out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        return self._upscale(img_out)

    # ──────────────────────────────────────────────────────────────────────
    def process(
        self,
        img_bytes: bytes,
        roi: dict | None,
        settings: dict,
    ) -> tuple[str, float, str]:
        """
        Main entry point: decode JPEG → crop ROI → preprocess → OCR.

        Returns:
            (full_text, avg_confidence 0..1, preview_jpeg_base64)
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return "", 0.0, ""

        h, w = img.shape[:2]

        # ── Crop to Region of Interest ──────────────────────────────────
        if roi and roi.get("w", 0) > 0 and roi.get("h", 0) > 0:
            x  = max(0, int(roi["x"] * w))
            y  = max(0, int(roi["y"] * h))
            rw = min(int(roi["w"] * w), w - x)
            rh = min(int(roi["h"] * h), h - y)
            if rw > 10 and rh > 10:
                img = img[y : y + rh, x : x + rw]

        # ── Preprocessing ───────────────────────────────────────────────
        img_proc = self.preprocess(img, settings)

        # ── Preview for frontend (compressed JPEG) ───────────────────────
        _, preview_buf = cv2.imencode(
            ".jpg", img_proc, [cv2.IMWRITE_JPEG_QUALITY, 35]
        )
        preview_b64 = base64.b64encode(preview_buf.tobytes()).decode()

        # ── OCR inference ────────────────────────────────────────────────
        try:
            result = self.ocr.ocr(img_proc, cls=True)
        except Exception as e:
            print(f"[OCR] Inference error: {e}")
            return "", 0.0, preview_b64

        if not result or not result[0]:
            return "", 0.0, preview_b64

        texts: list[str] = []
        confs: list[float] = []

        for line in result[0]:
            if line and len(line) >= 2:
                text_conf = line[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    texts.append(str(text_conf[0]))
                    confs.append(float(text_conf[1]))

        full_text  = " ".join(texts).strip()
        full_text = _normalize(full_text)
        avg_conf   = float(np.mean(confs)) if confs else 0.0

        return full_text, avg_conf, preview_b64
