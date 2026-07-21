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
    """PaddleOCR singleton z zoptymalizowanym preprocesorem dla napisów filmowych."""

    def __init__(self):
        # Wyłączamy use_angle_cls dla znacznego przyspieszenia detekcji napisów
        self.ocr = PaddleOCR(
            use_gpu=True,
            lang="pl",
            use_angle_cls=False,
            show_log=False,
        )
        print("[OCR] PaddleOCR initialized on GPU (fast mode).")

    @staticmethod
    def _smart_resize(img: np.ndarray, target_height: int = 140) -> np.ndarray:
        """Dynamiczne skalowanie do optymalnej wysokości dla rozpoznawania czcionki."""
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img

        # Jeśli obraz jest mały, powiększamy go do optymalnej wysokości target_height
        if h < target_height:
            scale = target_height / float(h)
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
        return img

    def preprocess(self, img: np.ndarray, settings: dict) -> np.ndarray:
        """
        Apply user-controlled preprocessing pipeline:
          1. Brightness / Contrast
          2. CLAHE (Adaptive Histogram Equalization)
          3. Gaussian blur (noise reduction)
          4. Sharpening
          5. Binarisation (none / otsu / adaptive)
          6. Smart scaling
        """
        brightness = float(settings.get("brightness", 0))        # -100 .. +100
        contrast   = float(settings.get("contrast", 100)) / 100  # 50..300 → 0.5..3.0
        blur       = int(settings.get("blur", 0))                 # 0..5
        sharpen    = int(settings.get("sharpen", 0))              # 0..10
        threshold  = settings.get("threshold", "none")            # none|otsu|adaptive

        # 1. Kontrast i Jasność (f(x) = α·x + β)
        img_out = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

        # 2. CLAHE (Adaptacyjny kontrast dla napisów na zmiennym tle)
        if settings.get("clahe", False):
            lab = cv2.cvtColor(img_out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 3. Odszumianie Gaussian Blur
        if blur > 0:
            k = blur * 2 + 1
            img_out = cv2.GaussianBlur(img_out, (k, k), 0)

        # 4. Wyostrzanie (Unsharp mask)
        if sharpen > 0:
            strength = sharpen / 10.0          # 0.1 .. 1.0
            blurred  = cv2.GaussianBlur(img_out, (0, 0), 3)
            img_out  = cv2.addWeighted(img_out, 1 + strength, blurred, -strength, 0)

        # 5. Binaryzacja
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

        return self._smart_resize(img_out)

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
            # cls=False przyspiesza wnioskowanie dla napisów (brak obrotu 180°)
            result = self.ocr.ocr(img_proc, cls=False)
        except Exception as e:
            print(f"[OCR] Inference error: {e}")
            return "", 0.0, preview_b64

        if not result or not result[0]:
            return "", 0.0, preview_b64

        # ── Sort lines top-to-bottom ─────────────────────────────────────
        lines = result[0]
        try:
            lines = sorted(lines, key=lambda l: l[0][0][1])
        except Exception:
            pass

        texts: list[str] = []
        confs: list[float] = []

        for line in lines:
            if line and len(line) >= 2:
                text_conf = line[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    texts.append(str(text_conf[0]))
                    confs.append(float(text_conf[1]))

        full_text = " ".join(texts).strip()
        full_text = _normalize(full_text)
        avg_conf  = float(np.mean(confs)) if confs else 0.0

        return full_text, avg_conf, preview_b64
