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
        # Wyłączamy use_angle_cls dla przyspieszenia, dodajemy optymalizację pod napisy
        self.ocr = PaddleOCR(
            use_gpu=True,
            lang="pl",
            use_angle_cls=False,
            show_log=False,
            det_db_box_thresh=0.35,       # Wykrywa drobniejsze/cieńsze czcionki
            det_db_unclip_ratio=2.0,      # Poszerza margines ramek detekcji (polskie znaki ś, ż, ą)
            rec_image_shape="3, 48, 640", # Szeroki bufor rozpoznawania dla długich zdań
        )
        self._last_frame = None
        self._last_text = ""
        self._last_conf = 0.0
        self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        print("[OCR] PaddleOCR initialized on GPU (subtitle-optimized mode).")

    @staticmethod
    def _add_padding(img: np.ndarray, pad: int = 25) -> np.ndarray:
        """Dodaje czarną ramkę wokół kadru, aby litery przy krawędziach nie były obcinane przez DBNet."""
        return cv2.copyMakeBorder(
            img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

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
          1. Max-channel (yellow/white subtitle extraction)
          2. Erode (glow/bloom removal)
          3. Brightness / Contrast
          4. CLAHE (Adaptive Histogram Equalization)
          5. Gaussian blur (noise reduction)
          6. Sharpening
          7. Binarisation (none / otsu / adaptive)
          8. Smart scaling + 25px Border padding
        """
        brightness = float(settings.get("brightness", 0))        # -100 .. +100
        contrast   = float(settings.get("contrast", 100)) / 100  # 50..300 → 0.5..3.0
        blur       = int(settings.get("blur", 0))                 # 0..5
        sharpen    = int(settings.get("sharpen", 0))              # 0..10
        threshold  = settings.get("threshold", "none")            # none|otsu|adaptive

        # 1. Max-Channel Preprocessing (Żółte i białe napisy → max pikseli)
        if settings.get("max_channel", False):
            max_ch = np.max(img[:, :, :3], axis=2)
            img = cv2.cvtColor(max_ch, cv2.COLOR_GRAY2BGR)

        # 2. Erozja poświaty (glow/bloom removal)
        if settings.get("erode", False):
            img = cv2.erode(img, self._erode_kernel, iterations=1)

        # 3. CLAHE (Adaptacyjny kontrast dla napisów na zmiennym tle)
        if settings.get("clahe", False):
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 4. Kontrast i Jasność (f(x) = α·x + β)
        img_out = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

        # 5. Odszumianie Gaussian Blur
        if blur > 0:
            k = blur * 2 + 1
            img_out = cv2.GaussianBlur(img_out, (k, k), 0)

        # 6. Wyostrzanie (Unsharp mask)
        if sharpen > 0:
            strength = sharpen / 10.0          # 0.1 .. 1.0
            blurred  = cv2.GaussianBlur(img_out, (0, 0), 3)
            img_out  = cv2.addWeighted(img_out, 1 + strength, blurred, -strength, 0)

        # 7. Binaryzacja
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

        return self._add_padding(self._smart_resize(img_out))

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

        # ── OPTIMIZATION: Skip OCR on unchanged frames ────────────────────
        if settings.get("skip_ocr", False) and self._last_frame is not None and self._last_frame.shape == img_proc.shape:
            score = np.sum(cv2.absdiff(img_proc, self._last_frame)) / img_proc.size
            if score < 3.0:
                _, preview_buf = cv2.imencode(".jpg", img_proc, [cv2.IMWRITE_JPEG_QUALITY, 40])
                p_b64 = base64.b64encode(preview_buf.tobytes()).decode()
                return self._last_text, self._last_conf, p_b64, preview_buf.tobytes()

        self._last_frame = img_proc.copy()

        # ── OCR inference ────────────────────────────────────────────────
        try:
            # cls=False przyspiesza wnioskowanie dla napisów (brak obrotu 180°)
            result = self.ocr.ocr(img_proc, cls=False)
        except Exception as e:
            print(f"[OCR] Inference error: {e}")
            _, preview_buf = cv2.imencode(".jpg", img_proc, [cv2.IMWRITE_JPEG_QUALITY, 40])
            p_b64 = base64.b64encode(preview_buf.tobytes()).decode()
            return "", 0.0, p_b64, preview_buf.tobytes()

        if not result or not result[0]:
            self._last_text = ""
            self._last_conf = 0.0
            _, preview_buf = cv2.imencode(".jpg", img_proc, [cv2.IMWRITE_JPEG_QUALITY, 40])
            p_b64 = base64.b64encode(preview_buf.tobytes()).decode()
            return "", 0.0, p_b64, preview_buf.tobytes()

        # ── Sort lines top-to-bottom & extract tight bounding box ──────
        lines = result[0]
        try:
            lines = sorted(lines, key=lambda l: l[0][0][1])
        except Exception:
            pass

        filter_height = settings.get("filter_height", False)
        texts: list[str] = []
        confs: list[float] = []
        all_xs: list[float] = []
        all_ys: list[float] = []

        for line in lines:
            if line and len(line) >= 2:
                bbox = line[0]
                text_conf = line[1]

                # ── Filtr wysokości tekstu (opcjonalny) ──────────────────
                if filter_height and isinstance(bbox, (list, tuple)) and len(bbox) >= 3:
                    try:
                        line_h = abs(bbox[2][1] - bbox[0][1])
                        # Ignoruj śmieci poniżej 12px lub powyżej 120px
                        if line_h < 12 or line_h > 120:
                            continue
                    except Exception:
                        pass

                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    texts.append(str(text_conf[0]))
                    confs.append(float(text_conf[1]))
                    for pt in bbox:
                        all_xs.append(pt[0])
                        all_ys.append(pt[1])

        full_text = " ".join(texts).strip()
        full_text = _normalize(full_text)
        avg_conf  = float(np.mean(confs)) if confs else 0.0

        # ── Tight crop around detected text lines only ──────────────────
        if all_xs and all_ys:
            pad = 12
            h_proc, w_proc = img_proc.shape[:2]
            min_x = max(0, int(min(all_xs)) - pad)
            max_x = min(w_proc, int(max(all_xs)) + pad)
            min_y = max(0, int(min(all_ys)) - pad)
            max_y = min(h_proc, int(max(all_ys)) + pad)

            if (max_x - min_x) > 20 and (max_y - min_y) > 10:
                tight_crop = img_proc[min_y:max_y, min_x:max_x]
            else:
                tight_crop = img_proc
        else:
            tight_crop = img_proc

        # Encode tight crop as compressed JPEG
        _, preview_buf = cv2.imencode(
            ".jpg", tight_crop, [cv2.IMWRITE_JPEG_QUALITY, 55]
        )
        cropped_bytes = preview_buf.tobytes()
        preview_b64   = base64.b64encode(cropped_bytes).decode()

        self._last_text = full_text
        self._last_conf = avg_conf

        return full_text, avg_conf, preview_b64, cropped_bytes
