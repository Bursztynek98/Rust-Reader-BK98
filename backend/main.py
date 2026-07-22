"""
SubVoice – Real-time OCR Subtitle Audio Generator
FastAPI backend: WebSocket frame processing, audio serving, voice upload
"""
import asyncio
import base64
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.ocr_worker import OCRWorker
from backend.tts_worker import TTSWorker
from backend.gemma_worker import GemmaWorker
from backend.diff_utils import has_significant_change

# ─── Paths ─────────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
AUDIO_DIR = Path("/app/audio_cache")
VOICES_DIR = Path("/app/voices")

AUDIO_DIR.mkdir(exist_ok=True, parents=True)
VOICES_DIR.mkdir(exist_ok=True, parents=True)

# ─── Global workers (singletons) ────────────────────────────────────────────
ocr_worker: OCRWorker | None = None
tts_worker: TTSWorker | None = None
gemma_worker: GemmaWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_worker, tts_worker, gemma_worker
    loop = asyncio.get_event_loop()

    print("[SubVoice] Loading OCR model (PaddleOCR GPU)...")
    ocr_worker = await loop.run_in_executor(None, OCRWorker)
    print("[SubVoice] OCR model ready.")

    print("[SubVoice] Loading TTS model (Piper GPU)...")
    tts_worker = TTSWorker()
    await tts_worker.load_model()
    print("[SubVoice] TTS model ready.")

    print("[SubVoice] Loading Gemma AI text correction model...")
    gemma_worker = GemmaWorker()
    await gemma_worker.load_model()
    print("[SubVoice] Gemma AI ready.")

    # Auto-load first available voice file
    voice_files = list(VOICES_DIR.glob("*.[mMwWoOfF][pPaAgGlL][3344acACPP]*"))
    if voice_files:
        await tts_worker.set_voice(str(voice_files[0]))
        print(f"[SubVoice] Auto-loaded voice: {voice_files[0].name}")

    print("[SubVoice] All systems ready!")
    yield
    print("[SubVoice] Shutting down...")


app = FastAPI(lifespan=lifespan, title="SubVoice", version="1.0.0")

# Serve generated audio files
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


# ─── Frontend routes ─────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/style.css")
async def serve_css():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

@app.get("/app.js")
async def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


# ─── API routes ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    except Exception:
        cuda_ok = False
        gpu_name = None

    return {
        "status": "ok",
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "ocr_ready": ocr_worker is not None,
        "tts_ready": tts_worker is not None and tts_worker.model is not None,
        "voice_loaded": tts_worker.current_model_path if tts_worker else None,
        "voice_filename": Path(tts_worker.current_model_path).name if (tts_worker and tts_worker.current_model_path) else None,
    }


@app.get("/voices-list")
async def list_voices():
    """List all voice model (.onnx) and audio files in /app/voices/ folder."""
    allowed = {".onnx", ".mp3", ".wav", ".ogg", ".flac", ".m4a"}
    files = [
        f.name for f in VOICES_DIR.iterdir()
        if f.suffix.lower() in allowed and f.is_file()
    ]
    return sorted(files)


@app.post("/select-voice")
async def select_voice(payload: dict):
    """Select an existing voice model (.onnx) from /app/voices/ by filename."""
    filename = payload.get("filename", "")
    path = VOICES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": f"File not found: {filename}"}, status_code=404)
    await tts_worker.set_voice(str(path))
    return {"status": "ok", "filename": filename}


@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    """Upload a new Piper voice model (.onnx / .json) or audio file."""
    allowed_exts = {".onnx", ".json", ".mp3", ".wav", ".ogg", ".flac", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        return JSONResponse({"error": f"Unsupported format: {ext}"}, status_code=400)

    dest = VOICES_DIR / file.filename
    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)

    await tts_worker.set_voice(str(dest))

    return {
        "status": "ok",
        "filename": file.filename,
        "path": str(dest),
        "voice_ready": tts_worker.model is not None,
    }


@app.post("/tts-test")
async def tts_test(payload: dict):
    """Generate audio from arbitrary text for voice model testing."""
    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Brak tekstu"}, status_code=400)
    if len(text) > 500:
        return JSONResponse({"error": "Tekst za długi (max 500 znaków)"}, status_code=400)
    if tts_worker is None or tts_worker.model is None:
        return JSONResponse({"error": "Model TTS nie jest załadowany"}, status_code=503)
    # if tts_worker.ref_audio_path is None:
    #     return JSONResponse({"error": "Brak wybranego głosu – wczytaj plik MP3/WAV"}, status_code=400)

    num_step = int(payload.get("num_step", 16))
    speed = float(payload.get("speed", 1.0))
    loop = asyncio.get_event_loop()
    audio_filename = f"test_{uuid.uuid4().hex[:8]}.wav"
    audio_path = AUDIO_DIR / audio_filename

    success = await loop.run_in_executor(
        None, tts_worker.generate_sync, text, str(audio_path), num_step, speed
    )

    if not success:
        return JSONResponse({"error": "Generowanie audio nie powiodło się"}, status_code=500)

    return {"status": "ok", "url": f"/audio/{audio_filename}", "text": text}


# ─── WebSocket endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    last_text: str = ""

    async def push(data: dict):
        """Helper: send JSON to client, ignore if disconnected."""
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "frame":
                img_b64: str = msg["image"]
                roi: dict | None = msg.get("roi")        # {x,y,w,h} normalized 0..1
                settings: dict = msg.get("settings", {})

                # Decode JPEG bytes – strip data-URI prefix if present
                if img_b64.startswith("data:"):
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                print(f"[WS] Frame recv: {len(img_bytes)//1024}KB roi={roi} settings={settings}")

                # OCR (blocking – run in thread pool)
                try:
                    text, confidence, preview_b64 = await loop.run_in_executor(
                        None, ocr_worker.process, img_bytes, roi, settings
                    )
                    print(f"[WS] OCR result: conf={confidence:.2f} text='{text[:60]}'")
                except Exception as ocr_err:
                    print(f"[WS] OCR ERROR: {ocr_err}")
                    await push({"type": "error", "message": f"OCR error: {ocr_err}"})
                    continue

                # Raw text output straight from PaddleOCR
                raw_ocr_text = text.strip()

                # Change detection & threshold comparison MUST be based on raw OCR output
                changed = has_significant_change(last_text, raw_ocr_text)

                gemma_enabled = settings.get("gemma_correction", False)
                final_text = raw_ocr_text

                # Apply AI correction only when changed and raw OCR text is non-empty
                if changed and raw_ocr_text:
                    last_text = raw_ocr_text
                    if gemma_enabled and gemma_worker:
                        corrected = await loop.run_in_executor(None, gemma_worker.correct_text_sync, raw_ocr_text)
                        if corrected:
                            final_text = corrected

                await push({
                    "type": "ocr_result",
                    "raw_text": raw_ocr_text,
                    "text": final_text,
                    "confidence": round(confidence * 100, 1),
                    "preview": preview_b64,
                    "changed": changed,
                    "gemma_corrected": gemma_enabled and (raw_ocr_text != final_text),
                })

                # Trigger TTS generation on final_text
                if changed and raw_ocr_text:
                    audio_filename = f"{uuid.uuid4().hex}.wav"
                    audio_path = AUDIO_DIR / audio_filename
                    speed = float(settings.get("speed", 1.0))

                    async def _gen(txt=final_text, raw_txt=raw_ocr_text, path=audio_path, fname=audio_filename, spd=speed):
                        success = await loop.run_in_executor(
                            None, tts_worker.generate_sync, txt, str(path), 16, spd
                        )
                        if success:
                            await push({
                                "type": "audio_ready",
                                "url": f"/audio/{fname}",
                                "raw_text": raw_txt,
                                "text": txt,
                                "gemma_corrected": gemma_enabled and (raw_txt != txt),
                            })
                        else:
                            await push({
                                "type": "tts_error",
                                "text": txt,
                                "message": "TTS generation failed (check model/voice)",
                            })

                    asyncio.create_task(_gen())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
        await push({"type": "error", "message": str(e)})
