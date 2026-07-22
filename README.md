# SubVoice 🎙️

**Real-Time Screen Subtitle OCR → AI Voice Generator (Piper TTS & Qwen2-VL)**

SubVoice captures a selected region of your screen (e.g., subtitles from movies, anime, or video games), processes frames using GPU-accelerated **PaddleOCR** or **Qwen2-VL Vision AI**, detects text changes in real time, and synthesizes speech using **Piper TTS (ONNX GPU)** directly in your browser.

---

## 🚀 Key Features

- **WebRTC Screen Capture & Custom ROI**: Capture any screen or window and draw an interactive bounding box (ROI) directly over the subtitle area.
- **Dual OCR Engine**:
  - **PaddleOCR (GPU)**: Subtitle-optimized layout analysis and fast recognition with custom text box clipping and line sorting.
  - **Qwen2-VL Vision AI (`Qwen2-VL-2B-Instruct`)**: Multimodal vision-language model to re-read and correct difficult or stylized subtitles directly from image pixels.
- **Advanced OpenCV Image Preprocessing**:
  - **Max-Channel Extraction**: Filters yellow and white subtitle text.
  - **Glow & Bloom Removal**: Morphological erosion to clean subtitle glow.
  - **CLAHE**: Adaptive histogram equalization for dynamic backgrounds.
  - **Filters**: Brightness, contrast, Gaussian blur, unsharp mask sharpening, and Otsu / Adaptive binarization.
  - **Unchanged Frame Skipping**: Skips redundant OCR inferences when the subtitle region is visually static.
- **Smart Text Differencing**: Levenshtein distance filtering (`diff_utils`) prevents duplicate audio generation for minor OCR noise.
- **Piper TTS (ONNX GPU)**: High-speed neural speech synthesis running natively on GPU CUDA VRAM using ONNX Runtime.
  - Pre-packaged with Polish voice (`pl_PL-darkzargh-medium.onnx`).
  - Supports loading and hot-swapping any Piper ONNX voice model (`.onnx` + `.onnx.json`).
  - Built-in Voice Tester UI to preview voices with custom text.
- **Seamless Browser Audio Queue**: Client-side audio manager with automatic playback speed adjustment (up to 3×) when new subtitle audio arrives while the current clip is playing.
- **Dockerized GPU Stack**: Built on Ubuntu 24.04, CUDA 13.2 runtime, PyTorch (cu130), PaddlePaddle GPU, and ONNX Runtime GPU.

---

## ⚡ Quick Start

### 1. Prerequisites (Windows / Linux)

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 (Windows) or Docker Engine (Linux).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- NVIDIA GPU with compatible CUDA drivers.

Verify GPU passthrough in Docker:
```powershell
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

### 2. Launch the Application

```powershell
docker-compose up --build
```
> **Note**: On first startup, the application automatically downloads the default Polish Piper voice model (`pl_PL-darkzargh-medium.onnx`). If Vision AI is enabled in the UI, `Qwen/Qwen2-VL-2B-Instruct` will download from HuggingFace.

Open your browser and navigate to: **`http://localhost:8000`**

---

## 🖥️ How to Use

1. **Click "Share Screen"** → Select the screen or application window containing subtitles.
2. **Draw OCR Region** → Click and drag a rectangle over the subtitle area on the canvas.
3. **Adjust Image Filters** (if subtitles are low contrast or stylized):
   - ☀ **Brightness / Contrast**: Adjust lighting levels.
   - 🟡 **Max Channel**: Extract bright yellow/white subtitles from dark scenes.
   - ∿ **Blur & Sharpen**: Reduce noise and sharpen character edges.
   - ⬛ **Binarization**: Otsu or Adaptive thresholding.
   - 🧠 **Vision AI (Qwen2-VL)**: Enable for complex fonts or hard-to-read backgrounds.
4. **Voice Management**:
   - Select or drag-and-drop new Piper voice models (`.onnx` + `.onnx.json`) into the **Voice Model** panel.
   - Test speech generation using the **TTS Voice Tester** input.
5. **Listen**: Generated audio plays automatically via WebSocket streaming.

---

## 🏗️ Architecture

```
Browser (WebRTC getDisplayMedia)
   │
   ├─► Interactive ROI Canvas (1 FPS crop extraction)
   │
   └─► WebSocket (ws://localhost:8000/ws)
            │
            ├─► OpenCV Preprocessing (CLAHE, Max-Channel, Blur/Sharpen, Binarization)
            │
            ├─► PaddleOCR GPU / Qwen2-VL-2B (Vision AI)
            │
            ├─► Levenshtein Diff Check (diff_utils)
            │
            ├─► Piper TTS Worker (ONNX Runtime GPU / CUDA VRAM)
            │
            └─► Audio Stream (/audio/*.wav) ──► Browser Audio Queue (Auto Speed-up)
```

---

## 🔧 Advanced Configuration

### Subtitle Text Filtering (`backend/diff_utils.py`)
```python
_CHANGE_THRESHOLD = 0.70   # Levenshtein similarity threshold (0.0 to 1.0)
_MIN_TEXT_LENGTH  = 3      # Minimum character length to trigger TTS synthesis
```

### OCR Capture Interval (`frontend/app.js`)
```javascript
const OCR_INTERVAL_MS = 500;  // Milliseconds between frame captures (default: 500ms)
```

### Audio Playback Speedup (`frontend/app.js`)
```javascript
const SPEEDUP_RATE   = 3.0;   // Playback rate multiplier when audio queue overflows
const SPEEDUP_THRESH = 2.0;   // Queue duration threshold (in seconds) to activate speed-up
```

---

## 🗂️ Project Structure

```
subvoice/
├── docker-compose.yml       ← Docker services & GPU resource allocation
├── Dockerfile               ← CUDA 13.2, PyTorch GPU, PaddlePaddle & Piper TTS container
├── requirements.txt         ← FastAPI, PaddleOCR, Transformers, ONNX Runtime GPU dependencies
├── voices/                  ← Piper voice models (.onnx and .onnx.json)
├── audio_cache/             ← Generated temporary WAV audio files
├── backend/
│   ├── main.py              ← FastAPI server, WebSocket endpoint & voice upload API
│   ├── ocr_worker.py        ← PaddleOCR GPU pipeline & OpenCV preprocessing
│   ├── vision_worker.py     ← Qwen2-VL Multimodal Vision AI OCR engine
│   ├── tts_worker.py        ← Piper TTS engine (ONNX CUDA GPU inference)
│   └── diff_utils.py        ← Levenshtein text differencing & normalization
└── frontend/
    ├── index.html           ← Single-page application interface
    ├── style.css            ← Modern dark-theme glassmorphism styling
    └── app.js               ← WebRTC screen capture, WebSocket client & audio player
```
