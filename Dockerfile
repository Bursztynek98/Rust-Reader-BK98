FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

# ── Environment ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ── System dependencies ───────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libsndfile1 \
    espeak-ng \
    libespeak-ng-dev \
    curl \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# ── Pip upgrade ───────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip setuptools wheel --ignore-installed

WORKDIR /app

# ── PaddlePaddle GPU (CUDA 13.0) ─────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install paddlepaddle-gpu==3.3.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# ── PyTorch GPU (CUDA 13.0) ───────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# ── App requirements ──────────────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# ── Piper TTS Python API ──────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install piper-tts --no-deps

# ── Sanity check – printed during build ───────────────────
RUN python3 -c "import sys, piper; print('=== Build verification ==='); print(f'Python: {sys.version}'); print('PiperVoice API ready'); print('=========================');"

# ── Copy application ──────────────────────────────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN mkdir -p /app/voices /app/audio_cache

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
