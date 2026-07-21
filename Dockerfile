FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

# ── Environment ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
# Ubuntu 24.04 uses PEP 668 "externally managed" python.
# In Docker containers we own the whole system, so allow pip to install globally.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ── System dependencies ───────────────────────────────────
# Note: Ubuntu 24.04 renames libgl1-mesa-glx → libgl1, libxrender-dev → libxrender1
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
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 ships Python 3.12 as default python3
RUN python3 --version

# ── Pip upgrade ───────────────────────────────────────────
# Ubuntu 24.04 installs pip via debian (no RECORD file) → use --ignore-installed
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip setuptools wheel --ignore-installed

WORKDIR /app

# ── PaddlePaddle GPU (CUDA 13.0) ─────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install paddlepaddle-gpu \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# ── PyTorch GPU (CUDA 13.0) ───────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# ── App requirements (everything except omnivoice) ────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v '^omnivoice' requirements.txt > /tmp/req_no_omni.txt && \
    pip3 install -r /tmp/req_no_omni.txt

# ── OmniVoice – installed WITHOUT deps ────────────────────
# Prevents pip from pulling CPU-torch or wrong transformers version
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-deps omnivoice

# ── Sanity check – printed during build ───────────────────
RUN python3 -c " \
import sys, torch, transformers; \
print('=== Build verification ==='); \
print(f'Python:       {sys.version}'); \
print(f'torch:        {torch.__version__}'); \
print(f'CUDA runtime: {torch.version.cuda}'); \
print(f'transformers: {transformers.__version__}'); \
print(f'sys.get_int_max_str_digits: {sys.get_int_max_str_digits()}'); \
print('========================='); \
"

# ── Copy application ──────────────────────────────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN mkdir -p /app/voices /app/audio_cache

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
