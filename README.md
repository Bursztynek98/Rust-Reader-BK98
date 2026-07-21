# SubVoice 🎙️

**Real-time OCR Subtitle → AI Voice Generator**

Przechwytuje wybrany region ekranu (napisy z filmów/gier), co sekundę analizuje tekst przez PaddleOCR GPU, a gdy napisy się zmienią – generuje audio przez OmniVoice GPU i odtwarza je w przeglądarce.

---

## ⚡ Szybki start

### 1. Wymagania hosta (Windows)

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) z WSL2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Sterownik GPU ≥ 596.49 (RTX 5000 series)

Zweryfikuj GPU:
```powershell
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

### 2. Głos referencyjny

Umieść plik MP3/WAV (3–10 sekund, czysty głos) w folderze `voices/`:
```
voices/moj_glos.mp3
```

### 3. Uruchomienie

```powershell
docker-compose up --build
```

Otwórz przeglądarkę: **http://localhost:8000**

---

## 🖥️ Użytkowanie

1. **Kliknij "Udostępnij ekran"** → wybierz okno z napisami
2. **Narysuj region OCR** – przeciągnij prostokąt na dolnym pasku z napisami
3. **Dostosuj obraz** (jeśli OCR nie radzi sobie):
   - ☀ Jasność / ◑ Kontrast – podstawowa korekcja
   - ∿ Rozmycie – redukuje szum/pikselozę
   - ✦ Wyostrzenie – poprawia czytelność
   - ⬛ Binaryzacja – Otsu lub Adaptywna (dobra dla jasnych napisów na ciemnym tle)
4. **Słuchaj** – audio generuje się automatycznie po zmianie napisów
   - Jeśli nowe audio jest gotowe a poprzednie gra → **automatyczne przyspieszenie 3×**

### Zmiana głosu (bez restartu)

Przeciągnij nowy plik MP3/WAV na pole "Model głosu" w UI lub kliknij je.

---

## 🏗️ Architektura

```
Przeglądarka ─── WebRTC getDisplayMedia ──→ Canvas
     │                                        │
     │ ← WebSocket (text/audio) ←─────────────┤
     │                                        ↓ 1kl/s JPEG
   FastAPI ────→ PaddleOCR GPU ──→ diff_utils ──→ TTSWorker
                                                       │
                                               OmniVoice GPU
                                                       │
                                             /audio/*.wav ──→ <audio>
```

---

## 🔧 Konfiguracja zaawansowana

### Próg wykrywania zmian (`backend/diff_utils.py`)

```python
_CHANGE_THRESHOLD = 0.70   # zmień na niższy by być mniej wrażliwym
_MIN_TEXT_LENGTH  = 3      # minimalna długość tekstu do TTS
```

### Interwał OCR (`frontend/app.js`)

```javascript
const OCR_INTERVAL_MS = 1000;  // ms – domyślnie 1s
```

### Prędkość przyspieszania (`frontend/app.js`)

```javascript
const SPEEDUP_RATE   = 3.0;   // prędkość gdy nowe audio czeka
const SPEEDUP_THRESH = 2.0;   // próg (s) – poniżej nie przyspiesza
```

---

## 🗂️ Struktura projektu

```
subvoice/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── voices/          ← pliki głosu referencyjnego (MP3/WAV)
├── audio_cache/     ← wygenerowane pliki WAV (auto-tworzone)
├── backend/
│   ├── main.py      ← FastAPI + WebSocket
│   ├── ocr_worker.py← PaddleOCR + preprocessing
│   ├── tts_worker.py← OmniVoice TTS
│   └── diff_utils.py← wykrywanie zmian tekstu
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```
