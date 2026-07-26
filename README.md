# 🦀 Rust Reader - Asystent OCR i Syntezy Mowy (TTS) w Czasie Rzeczywistym

<div align="center">
**Rust Reader** to ultra-szybka, zoptymalizowana aplikacja stacjonarna stworzona w architekturze **Tauri v2 + Rust** oraz **TypeScript**. Służy do automatycznego przechwytywania obszarów ekranu, rozpoznawania tekstu (OCR) w czasie rzeczywistym z okien gier, filmów lub aplikacji, oraz czytania napisów głosowo za pomocą zaawansowanej syntezy mowy (TTS).
</div>

---

## 📸 Omówienie Projektu

Rust Reader rozwiązuje problem czytania obcojęzycznych lub trudnodostępnych napisów w grach i wideo bez wsparcia dla lektora. Dzięki połączeniu modeli sztucznej inteligencji z głębokimi optymalizacjami na poziomie języka Rust, aplikacja zapewnia natychmiastowe rozpoznawanie i odczyt z minimalnym zużyciem pamięci i procesora.

```
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  xcap Screen Capture    │ ──> │  Differential Frame    │ ──> │  Rust Preprocessing    │
│  (Monitor / Okno)       │     │  Hashing (AvgHash 256) │     │  (SIMD, Contrast, Crop)│
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                           │
┌─────────────────────────┐     ┌────────────────────────┐                 ▼
│  Rodio Audio Output     │ <── │  Sherpa-ONNX TTS Engine│ <── ┌────────────────────────┐
│  (Nieblokujący Głośnik) │     │  (Piper / Supertonic)  │     │  PP-OCRv6 ONNX Engine  │
└─────────────────────────┘     └────────────────────────┘     │  + SymSpell AutoCorrect│
                                                               └────────────────────────┘
```

---

## ⚡ Kluczowe Funkcje Projektu

### 🔍 1. Hybrydowy Silnik OCR (PP-OCRv6)
- **Model Detekcji**: `PP-OCRv6 Tiny` (`pp-ocrv6_tiny_det.onnx`) – błyskawiczne lokalizowanie obszarów tekstowych w wyciętym kadrze.
- **Model Rozpoznawania**: `PP-OCRv6 Small` (`pp-ocrv6_small_rec.onnx`) – precyzyjny odczyt znaków z pełnym słownikiem 18 708 symboli (`ppocrv6_dict.txt`).
- **Akceleracja Hardware**: Natywne wsparcie dla **NVIDIA CUDA** oraz **TensorRT** poprzez ONNX Runtime, z automatycznym powrotem (fallback) do przetwarzania CPU.
- **Automatyczne Pobieranie**: Wymagane modele ONNX pobierają się automatycznie przy pierwszym uruchomieniu.

### 🪄 2. Inteligentny System Dwu-Progowej Auto-Korekty
- **Próg Korekty Słownikowej (domyślnie 95%)**: Słowa rozpoznane z pewnością poniżej progu są poddawane korekcie na bazie algorytmu **SymSpell** i słownika 50 000 polskich słów (`pl_50k.txt`).
- **Próg Odrzucania Szumów (domyślnie 40%)**: Słowa z pewnością niższą niż 40% traktowane są jako artefakty graficzne/szum i są bezpowrotnie odrzucane z tekstu oraz syntezy głosowej.

### 📊 3. Wizualizacja Pewności Per-Słowo
- **Podgląd Na Żywo**: Każde rozpoznane słowo posiada dynamiczny wskaźnik kolorystyczny:
  - 🟢 **Pewność $\ge$ 95%** – odczyt bezsprzeczny.
  - 🟡 **Pewność < 95%** – odczyt niepewny, zakwalifikowany do korekty.
  - 🔴 **Słowo po korekcie SymSpell** – automatycznie poprawiony błąd.
- **Tabela Historii**: Loguje wcześniejsze odczyty ze szczegółowym podglądem surowych i poprawionych słów (np. `~pelen (87%)~ pełen`).

### 🗣️ 4. Synteza Mowy (TTS) w Czasie Rzeczywistym
- **Silnik Sherpa-ONNX**: Bezpośrednia integracja z modelami **Piper VITS** (Jarvis, Żeński, Męski) oraz **Supertonic 3** (wielojęzycznym lektorem wysokiej jakości).
- **Menedżer Głosów**: Interaktywny menedżer w UI pobierający wybrane głosy z oficjalnych repozytoriów z podglądem postępu pobierania.
- **Dynamiczna Adaptacja Tempa**: Lektor automatycznie przyspiesza (nawet do 3.5x), kiedy w kolejce nakromadzą się nowe napisy, zapobiegając opóźnieniom.

### 🚀 5. Ultra-Niskie Zużycie Zasobów (70-90% Oszczędności)
- **Differential Frame Analysis (Frame Hashing)**: 256-bitowy *Average Hash* wylicza odległość Hamminga. Jeśli obraz w obszarze napisów się nie zmienił, cały procesing OCR jest pomijany.
- **Auto-Pauza Podglądu**: Kodowanie miniatur JPEG automatycznie wyłącza się, gdy okno aplikacji straci fokus (`blur`), drastycznie oszczędzając zasoby podczas grania.
- **Filtracja Obrazu w Ruste**: Bezpośrednie przekształcenia pikseli (kontrast, binarizacja, wyostrzanie, marginesy) wykonują się w czasach **< 1 ms**.
- **Ochrona Pamięci DOM**: Tabela historii automatycznie ogranicza liczbę węzłów w widoku do maksymalnie 100 wpisów.

---

## 🛠️ Architektura i Technologie

| Komponent | Technologia / Crate | Opis |
| :--- | :--- | :--- |
| **Framework Główny** | `Tauri v2` (Rust) | Bezpieczna, ultralekka powłoka aplikacji desktopowej |
| **Interfejs Użytkownika** | `TypeScript` + `Vite` | Dynamiczne UI bez obciążających frameworków |
| **Przechwytywanie Ekranu** | `xcap` | Natywne przechwytywanie okien i monitorów z dowolnym FPS |
| **Silnik OCR** | `oar-ocr` + `onnxruntime` | Wykrywanie i rozpoznawanie PP-OCRv6 (CUDA/TensorRT/CPU) |
| **Korekta Pisowni** | `symspell` | Błyskawiczna korekta błędów OCR w oparciu o odległość Damerau-Levenshteina |
| **Synteza Mowy (TTS)** | `sherpa-onnx` | Natywna obsługa modeli VITS / Piper oraz Supertonic 3 |
| **Odtwarzacz Audio** | `rodio` | Nieblokujący bufor strumieniowania audio bez pętli busy-wait |

---

## 📁 Struktura Projektu

```text
rust-reader/
├── src-tauri/             # Główny kod aplikacji w języku Rust
│   ├── src/
│   │   ├── main.rs            # Punkt wejścia Tauri i komendy IPC
│   │   ├── app_state.rs       # Główna pętla przechwytywania i stan aplikacji
│   │   ├── ocr_engine.rs      # Wrapper oar-ocr oraz logika SymSpell
│   │   ├── tts_engine.rs      # Wrapper sherpa-onnx i menedżer pobierania głosów
│   │   ├── audio_player.rs    # Odtwarzanie dźwięku przez rodio
│   │   ├── window_capture.rs  # Przechwytywanie obrazu przez xcap
│   │   ├── diff_filter.rs     # Frame Hashing i analiza różnicowa kadrów
│   │   └── box_filter.rs      # Filtrowanie i sanitaryzacja tekstu
│   ├── Cargo.toml             # Zależności Rust
│   └── tauri.conf.json        # Konfiguracja okien i budowania Tauri
│   └── pl_50k.txt            # Plik z 50k najpopulrniejszych słów w polskim do auto korekty
├── src/                   # Interfejs użytkownika (Frontend)
│   ├── index.html             # Główna struktura HTML
│   ├── app.ts                 # Logika UI, podgląd live i obsługa zdarzeń IPC
│   └── style.css              # Stylowanie w czystym CSS
├── package.json           # Skrypty Node / Bun i zależności frontendowe
└── README.md              # Dokumentacja projektu
```

---

## 🛠️ Wymagania i Instalacja

### Wymagania Wstępne:
1. **Rust Toolchain**: `rustc`, `cargo` (rekomendowany profil `stable`). [Pobierz Rust](https://www.rust-lang.org/tools/install)
2. **Bun** (zalecany) lub **Node.js**: [Pobierz Bun](https://bun.sh/)
3. **C++ Build Tools & CMake** (dla systemu Windows): Wymagane do kompilacji nagłówków ONNX Runtime oraz bibliotek audio.
4. **Sterowniki GPU** (opcjonalnie): Najnowsze sterowniki NVIDIA z obsługą Direct3D / Vulkan / CUDA dla pełnej akceleracji GPU.

### Krok 1: Klonowanie Repozytorium
```bash
git clone https://github.com/BursztyneK98/Rust-Reader-BK98.git
cd rust-reader
```

### Krok 2: Instalacja Zależności Frontendowych
```bash
bun install
```

### Krok 3: Uruchomienie w Trybie Deweloperskim
```bash
bun tauri dev
```
*Uwaga: Przy pierwszym uruchomieniu silnik pobierze wymagane modele OCR oraz przygotuje słownik `pl_50k.txt` w katalogu aplikacji.*

### Krok 4: Budowanie Wersji Produkcyjnej (Release)
```bash
bun tauri build
```
Gotowy plik instalacyjny `.msi` lub `.exe` znajdzie się w katalogu `src-tauri/target/release/bundle/`.

---

## 🎛️ Przewodnik po Panelu Sterowania

| Opcja w UI | Domyślna wartość | Opis |
| :--- | :--- | :--- |
| **Źródło Obrazu** | Pierwszy monitor | Wybór monitora lub konkretnego okna gry/aplikacji |
| **Interwał Skanowania** | `300 ms` | Częstotliwość cyklu OCR (regulacja 100 ms – 1000 ms) |
| **Próg Auto-Korekty** | `95%` | Wartość pewności, poniżej której aktywuje się korekta SymSpell |
| **Próg Odrzucania Szumów**| `40%` | Wartość pewności, poniżej której słowa są pomijane |
| **Obszar Kadrowania (Crop)**| Dolne `20%` | Marginesy wycinające obszar napisów na ekranie |
| **Silnik i Głos TTS** | `Polski Jarvis` | Wybór aktywnego lektora oraz pobieranie nowych głosów |
| **Prędkość i Głośność TTS**| `1.0x` / `100%` | Bazowe ustawienia odtwarzania lektora |
| **Podgląd Live** | `Włączony` | Przełącznik renderingu podglądu w celu oszczędzania zasobów |

---

## 💡 Rozwiązywanie Problemów (FAQ)

<details>
<summary><b>1. Aplikacja zgłasza błąd związany z CUDA / TensorRT</b></summary>

Rust Reader automatycznie próbuje wykryć i zainicjalizować provider CUDA/TensorRT. Jeśli na Twoim komputerze nie ma zainstalowanych bibliotek CUDA Toolkit lub odpowiedniej karty graficznej, ONNX Runtime bezpiecznie przełącza się na CPU execution provider.
</details>

<details>
<summary><b>2. Brak głosu lektora po wybraniu nowego głosu</b></summary>

Upewnij się, że głos został pobrany. W sekcji TTS kliknij przycisk pobierania obok wybranego głosu i poczekaj, aż pasek postępu osiągnie 100%. Pobrany plik zostanie rozpakowany w katalogu danych aplikacji.
</details>

<details>
<summary><b>3. Napisy są odczytywane wielokrotnie</b></summary>

Zwiększ próg odległości Hamminga lub wyreguluj obszar kadrowania (Crop), aby wyeliminować migające elementy UI gry (np. ikony czy paski zdrowia), które mogą powodować wyzwalanie cyklu OCR przy braku zmian w napisach.
</details>

---

## 📄 Licencja

Projekt jest udostępniany na warunkach licencji **MIT**. Szczegóły znajdują się w pliku [LICENSE](LICENSE).

Autor: **[BursztyneK98](https://github.com/BursztyneK98)**
