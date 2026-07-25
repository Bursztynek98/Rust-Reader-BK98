# 🦀 Rust Reader - Asystent OCR i Syntezy Mowy (TTS) w Czasie Rzeczywistym

**Rust Reader** to wysoce zoptymalizowana aplikacja stacjonarna stworzona w technologii **Tauri v2 + Rust** oraz **TypeScript**, przeznaczona do automatycznego przechwytywania, rozpoznawania (OCR) oraz czytania na głos (TTS) napisów z dowolnego okna gry, filmu lub ekranu monitora w czasie rzeczywistym.

---

## ⚡ Główne Cechy Projektu

* **Hybrydowy Silnik PP-OCRv6**:
  * **Detekcja**: `PP-OCRv6 Tiny` (`pp-ocrv6_tiny_det.onnx`) – błyskawiczne wykrywanie ramek tekstu.
  * **Rozpoznawanie**: `PP-OCRv6 Small` (`pp-ocrv6_small_rec.onnx`) – precyzyjny odczyt z pełnym słownikiem 18 708 znaków (`ppocrv6_dict.txt`).
  * **Akceleracja GPU**: Wsparcia dla TensorRT oraz CUDA poprzez ONNXRuntime.

* **Inteligentny System Dwu-Progowej Auto-Korekty**:
  * **Próg Auto-Korekty (domyślnie 95%)**: Słowa z pewnością odczytu poniżej 95% są automatycznie poprawiane przez słownik `SymSpell` (`pl_50k.txt`).
  * **Próg Odrzucania Szumów (domyślnie 40%)**: Słowa z pewnością poniżej 40% są traktowane jako szumy/artefakty OCR i całkowicie odrzucane (wywalane z tekstu i lektora).

* **Wizualizacja Pewności Per-Słowo**:
  * **Podgląd Główny**: Każde rozpoznane słowo posiada pod spodem mały wskaźnik procentowy pewności (🟢 $\ge$ 95%, 🟡 < 95%, 🔴 po korekcie).
  * **Tabela Historii**: Pokazuje historię odczytanych napisów. Dla słów skorygowanych prezentowane jest przekreślone surowe słowo wraz z jego dokładną pewnością (np. `~pelen (87%)~ pełen`).

* **Synteza Mowy (TTS) w Czasie Rzeczywistym**:
  * Wsparcie dla silników **Piper VITS** (Jarvis, Żeński, Męski) oraz **Supertonic 3** (wielojęzyczny lektor z polską wymową).
  * **Dynamiczna Regulacja Tempa**: Automatyczne przyspieszanie lektora (do 3.5x), gdy w kolejce nakromadzą się nowe napisy.

* **Optymalizacja Wydajnościowa i Pamięciowa**:
  * **Analiza Różnicowa Kadrów (Frame Hashing)**: 256-bitowy Average Hash wylicza odległość Hamminga – jeśli obraz na ekranie nie uległ zmianie, procesing OCR jest pomijany, redukując zużycie CPU/GPU o 70-90%.
  * **Auto-Pauza Podglądu w Tle**: Przechwytywanie i kodowanie miniatur JPEG wyłącza się automatycznie po utracie fokusu okna (`blur`), redukując obciążenie systemowe.
  * **Przetwarzanie Obrazu w Ruste**: Bezpośrednie filtrowanie obrazu (kontrast, binaryzacja, wyostrzanie, marginesy) wykonuje się w czasach poniżej 1 ms.
  * **Czyszczenie DOM**: Tabela historii automatycznie ogranicza liczbę węzłów DOM do max 100 wpisów.

---

## 🛠️ Architektura i Technologie

| Komponent | Technologia | Opis |
| :--- | :--- | :--- |
| **Główny Framework** | Tauri v2 (Rust) | Lekka i bezpieczna powłoka aplikacji desktopowej |
| **Interfejs (UI)** | TypeScript + Vite + Vanilla CSS | Responsywny układ przystosowany do ekranów 720p / 1080p |
| **Przechwytywanie** | `xcap` (Rust) | Natywne, bezbiblioteczne przechwytywanie okien i monitorów |
| **OCR** | `oar-ocr` (ONNXRuntime) | PP-OCRv6 Tiny Det + Small Rec z akceleracją CUDA/TensorRT |
| **Korekta Słownikowa** | `SymSpell` | Szybka korekta błędów literowych na bazie słownika 50k polskich słów |
| **TTS (Lektor)** | `sherpa-onnx` | Natywne ładowanie modeli Piper oraz Supertonic VITS |
| **Odtwarzacz Audio** | `rodio` | Nieblokujące odtwarzanie audio bez pętli busy-wait |

---

## 🚀 Uruchomienie i Rozwój

### Wymagania wstępne:
1. **Rust** (cargo, rustc): [install.rust-lang.org](https://www.rust-lang.org/tools/install)
2. **Bun** lub **Node.js**: `bun` jest rekomendowanym menedżerem pakietów.
3. **Pakiety Tauri**: Sterowniki graficzne z obsługą Direct3D / Vulkan / CUDA (opcjonalnie dla akceleracji GPU).

### Instalacja zależności:
```bash
bun install
```

### Uruchomienie w trybie deweloperskim:
```bash
bun tauri dev
```

### Budowanie wersji produkcyjnej (Release):
```bash
bun tauri build
```

---

## 🎛️ Skrót Panelu Sterowania

1. **Źródło Obrazu**: Wybierz monitor lub konkretne okno aplikacji do skanowania.
2. **Interwał Skanowania**: Regulacja od 100 ms do 1000 ms (domyślnie 300 ms).
3. **Auto-Korekta (95%)**: Wartość progu, od którego włącza się korekta słownikowa.
4. **Odrzucanie Szumów (40%)**: Wartość progu, poniżej którego niepewne słowa są wywalane.
5. **Obszar Kadrowania (Crop)**: Precyzyjna regulacja pozycji i wysokości obszaru napisów (domyślnie dolne 20% ekranu).
6. **Głos TTS & Prędkość**: Wybór lektora, prędkości bazowej oraz głośności.
7. **Podgląd na Żywo**: Możliwość manualnego wyłączenia podglądu w celu zaoszczędzenia zasobów.

---

## 📄 Licencja

Projekt objęty licencją MIT. Stworzony przez **BursztyneK98**.
