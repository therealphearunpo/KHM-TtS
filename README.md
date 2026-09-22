# KHM-TTS: Khmer Single-Speaker Text-to-Speech Mini-Project

A lightweight, non-autoregressive Khmer Text-to-Speech (TTS) pipeline designed for low-latency inference, ONNX export, and local deployment.

---

## System Architecture

The pipeline uses a decoupled two-stage architecture:
1. **Acoustic Model (FastSpeechLite)**: Maps normalized Khmer text character tokens to 80-channel log-mel spectrograms in a single non-autoregressive forward pass.
2. **Vocoder (HiFiGAN-tiny / Griffin-Lim)**: Converts predicted log-mel spectrograms into 22,050 Hz time-domain audio waveforms.

```
Khmer Text Input
       │
       ▼
[Khmer Normalizer] (NFC, Lek To expansion, digits to words)
       │
       ▼
[Khmer Tokenizer]  (Character-level vocab: 77 tokens)
       │
       ▼
[FastSpeechLite]   (Encoder 4x FFT ──► Duration Predictor ──► Length Regulator ──► Decoder 4x FFT)
       │
       ▼ Mel-spectrogram (80 bins)
[HiFiGAN-tiny / Griffin-Lim Fallback]
       │
       ▼
22,050 Hz Audio Waveform (.wav)
```

### Component Specifications

| Component | Architecture / Details | Parameters / Size |
| :--- | :--- | :--- |
| **Normalizer** | Unicode NFC, Lek To (`ៗ`) word duplication, Khmer/Arabic number expansion, zero-width space removal | Pure Python (`data/khmer_normalizer.py`) |
| **Tokenizer** | Character-level grapheme vocabulary (77 tokens including `<pad>`, `<unk>`, `<space>`) | `web/models/vocab.json` |
| **Acoustic Model** | **FastSpeechLite**: 4-layer FFT Encoder, Duration Predictor (conv1d), Length Regulator, 4-layer FFT Decoder (`d_model=192`, `d_ff=768`, `n_heads=2`, `n_mels=80`) | **23.58M params**<br>PyTorch: ~94.4 MB<br>ONNX: ~92.8 MB |
| **Vocoder** | **HiFiGAN-tiny**: Multi-receptive field Generator (upsample rates: `[8, 8, 2, 2]`, hop: 256) + Griffin-Lim fallback (32 iterations) | **2.19M params**<br>PyTorch: ~17.5 MB<br>ONNX: ~8.75 MB |
| **Audio Format** | Single-channel mono, **22,050 Hz**, 80-bin mel-spectrogram (`n_fft=1024`, `hop=256`, `win=1024`, `f_min=0`, `f_max=8000`) | Standard LJSpeech / LibriTTS mel spec |

---

## Dataset: `km_kh_male`

- **Source**: Studio-recorded Khmer male speaker dataset.
- **Utterances**: 2,906 audio files.
- **Total Duration**: ~3.97 hours (average utterance duration: ~4.9 seconds).
- **Original Format**: 48,000 Hz 16-bit PCM WAV.
- **Processed Format**: 22,050 Hz 16-bit PCM WAV (`km_kh_male/processed/wavs_22k/`).
- **Features Extracted**: 80-channel log-mel spectrograms (`.npy`), tokenized IDs (`.npy`), duration alignments (`.npy`).

---

## Quick Start & Web Studio

### 1. Environment Setup

```bash
# Clone and navigate to repository
cd tts-mini

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Local Web Studio & API Server

```bash
python server.py --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

> [!NOTE]
> **Architecture Clarification**: The web interface provides an interactive Dataset Explorer and Speech Studio. When you click **Synthesize**, the browser sends a `POST /api/synthesize` request to the local Python HTTP server, which runs inference via PyTorch / ONNX models and returns the audio waveform. It does not run standalone WASM inference inside the browser engine.

---

## End-to-End Pipeline Execution

If building or retraining the models from scratch:

### Step 1: Preprocess Dataset
Resamples raw audio to 22.05 kHz, builds the vocabulary, tokenizes transcripts, and precomputes log-mel spectrograms:
```bash
python data/prepare_dataset.py --data_dir km_kh_male
```

### Step 2: Extract Token Durations
Extracts token-level duration alignments using energy and spectral onset heuristics (self-contained, no external aligner required):
```bash
python data/extract_durations.py --data_dir km_kh_male
```

### Step 3: Train Acoustic Model (FastSpeechLite)
Trains the non-autoregressive transformer to predict mel spectrograms and durations:
```bash
python train_acoustic.py --data_dir km_kh_male --epochs 50 --batch_size 16 --lr 1e-3
```
Checkpoints are saved to `checkpoints/acoustic/` (`best_acoustic.pt`, `latest_acoustic.pt`).

### Step 4: Train Vocoder (HiFiGAN-tiny)
Trains the neural vocoder generator and multi-period discriminator:
```bash
python train_vocoder.py --data_dir km_kh_male --epochs 50 --batch_size 16 --lr 2e-4
```
Checkpoints are saved to `checkpoints/vocoder/` (`best_vocoder.pt`, `latest_vocoder.pt`).

### Step 5: Export to ONNX
Exports both PyTorch models to optimized ONNX graphs with dynamic axes and verifies numerical consistency:
```bash
python export_onnx.py --acoustic_ckpt checkpoints/acoustic/best_acoustic.pt --vocoder_ckpt checkpoints/vocoder/best_vocoder.pt --out_dir web/models
```

### Step 6: Run Command-Line Inference
Synthesize Khmer text directly to audio:
```bash
python infer.py --text "សួស្តីកម្ពុជា" --vocoder griffin_lim --out output.wav
```

---

## Performance Benchmarks

Measured on standard CPU (Intel/AMD x86_64, PyTorch CPU mode):

### 1. Real-Time Factor (RTF)

$$\text{RTF} = \frac{\text{Synthesis Latency (s)}}{\text{Generated Audio Duration (s)}}$$

| Input Type | Text Sample | Audio Duration | CPU Latency | Real-Time Factor (RTF) |
| :--- | :--- | :--- | :--- | :--- |
| **Short** | `សួស្តី` (5 chars) | 0.33 s | 0.043 s | **0.133x** (7.5x real-time) |
| **Medium** | `សួស្តីកម្ពុជា ស្អាតណាស់` (23 chars) | 1.72 s | 0.056 s | **0.033x** (30x real-time) |
| **Long** | Complete sentence (129 chars) | 8.96 s | 0.167 s | **0.019x** (52x real-time) |

Pipeline initialization time: **~0.185 seconds**.

### 2. ONNX Numerical Consistency

| Model | Max Absolute Error ($\|y_{\text{ONNX}} - y_{\text{PyTorch}}\|_{\infty}$) | Status |
| :--- | :--- | :--- |
| **FastSpeechLite (Acoustic)** | $1.67 \times 10^{-6}$ | Verified Match (Tolerance $10^{-4}$) |
| **HiFiGAN-tiny (Vocoder)** | $7.45 \times 10^{-8}$ | Verified Match (Tolerance $10^{-4}$) |

Dynamic sequence lengths: Verified with dynamic input lengths from 5 to 500+ tokens without truncation.

---

## Automated Test Suite

A comprehensive test suite containing 55 unit and integration tests is included in `tests/test_pipeline.py`:

```bash
python -m unittest discover tests -v
```

### Test Coverage Areas
- **Text Normalization (10 tests)**: NFC normalization, Lek To (`ៗ`) duplication, Khmer and Arabic numeral expansion, zero-width space removal, ligature preservation.
- **Tokenizer (8 tests)**: Encoding, decoding roundtrip, vocabulary coverage, unknown token handling, padding IDs.
- **Dataset Loading (7 tests)**: Tensor shapes, duration-phoneme alignment, duration sum = mel length consistency.
- **Audio Loading (4 tests)**: Sampling rate, channel layout, amplitude normalization $[-1, 1]$, NaN/Inf detection.
- **Model Architectures (6 tests)**: Acoustic forward pass in training and evaluation modes, vocoder generator and discriminator shapes.
- **End-to-End Inference (6 tests)**: Synthesis pipeline, output amplitude bounds, punctuation handling.
- **ONNX Consistency (3 tests)**: Numerical equivalence between PyTorch and ONNX Runtime, dynamic length verification.
- **Server API (5 tests)**: `/api/stats`, `/api/samples`, `/api/synthesize`, error handling (400 on empty text, 404 on missing audio).
- **Edge Cases & Error Handling (4 tests)**: Empty inputs, whitespace strings, unsupported emoji characters, very long inputs (>2,000 frames).

**Result**: 55 passed, 0 failed.

---

## Known Limitations & Production Recommendations

1. **Training Epochs**: The current acoustic checkpoint was trained for 5 epochs. Mel spectrogram envelopes are learned, but higher-frequency formants and phoneme clarity require 50–100 epochs of training.
2. **Vocoder State**: The neural vocoder checkpoint is at epoch 0. Use `--vocoder griffin_lim` for reliable speech reconstruction until HiFiGAN reaches epoch 30+.
3. **Phonetic Representation**: The tokenizer operates on Khmer graphemes rather than phonemes. For complex Khmer consonant clusters and unwritten vowels, integrating a full Khmer G2P (Grapheme-to-Phoneme) rule engine or neural aligner will improve pronunciation accuracy.
