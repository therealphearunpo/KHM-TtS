# KHM-TtS: Khmer Neural Text-to-Speech Studio & Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/therealphearunpo/KHM-TtS/blob/main/train_colab.ipynb)

A lightweight, non-autoregressive Khmer Text-to-Speech (TTS) pipeline designed for fast local GPU training and low-latency client-side Web deployment (ONNX Runtime Web / WebGPU).

---

## 📁 Project Directory Structure

```text
KHM-TtS/
├── checkpoints/                  # 💾 Model Checkpoint Storage
│   ├── acoustic/                 # FastSpeechLite checkpoints (best_acoustic.pt)
│   └── vocoder/                  # HiFiGAN-tiny checkpoints (best_vocoder.pt)
│
├── data/                         # 🛠️ Data Preprocessing & Alignment Tools
│   ├── align_durations.py        # External MFA TextGrid duration converter
│   ├── build_lexicon.py          # Khmer word-to-token lexicon builder
│   ├── extract_durations.py      # Energy-guided acoustic duration extractor
│   ├── khmer_normalizer.py       # Khmer text normalizer (Lek To ៗ, numerals, ZWSP)
│   ├── khmer_tokenizer.py        # Khmer grapheme/token encoder & decoder
│   └── prepare_dataset.py        # 22.05kHz resampling, silence trim, peak norm, mels
│
├── km_kh_male/                   # 🎙️ Raw Dataset (OpenSLR 42)
│   ├── line_index.tsv            # Raw index TSV
│   ├── metadata.csv              # Pipe-separated metadata (id|transcript)
│   └── wavs/                     # 48 kHz studio audio WAVs (2,906 files)
│
├── model/                        # 🧠 Neural Network Architectures
│   ├── acoustic_model.py         # FastSpeechLite (Transformer + ONNX LengthRegulator)
│   └── vocoder.py                # HiFiGAN-tiny Generator & Multi-Scale Discriminator
│
├── processed/                    # 📦 Processed Training Data (22.05 kHz)
│   ├── durations/                # Extracted frame duration arrays (.npy)
│   ├── mels/                     # Log mel-spectrogram arrays (.npy)
│   ├── phonemes/                 # Token integer ID sequences (.npy)
│   ├── wavs_22k/                 # Resampled & normalized WAV files
│   ├── vocab.json                # 77-token vocabulary mapping
│   ├── train_manifest.txt        # 90% training split (2,614 utterances)
│   ├── val_manifest.txt          # 5% validation split (146 utterances)
│   └── test_manifest.txt         # 5% held-out test split (146 utterances)
│
├── web/                          # 🌐 Web Studio & Dataset Explorer
│   ├── index.html                # Modern glassmorphism Studio UI
│   ├── style.css                 # Dark theme, dynamic glows & micro-animations
│   ├── app.js                    # Live tokenization, playback, & synthesis controller
│   ├── data/
│   │   └── samples.json          # Searchable dataset utterances (2,906 entries)
│   └── models/                   # ONNX artifacts & runtime dictionaries
│       ├── vocab.json            # Model vocabulary
│       ├── lexicon.json          # 4,347 words dictionary
│       ├── acoustic.onnx         # Exported FastSpeechLite (Stage 1)
│       └── vocoder.onnx          # Exported HiFiGAN-tiny (Stage 2)
│
├── evaluate.py                   # 📊 Evaluation script (Mel L1, MCD dB, RTF latency)
├── export_onnx.py                # 🚀 Exports models to web/models/
├── infer.py                      # 🔊 End-to-end synthesis CLI & Python API
├── server.py                     # ⚡ Local Studio Web Server & Audio Streaming (Port 8000)
├── train_acoustic.py             # 🎯 Train FastSpeechLite (Acoustic model)
└── train_vocoder.py              # 🎯 Train HiFiGAN-tiny (Neural vocoder)
```

---

## ⚡ Quickstart: Training & Running Web Studio

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Raw Audio & Generate Mels
Resamples audio to 22,050 Hz, trims silence safely (45 dB + 50ms buffer), normalizes peak amplitude, builds vocabulary, and creates splits:
```bash
python data/prepare_dataset.py --data_dir km_kh_male --out_dir processed
```

### 3. Extract Acoustic Energy-Guided Durations
Analyzes actual speech energy and silence regions from mel-spectrograms to align tokens with real acoustic frames:
```bash
python data/extract_durations.py --data_dir km_kh_male --proc_dir processed
```

### 4. Train Models
You can run these sequentially or in parallel on separate GPUs:

```bash
# Train FastSpeechLite (Text -> Mel-spectrogram)
python train_acoustic.py --data_dir km_kh_male --epochs 100 --batch_size 16

# Train HiFiGAN-tiny (Mel-spectrogram -> Audio Waveform)
python train_vocoder.py --data_dir km_kh_male --epochs 200 --batch_size 16
```

### 5. Evaluate on Held-out Test Set
Computes Mel L1 Loss, Mel-Cepstral Distortion (MCD in dB), Real-Time Factor (RTF), and saves side-by-side ground truth vs synthesized audio:
```bash
python evaluate.py --data_dir km_kh_male
```

### 6. Export to ONNX for Web Deployment
```bash
python export_onnx.py --out_dir web/models
```

### 7. Run the Web Studio
```bash
python server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to access:
- **Speech Synthesizer Studio**: Live text-to-speech with speed control and instant audio waveform player.
- **Real-Time Token Inspector**: Interactive breakdown of Khmer graphemes, token IDs, and Lek To (`ៗ`) expansions.
- **Dataset Explorer**: Search and listen to all 2,906 studio recordings from `km_kh_male` with instant "Use Text" copying.
