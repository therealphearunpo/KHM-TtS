# KHM-TtS: Khmer Neural Text-to-Speech Studio & Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/therealphearunpo/KHM-TtS/blob/main/train_colab.ipynb)

A lightweight, non-autoregressive Khmer Text-to-Speech (TTS) pipeline designed for fast local GPU training and low-latency client-side Web deployment (ONNX Runtime Web / WebGPU).

---

## Project Structure

```
KHM-TtS/
├── data/                               # Data preprocessing & alignment scripts
│   ├── khmer_normalizer.py             # Unicode NFC, number expansion, Lek To
│   ├── khmer_tokenizer.py              # Tokenizer & character vocab mapping
│   ├── build_lexicon.py                # Phoneme / grapheme lexicon builder
│   ├── prepare_dataset.py              # 22.05kHz audio resampling & mel extraction
│   └── align_whisperx.py               # Token & phoneme duration aligner
├── model/                              # PyTorch neural network architectures
│   ├── acoustic_model.py               # FastSpeechLite (Encoder, DurationPredictor, Decoder)
│   └── vocoder.py                      # HiFiGAN-tiny & Griffin-Lim vocoders
├── models/                             # Trained checkpoints & weights
│   └── acoustic_predict_model.pt       # Trained acoustic model weights
├── web/                                # Lightweight Web Speech Studio UI
│   ├── index.html                      # Interactive frontend
│   └── static/                         # CSS / JS assets
├── Khmer_TTS_Pipeline.ipynb            # 📓 End-to-end pipeline (Data -> Train -> Eval -> Infer)
├── Khmer_TTS_Demo.ipynb                # 📓 Clean interactive audio synthesis demo
├── history.csv                         # 📊 Epoch loss & validation metrics log
├── requirements.txt                    # 📦 Project dependencies
├── train_acoustic.py                   # 🏋️ Acoustic model training CLI
├── train_vocoder.py                    # 🏋️ Vocoder training CLI
├── infer.py                            # 🎙️ Direct inference CLI
├── evaluate.py                         # 📈 Model benchmarking & MCD evaluation
├── export_onnx.py                      # 📦 ONNX model export tool
├── server.py                           # 🚀 Web Studio & Synthesis API server
└── README.md                           # 📖 Project documentation
```

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
