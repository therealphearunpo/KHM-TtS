# KHM-TtS — Single-Speaker, Browser-Deployed (ONNX Runtime Web)

A small, non-autoregressive text-to-speech pipeline designed to be:
- **Trainable on a single local GPU** in hours, not days
- **Fast at inference** — no autoregressive decoding loop, so no accumulating latency per character
- **Exportable to ONNX** and runnable client-side with `onnxruntime-web` (WASM or WebGPU backend)

## Architecture

Two stages, trained and exported separately (standard practice — keeps each model small and each
training run fast/debuggable):

1. **Acoustic model** (`model/acoustic_model.py`) — "FastSpeechLite"
   Text/phonemes → mel-spectrogram. Non-autoregressive: a transformer encoder, a duration
   predictor + length regulator (expands hidden states to frame-rate instead of decoding
   step-by-step), and a transformer decoder that outputs mel frames all at once. This is why
   it's low-latency: inference is one forward pass, not N sequential steps.

2. **Vocoder** (`model/vocoder.py`) — "HiFiGAN-tiny"
   Mel-spectrogram → waveform. A shrunk HiFi-GAN generator (fewer channels/layers than the
   paper's default). Trained adversarially against a lightweight discriminator.

Why not one end-to-end model (e.g. VITS)? VITS-style models are higher quality but involve a
flow-based decoder + GAN + posterior encoder trained jointly — much more failure-prone to train
from scratch on a single small dataset, and harder to get a clean ONNX export from (the flow and
stochastic duration predictor use ops that export awkwardly). The two-stage design here trades a
little quality ceiling for something you can realistically train, debug, and ship yourself.

## Dataset format expected

LJSpeech-style layout (this is what most TTS tooling, including the aligner below, expects):

```
dataset/
  wavs/
    0001.wav
    0002.wav
    ...
  metadata.csv      # id|transcript text, one per line, pipe-separated
```

Requirements: mono, 22050 Hz (resample if needed — see `data/prepare_dataset.py`), reasonably
clean/dry (you said studio-clean, good — reverb and background noise hurt small models a lot
more than they hurt big ones).

## Pipeline

```
1. data/prepare_dataset.py      # resample audio, compute mel-spectrograms, run G2P on text
2. (external) Montreal Forced Aligner  # get phoneme-level durations
3. data/align_durations.py      # convert MFA TextGrids -> per-phoneme frame durations
4. train_acoustic.py            # train FastSpeechLite (text -> mel)
5. train_vocoder.py             # train HiFiGAN-tiny (mel -> waveform), can run in parallel with 4
6. export_onnx.py                # export both models to ONNX
7. web/index.html + web/app.js  # run inference in-browser via onnxruntime-web
```

## Why Montreal Forced Aligner (MFA) instead of learning alignment end-to-end

Models like Glow-TTS/VITS learn phoneme durations internally via monotonic alignment search,
which avoids needing an external tool — but it's another moving part to get right during
training (alignment collapse is a common failure mode on small datasets). MFA is a mature,
well-documented, CPU-only tool that gives you durations up front, so your acoustic model training
is a plain supervised regression problem. Install:

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa align dataset/ english_us_arpa english_us_arpa dataset/aligned/
```

## Setup & Running the Web App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the local studio server (streams dataset audio & serves web app)
python server.py
```
Open **http://localhost:8000** in your browser to view the **Mini TTS Speech Studio & Dataset Explorer**.

## Connected Dataset: `km_kh_male`
- **Total Audio Utterances**: 2,906 WAV files (48 kHz studio-recorded speech)
- **Vocabulary**: 73 Khmer graphemes/tokens (`web/models/vocab.json`)
- **Lexicon**: 4,347 words/syllables mapped to character tokens (`web/models/lexicon.json`)
- **Features in Web App**:
  - Live dataset explorer with search, audio playback, and instant "Use Text" into the synthesizer.
  - Real-time token / grapheme breakdown inspector.
  - Browser-side ONNX Runtime Web execution (WebGPU / WASM).

## Pipeline

```
1. python server.py             # Start Web App & Dataset Explorer at http://localhost:8000
2. data/prepare_dataset.py      # Resample audio, compute mel-spectrograms, tokenize text
3. (external) Montreal Forced Aligner  # Get phoneme/token-level durations
4. data/align_durations.py      # Convert MFA TextGrids -> per-token frame durations
5. train_acoustic.py            # Train FastSpeechLite (tokens -> mel)
6. train_vocoder.py             # Train HiFiGAN-tiny (mel -> waveform)
7. export_onnx.py               # Export both models to web/models/ for browser inference
```
