"""
Preprocess an audio dataset (OpenSLR 42 / LJSpeech format):
  dataset/wavs/*.wav
  dataset/metadata.csv (id|transcript) OR dataset/line_index.tsv (id\\ttext)

Produces:
  dataset/processed/mels/<id>.npy        mel-spectrograms, shape (n_mels, T)
  dataset/processed/phonemes/<id>.npy    token/phoneme id sequence, shape (L,)
  dataset/processed/wavs_22k/<id>.wav    resampled 22.05kHz mono wavs
  dataset/processed/vocab.json           token/phoneme -> id mapping
  dataset/processed/train_manifest.txt   training split (90%)
  dataset/processed/val_manifest.txt     validation split (5%)
  dataset/processed/test_manifest.txt    testing split (5%)

Usage:
  python data/prepare_dataset.py --data_dir km_kh_male
"""
import argparse
import json
import os
import random
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from tqdm import tqdm

from data.khmer_tokenizer import KhmerTokenizer

SAMPLE_RATE = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
FMIN = 0.0
FMAX = 8000.0

# Pre-instantiate MelSpectrogram transform
_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    power=1.0,
)
_resamplers = {}


def get_resampler(orig_sr: int) -> T.Resample:
    if orig_sr not in _resamplers:
        _resamplers[orig_sr] = T.Resample(orig_sr, SAMPLE_RATE)
    return _resamplers[orig_sr]


def trim_silence(wav: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Trim silence from beginning and end based on energy threshold."""
    if len(wav) == 0:
        return wav
    energy = np.abs(wav)
    max_e = np.max(energy)
    if max_e == 0:
        return wav
    thresh = max_e * (10 ** (-top_db / 20))
    non_silent = np.where(energy > thresh)[0]
    if len(non_silent) == 0:
        return wav
    start, end = non_silent[0], non_silent[-1] + 1
    return wav[start:end]


def build_mel(wav_np: np.ndarray) -> np.ndarray:
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
    mel_t = _mel_transform(wav_t)
    mel_t = torch.log(torch.clamp(mel_t, min=1e-5)).squeeze(0)
    return mel_t.numpy().astype(np.float32)


def load_metadata(data_dir: str):
    """Load metadata from metadata.csv (id|text) or line_index.tsv (id\\ttext)."""
    meta_csv = os.path.join(data_dir, "metadata.csv")
    meta_tsv = os.path.join(data_dir, "line_index.tsv")

    items = []
    if os.path.exists(meta_csv):
        with open(meta_csv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    items.append((parts[0].strip(), parts[1].strip()))
    elif os.path.exists(meta_tsv):
        with open(meta_tsv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                utt_id = parts[0].strip()
                text = parts[-1].strip()
                items.append((utt_id, text))
    else:
        raise FileNotFoundError(f"Neither metadata.csv nor line_index.tsv found in {data_dir}")

    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to dataset directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for data splits")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    wav_dir = os.path.join(args.data_dir, "wavs")
    out_dir = os.path.join(args.data_dir, "processed")
    mel_dir = os.path.join(out_dir, "mels")
    phon_dir = os.path.join(out_dir, "phonemes")
    wav_out_dir = os.path.join(out_dir, "wavs_22k")
    for d in (mel_dir, phon_dir, wav_out_dir):
        os.makedirs(d, exist_ok=True)

    items = load_metadata(args.data_dir)
    print(f"Loaded {len(items)} items from metadata.")

    # Initialize Tokenizer and build vocabulary
    tokenizer = KhmerTokenizer()
    all_texts = [text for _, text in items]
    vocab = tokenizer.build_vocab_from_texts(all_texts)
    print(f"Khmer vocabulary size: {len(vocab)} tokens.")

    manifest = []
    for utt_id, text in tqdm(items, desc="preprocessing"):
        wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
        if not os.path.exists(wav_path):
            continue

        try:
            # Read audio and resample to 22050Hz
            wav_raw, orig_sr = sf.read(wav_path)
            if len(wav_raw.shape) > 1:
                wav_raw = wav_raw.mean(axis=1)  # convert to mono
            if orig_sr != SAMPLE_RATE:
                resampler = get_resampler(orig_sr)
                wav_t = resampler(torch.from_numpy(wav_raw).float().unsqueeze(0)).squeeze(0)
                wav = wav_t.numpy()
            else:
                wav = wav_raw.astype(np.float32)

            # Trim leading/trailing silence
            wav = trim_silence(wav, top_db=30.0)
            if len(wav) < HOP_LENGTH:
                continue

            sf.write(os.path.join(wav_out_dir, f"{utt_id}.wav"), wav, SAMPLE_RATE)

            # Mel Spectrogram
            mel = build_mel(wav)
            np.save(os.path.join(mel_dir, f"{utt_id}.npy"), mel)

            # Tokenize & encode token IDs
            token_ids = np.array(tokenizer.text_to_ids(text), dtype=np.int64)
            np.save(os.path.join(phon_dir, f"{utt_id}.npy"), token_ids)

            # Label file for forced aligner
            with open(os.path.join(wav_out_dir, f"{utt_id}.lab"), "w", encoding="utf-8") as lf:
                lf.write(tokenizer.normalizer.normalize(text))

            manifest.append(utt_id)
        except Exception as e:
            print(f"Error processing {utt_id}: {e}")

    # Save vocabulary
    tokenizer.save_vocab(os.path.join(out_dir, "vocab.json"))
    # Also mirror to web/models/vocab.json
    tokenizer.save_vocab("web/models/vocab.json")

    # Split dataset into Train (90%), Val (5%), Test (5%)
    random.shuffle(manifest)
    total = len(manifest)
    val_size = int(total * 0.05)
    test_size = int(total * 0.05)
    train_size = total - val_size - test_size

    train_ids = manifest[:train_size]
    val_ids = manifest[train_size: train_size + val_size]
    test_ids = manifest[train_size + val_size:]

    with open(os.path.join(out_dir, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest))
    with open(os.path.join(out_dir, "train_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(out_dir, "val_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids))
    with open(os.path.join(out_dir, "test_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_ids))

    print(f"\nPreprocessing Complete!")
    print(f"Total: {total} | Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    print(f"Vocab size: {len(vocab)} saved to {out_dir}/vocab.json")


if __name__ == "__main__":
    main()
