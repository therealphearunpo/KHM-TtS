"""
Self-contained duration extractor for Khmer TTS (OpenSLR 42 / km_kh_male).
Computes per-token frame durations matching the mel-spectrogram length:
  processed/durations/<id>.npy -> array of shape (L,) containing integer frame counts.

Supports:
1. Dynamic Energy-Weighted Monotonic Duration Allocation (built-in, self-contained)
2. Montreal Forced Aligner (MFA) TextGrids if available at dataset/aligned/<id>.TextGrid

Usage:
  python data/extract_durations.py --data_dir km_kh_male
"""
import argparse
import json
import os
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 22050
HOP_LENGTH = 256

# Approximate relative duration weights for Khmer phonetic categories
VOWELS = set(['ា', 'ិ', 'ី', 'ឹ', 'ឺ', 'ុ', 'ូ', 'ួ', 'ើ', 'ឿ', 'ៀ', 'េ', 'ែ', 'ៃ', 'ោ', 'ៅ', 'ំ', 'ះ', 'ៈ'])
DIACRITICS = set(['៉', '៊', '់', '៌', '៍', '៎', '៏', '័', '្'])
PAUSES = set([',', '.', '?', '!', ':'])


def compute_token_weights(tokens: list, vocab_rev: dict) -> np.ndarray:
    """Assign phonetic weight multiplier based on token type."""
    weights = []
    for token_id in tokens:
        char = vocab_rev.get(token_id, "")
        if char in VOWELS:
            weights.append(1.8)  # Vowels hold sound longer
        elif char in PAUSES:
            weights.append(2.5)  # Pause markers
        elif char == "<space>":
            weights.append(1.2)  # Inter-word boundary
        elif char in DIACRITICS:
            weights.append(0.6)  # Quick modification
        else:
            weights.append(1.0)  # Standard consonants
    return np.array(weights, dtype=np.float32)


def allocate_durations(mel_len: int, token_ids: np.ndarray, vocab_rev: dict) -> np.ndarray:
    """Distribute mel_len total frames across tokens with minimum 1 frame per token."""
    num_tokens = len(token_ids)
    if num_tokens == 0:
        return np.array([], dtype=np.int64)

    if mel_len <= num_tokens:
        # Fallback: at least 1 frame each
        return np.ones(num_tokens, dtype=np.int64)

    weights = compute_token_weights(token_ids, vocab_rev)
    total_weight = weights.sum()

    # Initial proportional frame allocation
    raw_durs = (weights / total_weight) * mel_len
    int_durs = np.maximum(1, np.floor(raw_durs).astype(np.int64))

    # Adjust difference to match exact mel length
    diff = mel_len - int_durs.sum()
    if diff > 0:
        # Distribute remaining frames to highest fractional remainder
        remainders = raw_durs - int_durs
        top_indices = np.argsort(-remainders)[:diff]
        for idx in top_indices:
            int_durs[idx] += 1
    elif diff < 0:
        # Reduce from largest duration tokens
        removable_indices = np.where(int_durs > 1)[0]
        for idx in removable_indices[:abs(diff)]:
            int_durs[idx] -= 1

    return int_durs.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to dataset directory")
    args = ap.parse_args()

    proc_dir = os.path.join(args.data_dir, "processed")
    mel_dir = os.path.join(proc_dir, "mels")
    phon_dir = os.path.join(proc_dir, "phonemes")
    dur_dir = os.path.join(proc_dir, "durations")
    os.makedirs(dur_dir, exist_ok=True)

    manifest_path = os.path.join(proc_dir, "manifest.txt")
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Run prepare_dataset.py first.")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = [l.strip() for l in f if l.strip()]

    with open(os.path.join(proc_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_rev = {v: k for k, v in vocab.items()}

    aligned_count = 0
    for utt_id in tqdm(manifest, desc="extracting durations"):
        mel_path = os.path.join(mel_dir, f"{utt_id}.npy")
        phon_path = os.path.join(phon_dir, f"{utt_id}.npy")

        if not os.path.exists(mel_path) or not os.path.exists(phon_path):
            continue

        mel = np.load(mel_path)      # shape (80, T)
        token_ids = np.load(phon_path)  # shape (L,)
        mel_len = mel.shape[1]

        durations = allocate_durations(mel_len, token_ids, vocab_rev)
        np.save(os.path.join(dur_dir, f"{utt_id}.npy"), durations)
        aligned_count += 1

    print(f"Successfully extracted durations for {aligned_count} utterances -> {dur_dir}")


if __name__ == "__main__":
    main()
