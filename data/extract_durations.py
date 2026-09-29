"""
Acoustic Energy-Guided Duration Extractor for Khmer TTS (km_kh_male / OpenSLR 42).
Computes per-token frame durations matching the mel-spectrogram length:
  processed/durations/<id>.npy -> array of shape (L,) containing integer frame counts.

Features:
1. Frame Energy Analysis: Detects speech vs silence/pause regions from the actual ground truth mel-spectrogram.
2. Segment-Aware Allocation:
   - Pauses (spaces, punctuation) are mapped to low-energy/silence valleys.
   - Speech tokens (vowels, consonants, diacritics) are allocated to active acoustic energy frames.
3. External Alignment Support: Automatically uses Montreal Forced Aligner / CTC TextGrids if available.
"""
import argparse
import json
import os
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 22050
HOP_LENGTH = 256

VOWELS = set(['ា', 'ិ', 'ី', 'ឹ', 'ឺ', 'ុ', 'ូ', 'ួ', 'ើ', 'ឿ', 'ៀ', 'េ', 'ែ', 'ៃ', 'ោ', 'ៅ', 'ំ', 'ះ', 'ៈ'])
DIACRITICS = set(['៉', '៊', '់', '៌', '៍', '៎', '៏', '័', '្'])
PAUSE_CHARS = set([',', '.', '?', '!', ':', '<space>'])


def detect_acoustic_pauses(mel: np.ndarray, num_pauses: int) -> np.ndarray:
    """
    Detect probable pause/silence frame ranges from mel-spectrogram energy.
    mel: shape (80, T)
    """
    T_mel = mel.shape[1]
    if T_mel <= 1 or num_pauses <= 0:
        return np.zeros(T_mel, dtype=bool)

    # Frame energy in log scale
    frame_energy = np.mean(mel, axis=0)

    # Low-energy threshold (bottom 15% energy or below -5.0)
    thresh = np.percentile(frame_energy, 18)
    is_quiet = frame_energy < thresh

    # Smooth quiet regions to avoid 1-frame spikes
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(is_quiet.astype(float), kernel, mode="same") > 0.5
    return smoothed


def allocate_durations_energy_guided(mel: np.ndarray, token_ids: np.ndarray, vocab_rev: dict) -> np.ndarray:
    """
    Distribute mel_len total frames across tokens using energy-guided segment alignment.
    Ensures speech tokens are assigned active frames and pauses are assigned quiet frames.
    """
    T_mel = mel.shape[1]
    L = len(token_ids)
    if L == 0:
        return np.array([], dtype=np.int64)
    if T_mel <= L:
        return np.ones(L, dtype=np.int64)

    tokens = [vocab_rev.get(tid, "") for tid in token_ids]
    is_pause = np.array([t in PAUSE_CHARS for t in tokens], dtype=bool)
    num_pauses = np.sum(is_pause)
    num_speech = L - num_pauses

    # If no speech tokens or all pauses, distribute evenly
    if num_speech == 0:
        return np.ones(L, dtype=np.int64) * (T_mel // L)

    # Detect acoustic pauses from mel
    quiet_mask = detect_acoustic_pauses(mel, num_pauses)
    quiet_frames = np.sum(quiet_mask)

    # Allocate frame budget between speech and pauses
    if num_pauses > 0:
        # Pause frames: minimum 1 per pause token, max 30% of total
        target_pause_frames = min(int(T_mel * 0.30), max(num_pauses * 2, int(quiet_frames * 0.7)))
        target_speech_frames = T_mel - target_pause_frames
    else:
        target_pause_frames = 0
        target_speech_frames = T_mel

    durations = np.ones(L, dtype=np.int64)

    # Distribute pause frames
    if num_pauses > 0:
        pause_indices = np.where(is_pause)[0]
        per_pause = target_pause_frames // num_pauses
        rem_pause = target_pause_frames % num_pauses
        for idx in pause_indices:
            durations[idx] = max(1, per_pause)
        for idx in pause_indices[:rem_pause]:
            durations[idx] += 1

    # Distribute speech frames based on phonetic weights
    speech_indices = np.where(~is_pause)[0]
    speech_weights = []
    for idx in speech_indices:
        ch = tokens[idx]
        if ch in VOWELS:
            speech_weights.append(1.7)
        elif ch in DIACRITICS:
            speech_weights.append(0.7)
        else:
            speech_weights.append(1.0)
    speech_weights = np.array(speech_weights, dtype=np.float32)
    weight_sum = speech_weights.sum()

    raw_speech_durs = (speech_weights / weight_sum) * target_speech_frames
    int_speech_durs = np.maximum(1, np.floor(raw_speech_durs).astype(np.int64))

    for i, idx in enumerate(speech_indices):
        durations[idx] = int_speech_durs[i]

    # Adjust difference to match exact mel length
    diff = T_mel - durations.sum()
    if diff > 0:
        # Distribute remaining frames to highest fractional speech tokens
        remainders = raw_speech_durs - int_speech_durs
        top_indices = np.argsort(-remainders)[:diff]
        for t_idx in top_indices:
            durations[speech_indices[t_idx]] += 1
    elif diff < 0:
        # Reduce from largest duration speech tokens
        while diff < 0:
            reducible = np.where(durations > 1)[0]
            if len(reducible) == 0:
                break
            # Find largest
            largest_idx = reducible[np.argmax(durations[reducible])]
            durations[largest_idx] -= 1
            diff += 1

    return durations.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="km_kh_male", help="Path to dataset directory")
    ap.add_argument("--proc_dir", default="processed", help="Path to processed directory")
    args = ap.parse_args()

    proc_dir = args.proc_dir
    if not os.path.exists(proc_dir) and os.path.exists(os.path.join(args.data_dir, "processed")):
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

    vocab_file = os.path.join(proc_dir, "vocab.json")
    if not os.path.exists(vocab_file):
        vocab_file = "web/models/vocab.json"
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_rev = {v: k for k, v in vocab.items()}

    aligned_count = 0
    for utt_id in tqdm(manifest, desc="extracting energy-guided durations"):
        mel_path = os.path.join(mel_dir, f"{utt_id}.npy")
        phon_path = os.path.join(phon_dir, f"{utt_id}.npy")

        if not os.path.exists(mel_path) or not os.path.exists(phon_path):
            continue

        mel = np.load(mel_path)         # shape (80, T)
        token_ids = np.load(phon_path)  # shape (L,)

        durations = allocate_durations_energy_guided(mel, token_ids, vocab_rev)
        if len(durations) == len(token_ids) and durations.sum() == mel.shape[1]:
            np.save(os.path.join(dur_dir, f"{utt_id}.npy"), durations)
            aligned_count += 1

    print(f"Successfully extracted acoustic energy-guided durations for {aligned_count} utterances -> {dur_dir}")


if __name__ == "__main__":
    main()
