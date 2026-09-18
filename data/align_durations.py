"""
Convert Montreal Forced Aligner TextGrid output into per-phoneme frame-count
durations matching the mel-spectrogram hop size used in prepare_dataset.py.

Expects MFA output at dataset/aligned/<id>.TextGrid (one "phones" tier), and
the phoneme id sequence already saved by prepare_dataset.py at
dataset/processed/phonemes/<id>.npy — durations are aligned to that sequence
by phoneme order (MFA's phone set is ARPAbet-with-stress like g2p_en's, but if
your alignment merges/silences differently, this script will report mismatches
instead of silently producing wrong durations).

Usage:
    python align_durations.py --data_dir /path/to/dataset
"""

import argparse
import json
import os

import numpy as np
import textgrid
from tqdm import tqdm

SAMPLE_RATE = 22050
HOP_LENGTH = 256


def frames_for(seconds: float) -> int:
    return int(round(seconds * SAMPLE_RATE / HOP_LENGTH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    proc_dir = os.path.join(args.data_dir, "processed")
    aligned_dir = os.path.join(args.data_dir, "aligned")
    dur_dir = os.path.join(proc_dir, "durations")
    os.makedirs(dur_dir, exist_ok=True)

    with open(os.path.join(proc_dir, "manifest.txt")) as f:
        manifest = [l.strip() for l in f if l.strip()]

    ok, skipped = 0, 0
    for utt_id in tqdm(manifest, desc="aligning durations"):
        tg_path = os.path.join(aligned_dir, f"{utt_id}.TextGrid")
        phon_path = os.path.join(proc_dir, "phonemes", f"{utt_id}.npy")
        if not os.path.exists(tg_path):
            skipped += 1
            continue

        tg = textgrid.TextGrid.fromFile(tg_path)
        phone_tier = None
        for tier in tg.tiers:
            if tier.name.lower() == "phones":
                phone_tier = tier
                break
        if phone_tier is None:
            skipped += 1
            continue

        durations = []
        for interval in phone_tier.intervals:
            label = interval.mark.strip()
            if label == "":
                continue  # MFA silence markers; drop or handle separately if you keep sil tokens
            n_frames = frames_for(interval.maxTime - interval.minTime)
            durations.append(max(n_frames, 1))

        phon_ids = np.load(phon_path)
        if len(durations) != len(phon_ids):
            # Mismatch usually means g2p_en's phoneme segmentation differs from MFA's
            # dictionary segmentation for this utterance. Skip rather than misalign.
            skipped += 1
            continue

        np.save(
            os.path.join(dur_dir, f"{utt_id}.npy"), np.array(durations, dtype=np.int64)
        )
        ok += 1

    print(f"durations written: {ok}, skipped (missing/mismatched): {skipped}")
    if skipped > ok * 0.1:
        print(
            "WARNING: high skip rate. Check that the phoneme set used for G2P in "
            "prepare_dataset.py matches the MFA dictionary/acoustic model you aligned with."
        )


if __name__ == "__main__":
    main()
