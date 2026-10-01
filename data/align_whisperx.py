"""Create speech-aligned token durations with a Khmer CTC aligner.

Install WhisperX in an isolated environment, then run:
    python data/align_whisperx.py --data_dir km_kh_male --limit 20
    python data/align_whisperx.py --data_dir km_kh_male

Aligned durations are written to processed/durations_aligned/ so the
heuristic labels in processed/durations/ remain untouched.
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.khmer_tokenizer import KhmerTokenizer

SAMPLE_RATE = 22050
ALIGN_SAMPLE_RATE = 16000
HOP_LENGTH = 256
DEFAULT_MODEL = "vitouphy/wav2vec2-xls-r-300m-khmer"


def alignment_text(tokens: Sequence[str]) -> str:
    """Render tokenizer tokens as the exact character sequence to align."""
    if any(token == "<unk>" for token in tokens):
        raise ValueError("Transcript contains a character missing from the tokenizer vocabulary")
    return "".join(" " if token == "<space>" else token for token in tokens)


def durations_from_char_segments(
    tokens: Sequence[str],
    char_segments: Iterable[Dict],
    mel_frames: int,
) -> np.ndarray:
    """Map WhisperX character timings to positive frame counts summing to mel_frames."""
    if mel_frames < len(tokens):
        raise ValueError(
            f"Cannot assign at least one frame to each of {len(tokens)} tokens "
            f"within {mel_frames} mel frames"
        )

    expected_text = alignment_text(tokens)
    aligned_chars = []
    frame_weights = []
    for segment in char_segments:
        char = segment.get("char")
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(char, str) or not char:
            continue
        if start is None or end is None or end < start:
            raise ValueError(f"Character {char!r} has invalid alignment timestamps")

        codepoints = list(char)
        span = (end - start) * SAMPLE_RATE / HOP_LENGTH
        for codepoint in codepoints:
            aligned_chars.append(codepoint)
            frame_weights.append(span / len(codepoints))

    if "".join(aligned_chars) != expected_text:
        raise ValueError(
            "Aligned characters do not match the normalized tokenizer sequence"
        )
    if len(frame_weights) != len(tokens):
        raise ValueError(
            f"Aligner returned {len(frame_weights)} characters for {len(tokens)} tokens"
        )
    if not tokens:
        raise ValueError("Cannot create durations for an empty token sequence")

    weights = np.maximum(np.asarray(frame_weights, dtype=np.float64), 0.0)
    remaining = mel_frames - len(tokens)
    if remaining == 0:
        return np.ones(len(tokens), dtype=np.int64)
    if weights.sum() == 0:
        weights.fill(1.0)

    quotas = remaining * weights / weights.sum()
    extras = np.floor(quotas).astype(np.int64)
    unassigned = remaining - int(extras.sum())
    if unassigned:
        order = np.argsort(-(quotas - extras), kind="stable")
        extras[order[:unassigned]] += 1

    durations = extras + 1
    if int(durations.sum()) != mel_frames or np.any(durations < 1):
        raise RuntimeError("Duration allocation failed to preserve the mel-frame total")
    return durations


def read_transcripts(path: str) -> Dict[str, str]:
    transcripts = {}
    with open(path, encoding="utf-8", newline="") as metadata:
        for row in csv.reader(metadata, delimiter="|"):
            if len(row) >= 2:
                transcripts[row[0].strip()] = "|".join(row[1:]).strip()
    return transcripts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face CTC aligner")
    parser.add_argument("--duration_dir", default="durations_aligned")
    parser.add_argument("--limit", type=int, default=0, help="Align only the first N manifest items")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    proc_dir = os.path.join(args.data_dir, "processed")
    output_dir = os.path.join(proc_dir, args.duration_dir)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = KhmerTokenizer(vocab_path=os.path.join(proc_dir, "vocab.json"))
    transcripts = read_transcripts(os.path.join(args.data_dir, "metadata.csv"))
    with open(os.path.join(proc_dir, "manifest.txt"), encoding="utf-8") as manifest_file:
        manifest = [line.strip() for line in manifest_file if line.strip()]
    if args.limit:
        manifest = manifest[:args.limit]

    import torch
    import torchaudio.functional as audio_functional
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Khmer forced aligner {args.model} on {device}")
    align_model, align_metadata = whisperx.load_align_model(
        language_code="km",
        device=device,
        model_name=args.model,
    )

    aligned = 0
    skipped = 0
    report_path = os.path.join(output_dir, "alignment_report.jsonl")
    with open(report_path, "w", encoding="utf-8") as report_file:
        for utt_id in tqdm(manifest, desc="aligning Khmer speech"):
            output_path = os.path.join(output_dir, f"{utt_id}.npy")
            if os.path.exists(output_path) and not args.overwrite:
                skipped += 1
                continue

            try:
                if utt_id not in transcripts:
                    raise ValueError("Transcript is missing from metadata.csv")
                token_ids = np.load(os.path.join(proc_dir, "phonemes", f"{utt_id}.npy"))
                tokens = tokenizer.ids_to_tokens(token_ids.tolist())
                text = alignment_text(tokens)
                if not text.strip():
                    raise ValueError("Normalized transcript is empty")

                mel = np.load(os.path.join(proc_dir, "mels", f"{utt_id}.npy"), mmap_mode="r")
                waveform, sample_rate = sf.read(
                    os.path.join(proc_dir, "wavs_22k", f"{utt_id}.wav"), dtype="float32"
                )
                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=1)
                if sample_rate != SAMPLE_RATE:
                    raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
                if len(waveform) == 0:
                    raise ValueError("Audio file is empty")

                audio_16k = audio_functional.resample(
                    torch.from_numpy(waveform), SAMPLE_RATE, ALIGN_SAMPLE_RATE
                ).numpy()
                audio_duration = len(audio_16k) / ALIGN_SAMPLE_RATE
                result = whisperx.align(
                    [{"start": 0.0, "end": audio_duration, "text": text}],
                    align_model,
                    align_metadata,
                    audio_16k,
                    device,
                    return_char_alignments=True,
                )
                char_segments = [
                    char
                    for segment in result.get("segments", [])
                    for char in segment.get("chars", [])
                ]
                durations = durations_from_char_segments(tokens, char_segments, mel.shape[1])
                np.save(output_path, durations)
                report_file.write(
                    json.dumps(
                        {
                            "id": utt_id,
                            "text": text,
                            "characters": char_segments,
                            "durations": durations.tolist(),
                            "mel_frames": int(mel.shape[1]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                report_file.flush()
                aligned += 1
            except (OSError, ValueError, KeyError, RuntimeError, IndexError) as error:
                skipped += 1
                print(f"\nSkipped {utt_id}: {error}", file=sys.stderr)

    print(
        f"Aligned {aligned}/{len(manifest)} utterances; skipped {skipped}. "
        f"Durations: {output_dir}; review: {report_path}"
    )
    if aligned == 0:
        raise RuntimeError("No utterances were aligned; no training should be started")


if __name__ == "__main__":
    main()
