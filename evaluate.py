"""
Evaluation script for Khmer TTS on the held-out test split (OpenSLR 42).
Calculates:
1. Mel L1 Reconstruction Loss
2. Mel Spectral Distortion (MCD in dB)
3. Synthesis Latency & Real-Time Factor (RTF)
4. Saves audio samples (Ground Truth vs Synthesized) for listening comparison

Usage:
    python evaluate.py --data_dir km_kh_male --acoustic_ckpt checkpoints/acoustic/best_acoustic.pt --vocoder_ckpt checkpoints/vocoder/best_vocoder.pt
"""
import argparse
import os
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from model.acoustic_model import FastSpeechLite
from model.vocoder import Generator
from data.prepare_dataset import SAMPLE_RATE


def compute_mcd(mel_pred: np.ndarray, mel_target: np.ndarray) -> float:
    """Compute Mel-Cepstral / Mel-Spectral Distortion approximation (dB)."""
    min_len = min(mel_pred.shape[0], mel_target.shape[0])
    if min_len == 0:
        return 0.0
    p = mel_pred[:min_len]
    t = mel_target[:min_len]
    diff = p - t
    mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=1))) * (10.0 / np.log(10.0)) * np.sqrt(2.0)
    return float(mcd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="km_kh_male")
    ap.add_argument("--acoustic_ckpt", default="checkpoints/acoustic/best_acoustic.pt")
    ap.add_argument("--vocoder_ckpt", default="checkpoints/vocoder/best_vocoder.pt")
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--num_samples", type=int, default=10, help="Number of audio comparison samples to save")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proc_dir = os.path.join(args.data_dir, "processed")
    test_manifest_path = os.path.join(proc_dir, "test_manifest.txt")
    if not os.path.exists(test_manifest_path):
        test_manifest_path = os.path.join(proc_dir, "manifest.txt")

    with open(test_manifest_path, encoding="utf-8") as f:
        test_ids = [l.strip() for l in f if l.strip()]

    print(f"Running evaluation on {len(test_ids)} test utterances...")

    # Load Models
    vocab_size = 77
    if os.path.exists(args.acoustic_ckpt):
        ckpt = torch.load(args.acoustic_ckpt, map_location=device)
        if "vocab" in ckpt:
            vocab_size = len(ckpt["vocab"])
        acoustic = FastSpeechLite(vocab_size=vocab_size).to(device)
        acoustic.load_state_dict(ckpt["model"])
    else:
        acoustic = FastSpeechLite(vocab_size=vocab_size).to(device)
    acoustic.eval()

    vocoder = Generator().to(device)
    if os.path.exists(args.vocoder_ckpt):
        ckpt = torch.load(args.vocoder_ckpt, map_location=device)
        vocoder.load_state_dict(ckpt["gen"])
        vocoder.remove_weight_norm()
    vocoder.eval()

    mel_l1_errors = []
    mcd_scores = []
    rtf_scores = []

    for idx, uid in enumerate(test_ids):
        phon_path = os.path.join(proc_dir, "phonemes", f"{uid}.npy")
        mel_path = os.path.join(proc_dir, "mels", f"{uid}.npy")
        wav_path = os.path.join(proc_dir, "wavs_22k", f"{uid}.wav")

        if not os.path.exists(phon_path) or not os.path.exists(mel_path):
            continue

        phon = torch.from_numpy(np.load(phon_path)).long().unsqueeze(0).to(device)
        target_mel = np.load(mel_path).T  # (T, 80)

        t0 = time.perf_counter()
        with torch.no_grad():
            mel_pred, _, out_lens = acoustic(phon)
            mel_len = int(out_lens[0].item())
            mel_for_gen = mel_pred[:, :mel_len, :].transpose(1, 2)
            wav_pred = vocoder(mel_for_gen).squeeze().cpu().numpy()
        t1 = time.perf_counter()

        pred_mel_np = mel_pred[0, :mel_len, :].cpu().numpy()
        audio_dur = len(wav_pred) / SAMPLE_RATE
        infer_time = t1 - t0
        rtf = infer_time / max(audio_dur, 0.01)
        rtf_scores.append(rtf)

        # Mel L1 & MCD
        min_len = min(pred_mel_np.shape[0], target_mel.shape[0])
        l1 = np.mean(np.abs(pred_mel_np[:min_len] - target_mel[:min_len]))
        mcd = compute_mcd(pred_mel_np, target_mel)
        mel_l1_errors.append(l1)
        mcd_scores.append(mcd)

        # Save side-by-side comparison audio for first few samples
        if idx < args.num_samples:
            sf.write(os.path.join(args.out_dir, f"{uid}_synth.wav"), wav_pred, SAMPLE_RATE)
            if os.path.exists(wav_path):
                target_wav, _ = sf.read(wav_path)
                sf.write(os.path.join(args.out_dir, f"{uid}_groundtruth.wav"), target_wav, SAMPLE_RATE)

    print("\n==================================================")
    print("           Khmer TTS Evaluation Summary           ")
    print("==================================================")
    print(f"Total Test Utterances: {len(mel_l1_errors)}")
    print(f"Average Mel L1 Error: {np.mean(mel_l1_errors):.4f}")
    print(f"Average MCD Distortion: {np.mean(mcd_scores):.2f} dB")
    print(f"Average Real-Time Factor (RTF): {np.mean(rtf_scores):.4f}x (Lower is faster)")
    print(f"Saved {args.num_samples} comparison audio pairs -> {args.out_dir}/")
    print("==================================================")


if __name__ == "__main__":
    main()
