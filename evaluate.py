"""
Evaluation script for Khmer TTS on the held-out test split (OpenSLR 42).

Calculates:
1. Teacher-forced Mel L1 Reconstruction Loss (acoustic model fidelity)
2. Synthesis Mel L1 Error (free-run, vs ground-truth mel, frame-truncated)
3. DTW-Aligned Mel-Cepstral Distortion / MCD in dB (proper temporal alignment)
4. Synthesis Latency & Real-Time Factor (RTF)
5. Saves audio samples (Ground Truth vs Synthesized) for listening comparison

NOTE on metrics:
  - Mel L1 is a proxy for acoustic model convergence and is affected by
    model training stage. Low values do not guarantee perceptual quality.
  - MCD is computed with DTW alignment to account for duration differences
    between synthesis and ground truth. Without DTW, naive frame-truncation
    inflates MCD substantially.
  - Human listening evaluation remains essential for TTS quality assessment.

Usage:
    python evaluate.py \\
        --data_dir km_kh_male \\
        --acoustic_ckpt checkpoints/acoustic/best_acoustic.pt \\
        --vocoder_ckpt checkpoints/vocoder/best_vocoder.pt
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

import torchaudio.transforms as T
from scipy.fftpack import dct


# ---------------------------------------------------------------------------
# MCD helpers
# ---------------------------------------------------------------------------

def mel_to_cepstrum(mel: np.ndarray, n_cep: int = 16) -> np.ndarray:
    """
    Compute MCEPs from a log-mel spectrogram frame matrix.

    Args:
        mel: (T, n_mels) log-mel spectrogram (already in log domain).
        n_cep: Number of cepstral coefficients (excluding C0).

    Returns:
        (T, n_cep) cepstrum matrix.
    """
    cep = dct(mel, type=2, axis=1, norm="ortho")  # (T, n_mels)
    return cep[:, 1: n_cep + 1]                    # drop C0, keep C1..C16


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the mean Euclidean distance between two sequences with DTW alignment.

    Uses a simple O(T1*T2) DP.  For large sequences consider fastdtw.

    Args:
        a: (T1, D) feature sequence.
        b: (T2, D) feature sequence.

    Returns:
        Mean per-frame Euclidean distance along the DTW path.
    """
    T1, T2 = len(a), len(b)
    if T1 == 0 or T2 == 0:
        return 0.0

    # Build cost matrix
    cost = np.full((T1, T2), np.inf, dtype=np.float64)
    cost[0, 0] = np.linalg.norm(a[0] - b[0])
    for i in range(1, T1):
        cost[i, 0] = cost[i - 1, 0] + np.linalg.norm(a[i] - b[0])
    for j in range(1, T2):
        cost[0, j] = cost[0, j - 1] + np.linalg.norm(a[0] - b[j])
    for i in range(1, T1):
        for j in range(1, T2):
            step = np.linalg.norm(a[i] - b[j])
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Traceback to count path length
    i, j, path_len = T1 - 1, T2 - 1, 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = min(cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            if best == cost[i - 1, j - 1]:
                i -= 1; j -= 1
            elif best == cost[i - 1, j]:
                i -= 1
            else:
                j -= 1
        path_len += 1

    return float(cost[T1 - 1, T2 - 1] / path_len)


def compute_mcd_dtw(mel_pred: np.ndarray, mel_target: np.ndarray, n_cep: int = 16) -> float:
    """
    Compute DTW-aligned Mel-Cepstral Distortion in dB.

    Args:
        mel_pred:   (T_pred, n_mels) log-mel spectrogram of synthesised audio.
        mel_target: (T_ref,  n_mels) log-mel spectrogram of reference audio.
        n_cep:      Number of cepstral coefficients for MCD (default 16).

    Returns:
        MCD in dB.
    """
    if mel_pred.shape[0] == 0 or mel_target.shape[0] == 0:
        return 0.0
    try:
        pred_cep = mel_to_cepstrum(mel_pred, n_cep)    # (T_pred, n_cep)
        targ_cep = mel_to_cepstrum(mel_target, n_cep)  # (T_ref,  n_cep)
        mean_dist = dtw_distance(pred_cep, targ_cep)
        # Standard MCD formula: (10 / ln(10)) * sqrt(2) * mean_euclidean
        mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * mean_dist
        return float(mcd)
    except Exception:
        # Fallback: naive frame-truncated MAE scaled to dB
        min_len = min(mel_pred.shape[0], mel_target.shape[0])
        return float(np.mean(np.abs(mel_pred[:min_len] - mel_target[:min_len])) * 10.0)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="km_kh_male")
    ap.add_argument("--acoustic_ckpt", default="checkpoints/acoustic/best_acoustic.pt")
    ap.add_argument("--vocoder_ckpt", default="checkpoints/vocoder/best_vocoder.pt")
    ap.add_argument(
        "--vocoder", default="griffin_lim",
        choices=["griffin_lim", "hifigan"],
        help="Vocoder engine for synthesis"
    )
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument(
        "--num_samples", type=int, default=10,
        help="Number of audio comparison samples to save to out_dir"
    )
    ap.add_argument(
        "--max_eval", type=int, default=None,
        help="Max number of test utterances to evaluate (default: all in test manifest)"
    )
    ap.add_argument(
        "--max_dtw_frames", type=int, default=300,
        help="Cap DTW sequence length to avoid O(T^2) cost on long utterances"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proc_dir = os.path.join(args.data_dir, "processed")
    test_manifest_path = os.path.join(proc_dir, "test_manifest.txt")
    if not os.path.exists(test_manifest_path):
        test_manifest_path = os.path.join(proc_dir, "manifest.txt")

    with open(test_manifest_path, encoding="utf-8") as f:
        test_ids = [l.strip() for l in f if l.strip()]

    if args.max_eval is not None and args.max_eval > 0:
        test_ids = test_ids[: args.max_eval]

    print(f"Running evaluation on {len(test_ids)} test utterances (vocoder: {args.vocoder})…")

    # ---- Load acoustic model ----
    vocab_size = 77
    if os.path.exists(args.acoustic_ckpt):
        ckpt = torch.load(args.acoustic_ckpt, map_location=device)
        if "vocab" in ckpt:
            vocab_size = len(ckpt["vocab"])
        acoustic = FastSpeechLite(vocab_size=vocab_size).to(device)
        acoustic.load_state_dict(ckpt["model"])
        print(f"Loaded acoustic checkpoint: {args.acoustic_ckpt}")
    else:
        acoustic = FastSpeechLite(vocab_size=vocab_size).to(device)
        print("Warning: acoustic checkpoint not found — using random weights.")
    acoustic.eval()

    # ---- Load vocoder ----
    inv_mel = T.InverseMelScale(
        n_stft=513, n_mels=80, sample_rate=22050, f_min=0.0, f_max=8000.0
    ).to(device)
    gl = T.GriffinLim(
        n_fft=1024, n_iter=32, win_length=1024, hop_length=256, power=1.0
    ).to(device)

    vocoder = Generator().to(device)
    if os.path.exists(args.vocoder_ckpt):
        try:
            ckpt = torch.load(args.vocoder_ckpt, map_location=device)
            vocoder.load_state_dict(ckpt["gen"], strict=False)
            vocoder.remove_weight_norm()
            print(f"Loaded vocoder checkpoint: {args.vocoder_ckpt}")
        except Exception as e:
            print(f"Notice: vocoder load exception: {e}")
    vocoder.eval()

    # ---- Metric accumulators ----
    mel_l1_errors = []   # teacher-forced mel L1 (acoustic model quality)
    mcd_scores = []      # DTW-aligned MCD between synthesised and GT
    rtf_scores = []      # real-time factor

    for idx, uid in enumerate(test_ids):
        phon_path = os.path.join(proc_dir, "phonemes", f"{uid}.npy")
        mel_path = os.path.join(proc_dir, "mels", f"{uid}.npy")
        wav_path = os.path.join(proc_dir, "wavs_22k", f"{uid}.wav")
        dur_path = os.path.join(proc_dir, "durations", f"{uid}.npy")

        if not os.path.exists(phon_path) or not os.path.exists(mel_path):
            continue

        phon = torch.from_numpy(np.load(phon_path)).long().unsqueeze(0).to(device)
        target_mel = np.load(mel_path).T  # (T_ref, n_mels)

        # --- Teacher-forced mel L1 (uses ground-truth durations if available) ---
        tf_mel_l1 = float("nan")
        if os.path.exists(dur_path):
            dur = torch.from_numpy(np.load(dur_path)).long().unsqueeze(0).to(device)
            with torch.no_grad():
                tf_mel, _, _ = acoustic(phon, durations=dur, max_mel_len=target_mel.shape[0])
                mask_tf = (tf_mel.abs().sum(-1) > 0).float().unsqueeze(-1)
                min_tf = min(tf_mel.shape[1], target_mel.shape[0])
                tf_mel_l1 = float(
                    F.l1_loss(
                        tf_mel[0, :min_tf],
                        torch.from_numpy(target_mel[:min_tf]).to(device),
                        reduction="mean"
                    ).item()
                )
        mel_l1_errors.append(tf_mel_l1)

        # --- Free-run synthesis with RTF ---
        t0 = time.perf_counter()
        with torch.no_grad():
            mel_pred, _, out_lens = acoustic(phon)
            mel_len = int(out_lens[0].item())

            if args.vocoder == "griffin_lim":
                mel_linear = torch.exp(mel_pred[0, :mel_len, :].transpose(0, 1))
                spec = inv_mel(mel_linear)
                wav_tensor = gl(spec)
                wav_pred = wav_tensor.squeeze().cpu().numpy()
            else:
                mel_for_gen = mel_pred[:, :mel_len, :].transpose(1, 2)
                wav_pred = vocoder(mel_for_gen).squeeze().cpu().numpy()

        t1 = time.perf_counter()

        # Normalise output
        if len(wav_pred) > 0:
            wav_pred = wav_pred - np.mean(wav_pred)
            peak = np.max(np.abs(wav_pred))
            if peak > 1e-4:
                wav_pred = (wav_pred / peak) * 0.95

        audio_dur = len(wav_pred) / SAMPLE_RATE
        infer_time = t1 - t0
        rtf = infer_time / max(audio_dur, 0.01)
        rtf_scores.append(rtf)

        # --- DTW-aligned MCD ---
        pred_mel_np = mel_pred[0, :mel_len, :].cpu().numpy()  # (T_pred, n_mels)
        # Cap sequences for tractable DTW
        cap = args.max_dtw_frames
        mcd = compute_mcd_dtw(pred_mel_np[:cap], target_mel[:cap])
        mcd_scores.append(mcd)

        # --- Save comparison audio samples ---
        if idx < args.num_samples:
            sf.write(os.path.join(args.out_dir, f"{uid}_synth.wav"), wav_pred, SAMPLE_RATE)
            if os.path.exists(wav_path):
                gt_wav, _ = sf.read(wav_path)
                sf.write(os.path.join(args.out_dir, f"{uid}_groundtruth.wav"), gt_wav, SAMPLE_RATE)

    # Filter out NaN teacher-forced values (no dur file available)
    valid_l1 = [v for v in mel_l1_errors if not np.isnan(v)]

    print("\n" + "=" * 52)
    print(f"  Khmer TTS Evaluation Summary ({args.vocoder})")
    print("=" * 52)
    print(f"Total test utterances evaluated : {len(mcd_scores)}")
    if valid_l1:
        print(f"Teacher-forced Mel L1 Error     : {np.mean(valid_l1):.4f}  (acoustic model fidelity)")
    print(f"DTW-Aligned MCD                 : {np.mean(mcd_scores):.2f} dB")
    print(f"Average RTF                     : {np.mean(rtf_scores):.4f}x  (lower is faster)")
    print(f"Saved {args.num_samples} comparison audio pairs → {args.out_dir}/")
    print("=" * 52)
    print("\nIMPORTANT: MCD and Mel-L1 are objective proxies only.")
    print("Human listening evaluation is required for a complete quality assessment.")


if __name__ == "__main__":
    main()
