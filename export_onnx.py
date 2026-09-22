"""
Export the trained acoustic model and vocoder to ONNX for onnxruntime-web.

The acoustic model is exported as a streamlined inference graph that uses
torch.repeat_interleave (opset-supported) to implement the length regulator,
so output length is dynamic and matches the actual text input length at
runtime.  The legacy LengthRegulator.max_len=None path baked a constant int
via .item() during tracing, causing wrong-length outputs for all inputs —
this wrapper avoids that.

The ONNX export is numerically verified against PyTorch at the end;
exporting stops with an error if abs-diff > 1e-4.

Usage:
  python export_onnx.py \\
      --acoustic_ckpt checkpoints/acoustic/best_acoustic.pt \\
      --vocoder_ckpt checkpoints/vocoder/best_vocoder.pt \\
      --out_dir web/models
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from model.acoustic_model import FastSpeechLite
from model.vocoder import Generator


# ---------------------------------------------------------------------------
# Acoustic inference wrapper — avoids the .item() constant-baking in
# LengthRegulator when used via torch.onnx.export with dynamo=False.
# ---------------------------------------------------------------------------
class AcousticInferenceWrapper(nn.Module):
    """
    Wraps FastSpeechLite for ONNX export.

    Replaces the LengthRegulator forward (which calls .item() and bakes
    the output length as a constant during tracing) with a direct
    torch.repeat_interleave on the batch-size-1 tensors, which is fully
    supported by ONNX opset 13+ and exports with correct dynamic axes.
    """

    def __init__(self, model: FastSpeechLite):
        super().__init__()
        self.encoder = model.encoder
        self.duration_predictor = model.duration_predictor
        self.decoder = model.decoder

    def forward(self, phoneme_ids: torch.Tensor):
        # phoneme_ids: (1, T)
        enc_out = self.encoder(phoneme_ids)                          # (1, T, C)
        log_dur = self.duration_predictor(enc_out)                   # (1, T)
        durations = torch.clamp(
            torch.round(torch.exp(log_dur) - 1), min=1
        ).long()                                                     # (1, T)
        # Squeeze to 1-D for repeat_interleave (batch-size-1 only)
        expanded = torch.repeat_interleave(
            enc_out[0], durations[0], dim=0
        ).unsqueeze(0)                                               # (1, T_mel, C)
        mel = self.decoder(expanded)                                 # (1, T_mel, n_mels)
        return mel, log_dur


def export_acoustic(ckpt_path: str, out_dir: str, verify: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    base_model = FastSpeechLite(vocab_size=len(vocab))
    base_model.load_state_dict(ckpt["model"])
    base_model.eval()

    model = AcousticInferenceWrapper(base_model)
    model.eval()

    dummy_phon = torch.randint(1, len(vocab), (1, 30))

    onnx_path = os.path.join(out_dir, "acoustic.onnx")
    torch.onnx.export(
        model,
        (dummy_phon,),
        onnx_path,
        input_names=["phoneme_ids"],
        output_names=["mel", "log_duration"],
        dynamic_axes={
            "phoneme_ids": {0: "batch", 1: "phon_len"},
            "mel": {0: "batch", 1: "mel_len"},
            "log_duration": {0: "batch", 1: "phon_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported acoustic model → {onnx_path}")

    # Save vocabulary
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    if verify:
        _verify_acoustic(base_model, model, onnx_path, vocab)


def _verify_acoustic(
    base_model: FastSpeechLite,
    infer_wrapper: AcousticInferenceWrapper,
    onnx_path: str,
    vocab: dict,
    atol: float = 1e-4,
):
    """Assert PyTorch and ONNX Runtime outputs match within tolerance."""
    torch.manual_seed(0)
    test_ids = torch.randint(1, len(vocab), (1, 25))

    with torch.no_grad():
        pt_mel, _ = infer_wrapper(test_ids)

    sess = ort.InferenceSession(onnx_path)
    onnx_mel = sess.run(None, {"phoneme_ids": test_ids.numpy()})[0]

    max_diff = float(np.abs(pt_mel.numpy() - onnx_mel).max())
    mean_diff = float(np.abs(pt_mel.numpy() - onnx_mel).mean())
    print(f"  Acoustic ONNX verification: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    if max_diff > atol:
        raise RuntimeError(
            f"Acoustic ONNX verification failed: max_diff {max_diff:.2e} > atol {atol:.2e}"
        )
    print("  Acoustic ONNX ✓ numerically verified")


def export_vocoder(ckpt_path: str, out_dir: str, verify: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    gen = Generator()
    gen.load_state_dict(ckpt["gen"])
    gen.eval()
    gen.remove_weight_norm()

    dummy_mel = torch.randn(1, 80, 50)

    onnx_path = os.path.join(out_dir, "vocoder.onnx")
    torch.onnx.export(
        gen,
        (dummy_mel,),
        onnx_path,
        input_names=["mel"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {0: "batch", 2: "mel_len"},
            "waveform": {0: "batch", 2: "wav_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported vocoder → {onnx_path}")

    if verify:
        _verify_vocoder(gen, onnx_path)


def _verify_vocoder(gen: Generator, onnx_path: str, atol: float = 1e-4):
    """Assert PyTorch and ONNX Runtime vocoder outputs match."""
    np.random.seed(0)
    dummy = np.random.randn(1, 80, 80).astype(np.float32)

    with torch.no_grad():
        pt_wav = gen(torch.from_numpy(dummy)).numpy()

    sess = ort.InferenceSession(onnx_path)
    onnx_wav = sess.run(None, {"mel": dummy})[0]

    max_diff = float(np.abs(pt_wav - onnx_wav).max())
    mean_diff = float(np.abs(pt_wav - onnx_wav).mean())
    print(f"  Vocoder ONNX verification: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    if max_diff > atol:
        raise RuntimeError(
            f"Vocoder ONNX verification failed: max_diff {max_diff:.2e} > atol {atol:.2e}"
        )
    print("  Vocoder ONNX ✓ numerically verified")


def main():
    ap = argparse.ArgumentParser(description="Export TTS models to ONNX and verify.")
    ap.add_argument("--acoustic_ckpt", required=True, help="Path to acoustic checkpoint (.pt)")
    ap.add_argument("--vocoder_ckpt", required=True, help="Path to vocoder checkpoint (.pt)")
    ap.add_argument("--out_dir", default="web/models", help="Output directory for ONNX files")
    ap.add_argument("--no_verify", action="store_true", help="Skip numerical verification")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    verify = not args.no_verify

    export_acoustic(args.acoustic_ckpt, args.out_dir, verify=verify)
    export_vocoder(args.vocoder_ckpt, args.out_dir, verify=verify)
    print(f"\nDone. ONNX files + vocab.json written to: {args.out_dir}")


if __name__ == "__main__":
    main()
