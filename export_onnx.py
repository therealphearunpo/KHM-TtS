"""
Export the trained acoustic model and vocoder to ONNX for onnxruntime-web and Python ONNX Runtime.

Usage:
  python export_onnx.py --acoustic_ckpt checkpoints/acoustic/best_acoustic.pt --vocoder_ckpt checkpoints/vocoder/best_vocoder.pt --out_dir web/models
"""
import argparse
import json
import os
import torch
import torch.nn as nn

from model.acoustic_model import FastSpeechLite
from model.vocoder import Generator


class AcousticExportWrapper(nn.Module):
    """
    Wraps FastSpeechLite for seamless ONNX export:
    Inputs: phoneme_ids (shape: 1 x L)
    Outputs:
      mel: shape (1, 80, T_mel) -> directly compatible with Vocoder input!
      mel_lengths: shape (1,)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, phoneme_ids):
        mel, _, out_lens = self.model(phoneme_ids, speed=1.0)
        # Transpose (B, T, 80) -> (B, 80, T) for direct vocoder chaining
        mel_t = mel.transpose(1, 2)
        return mel_t, out_lens


def export_acoustic(ckpt_path: str, out_dir: str):
    if not os.path.exists(ckpt_path):
        print(f"Warning: {ckpt_path} not found. Skipping acoustic export.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt.get("vocab")
    if vocab is None:
        for vpath in ("web/models/vocab.json", "processed/vocab.json"):
            if os.path.exists(vpath):
                with open(vpath, encoding="utf-8") as f:
                    vocab = json.load(f)
                break
    if vocab is None:
        raise ValueError("Vocabulary not found in checkpoint or disk.")

    model = FastSpeechLite(vocab_size=len(vocab))
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    export_model = AcousticExportWrapper(model)
    export_model.eval()

    dummy_phon = torch.randint(1, len(vocab), (1, 30), dtype=torch.long)
    onnx_path = os.path.join(out_dir, "acoustic.onnx")

    torch.onnx.export(
        export_model,
        (dummy_phon,),
        onnx_path,
        input_names=["phoneme_ids"],
        output_names=["mel", "mel_lengths"],
        dynamic_axes={
            "phoneme_ids": {0: "batch", 1: "phon_len"},
            "mel": {0: "batch", 2: "mel_len"},
            "mel_lengths": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    vocab_out = os.path.join(out_dir, "vocab.json")
    with open(vocab_out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Successfully exported acoustic model -> {onnx_path}")
    print(f"Synced vocabulary ({len(vocab)} tokens) -> {vocab_out}")


def export_vocoder(ckpt_path: str, out_dir: str):
    if not os.path.exists(ckpt_path):
        print(f"Warning: {ckpt_path} not found. Skipping vocoder export.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    gen = Generator()
    state = ckpt.get("gen", ckpt)
    gen.load_state_dict(state, strict=True)
    gen.eval()
    gen.remove_weight_norm()

    dummy_mel = torch.randn(1, 80, 50, dtype=torch.float32)
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
        do_constant_folding=True,
    )
    print(f"Successfully exported vocoder -> {onnx_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic_ckpt", default="checkpoints/acoustic/best_acoustic.pt")
    ap.add_argument("--vocoder_ckpt", default="checkpoints/vocoder/best_vocoder.pt")
    ap.add_argument("--out_dir", default="web/models")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    export_acoustic(args.acoustic_ckpt, args.out_dir)
    export_vocoder(args.vocoder_ckpt, args.out_dir)
    print(f"ONNX export complete. Artifacts saved in {args.out_dir}")


if __name__ == "__main__":
    main()
