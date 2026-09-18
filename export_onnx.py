"""
Export the trained acoustic model and vocoder to ONNX for onnxruntime-web.

Usage:
python export_onnx.py \
    --acoustic_ckpt checkpoints/acoustic/epoch_199.pt \
    --vocoder_ckpt checkpoints/vocoder/epoch_499.pt \
    --out_dir web/models
"""

import argparse
import json
import os

import torch

from model.acoustic_model import FastSpeechLite
from model.vocoder import Generator


def export_acoustic(ckpt_path, out_dir):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    model = FastSpeechLite(vocab_size=len(vocab))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy_phon = torch.randint(1, len(vocab), (1, 40))

    torch.onnx.export(
        model,
        (dummy_phon,),
        os.path.join(out_dir, "acoustic.onnx"),
        input_names=["phoneme_ids"],
        output_names=["mel", "log_duration", "mel_lengths"],
        dynamic_axes={
            "phoneme_ids": {0: "batch", 1: "phon_len"},
            "mel": {0: "batch", 1: "mel_len"},
            "log_duration": {0: "batch", 1: "phon_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("exported acoustic.onnx")


def export_vocoder(ckpt_path, out_dir):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    gen = Generator()
    gen.load_state_dict(ckpt["gen"])
    gen.eval()
    gen.remove_weight_norm()

    dummy_mel = torch.randn(1, 80, 50)

    torch.onnx.export(
        gen,
        (dummy_mel,),
        os.path.join(out_dir, "vocoder.onnx"),
        input_names=["mel"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {0: "batch", 2: "mel_len"},
            "waveform": {0: "batch", 2: "wav_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    print("exported vocoder.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic_ckpt", required=True)
    ap.add_argument("--vocoder_ckpt", required=True)
    ap.add_argument("--out_dir", default="web/models")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    export_acoustic(args.acoustic_ckpt, args.out_dir)
    export_vocoder(args.vocoder_ckpt, args.out_dir)
    print(f"done. ONNX files + vocab.json are in {args.out_dir}")


if __name__ == "__main__":
    main()
