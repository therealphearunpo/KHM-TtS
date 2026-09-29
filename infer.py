"""
End-to-End Inference CLI for Khmer Text-to-Speech (TTS).
Synthesizes written Khmer text into WAV audio using FastSpeechLite + HiFiGAN-tiny.

Usage:
  python infer.py --text "ស្ពាន កំពង់ ចម្លង អ្នកលឿង" --out output.wav
  python infer.py --text "ភ្លើង កំពុង ឆាប ឆេះ ផ្ទះ" --acoustic_ckpt checkpoints/acoustic/best_acoustic.pt --vocoder_ckpt checkpoints/vocoder/best_vocoder.pt
"""
import argparse
import os
import soundfile as sf
import torch
import numpy as np

from model.acoustic_model import FastSpeechLite
from model.vocoder import Generator
from data.khmer_tokenizer import KhmerTokenizer

SAMPLE_RATE = 22050


class KhmerTTSPipeline:
    def __init__(self, acoustic_ckpt: str = "checkpoints/acoustic/best_acoustic.pt",
                 vocoder_ckpt: str = "checkpoints/vocoder/best_vocoder.pt",
                 vocab_path: str = "web/models/vocab.json",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Khmer TTS Pipeline on: {self.device}")

        # 1. Tokenizer
        self.tokenizer = KhmerTokenizer(vocab_path=vocab_path)
        vocab_size = len(self.tokenizer.vocab)

        # 2. Acoustic Model
        self.acoustic = FastSpeechLite(vocab_size=vocab_size).to(self.device)
        if os.path.exists(acoustic_ckpt):
            ckpt = torch.load(acoustic_ckpt, map_location=self.device)
            checkpoint_vocab = ckpt.get("vocab")
            if checkpoint_vocab is not None and checkpoint_vocab != self.tokenizer.vocab:
                raise ValueError(
                    "Tokenizer vocabulary does not match the acoustic checkpoint. "
                    "Use the checkpoint's vocab.json/processed dataset."
                )
            try:
                self.acoustic.load_state_dict(ckpt["model"], strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    "Acoustic checkpoint is incompatible with the loaded vocabulary/model."
                ) from exc
            print(f"Loaded acoustic checkpoint: {acoustic_ckpt}")
        else:
            print(f"Warning: Acoustic checkpoint {acoustic_ckpt} not found. Using initialized weights.")
        self.acoustic.eval()

        # 3. Vocoder
        self.vocoder = Generator().to(self.device)
        if os.path.exists(vocoder_ckpt):
            ckpt = torch.load(vocoder_ckpt, map_location=self.device)
            try:
                self.vocoder.load_state_dict(ckpt["gen"], strict=True)
            except RuntimeError as exc:
                raise ValueError("Vocoder checkpoint is incompatible with this model.") from exc
            self.vocoder.remove_weight_norm()
            print(f"Loaded vocoder checkpoint: {vocoder_ckpt}")
        else:
            print(f"Warning: Vocoder checkpoint {vocoder_ckpt} not found. Using initialized weights.")
        self.vocoder.eval()

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Convert Khmer text into audio waveform numpy array."""
        token_ids = self.tokenizer.text_to_ids(text)
        if not token_ids:
            return np.zeros(0, dtype=np.float32)

        phon_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # Acoustic model forward pass (text -> mel)
            if speed <= 0:
                raise ValueError("speed must be greater than zero")
            mel_pred, log_dur_pred, out_lens = self.acoustic(phon_tensor, speed=speed)
            
            # Mel shape: (1, T_mel, 80) -> Vocoder expects (1, 80, T_mel)
            mel_len = int(out_lens[0].item())
            mel_for_vocoder = mel_pred[:, :mel_len, :].transpose(1, 2)

            # Vocoder forward pass (mel -> waveform)
            wav_pred = self.vocoder(mel_for_vocoder)
            waveform = wav_pred.squeeze().cpu().numpy()

        return waveform

    def synthesize_to_file(self, text: str, out_path: str, speed: float = 1.0):
        """Synthesize text and save directly to WAV file."""
        wav = self.synthesize(text, speed=speed)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        sf.write(out_path, wav, SAMPLE_RATE)
        duration = len(wav) / SAMPLE_RATE
        print(f"Saved audio ({duration:.2f}s) to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Input Khmer text to synthesize")
    ap.add_argument("--out", default="synthesized_output.wav", help="Output WAV path")
    ap.add_argument("--acoustic_ckpt", default="checkpoints/acoustic/best_acoustic.pt")
    ap.add_argument("--vocoder_ckpt", default="checkpoints/vocoder/best_vocoder.pt")
    ap.add_argument("--vocab", default="web/models/vocab.json")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    pipeline = KhmerTTSPipeline(
        acoustic_ckpt=args.acoustic_ckpt,
        vocoder_ckpt=args.vocoder_ckpt,
        vocab_path=args.vocab
    )
    pipeline.synthesize_to_file(args.text, args.out, speed=args.speed)


if __name__ == "__main__":
    main()
