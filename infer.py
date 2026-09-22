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


import torchaudio.transforms as T

class KhmerTTSPipeline:
    def __init__(self, acoustic_ckpt: str = None,
                 vocoder_ckpt: str = "checkpoints/vocoder/best_vocoder.pt",
                 vocab_path: str = "web/models/vocab.json",
                 vocoder_type: str = "griffin_lim",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocoder_type = vocoder_type
        print(f"Loading Khmer TTS Pipeline on: {self.device} (vocoder: {self.vocoder_type})")

        # 1. Tokenizer
        if not os.path.exists(vocab_path) and os.path.exists("km_kh_male/processed/vocab.json"):
            vocab_path = "km_kh_male/processed/vocab.json"
        self.tokenizer = KhmerTokenizer(vocab_path=vocab_path)
        vocab_size = len(self.tokenizer.vocab)

        # 2. Acoustic Model
        if acoustic_ckpt is None or not os.path.exists(acoustic_ckpt):
            if os.path.exists("checkpoints/acoustic/latest_acoustic.pt"):
                acoustic_ckpt = "checkpoints/acoustic/latest_acoustic.pt"
            elif os.path.exists("checkpoints/acoustic/best_acoustic.pt"):
                acoustic_ckpt = "checkpoints/acoustic/best_acoustic.pt"

        self.acoustic = FastSpeechLite(vocab_size=vocab_size).to(self.device)
        if acoustic_ckpt and os.path.exists(acoustic_ckpt):
            ckpt = torch.load(acoustic_ckpt, map_location=self.device)
            self.acoustic.load_state_dict(ckpt["model"], strict=False)
            print(f"Loaded acoustic checkpoint (strict=False): {acoustic_ckpt}")
        else:
            print(f"Warning: Acoustic checkpoint {acoustic_ckpt} not found. Using initialized weights.")
        self.acoustic.eval()

        # 3. Vocoders (Initialize Griffin-Lim and neural vocoder if available)
        self.inv_mel = T.InverseMelScale(n_stft=513, n_mels=80, sample_rate=22050, f_min=0.0, f_max=8000.0).to(self.device)
        self.gl = T.GriffinLim(n_fft=1024, n_iter=32, win_length=1024, hop_length=256, power=1.0).to(self.device)

        self.vocoder = Generator().to(self.device)
        self.vocoder_loaded = False
        if vocoder_ckpt and os.path.exists(vocoder_ckpt):
            try:
                ckpt = torch.load(vocoder_ckpt, map_location=self.device)
                self.vocoder.load_state_dict(ckpt["gen"], strict=False)
                self.vocoder.remove_weight_norm()
                self.vocoder.eval()
                self.vocoder_loaded = True
                print(f"Loaded vocoder checkpoint (strict=False): {vocoder_ckpt}")
            except Exception as e:
                print(f"Warning: Could not load vocoder checkpoint: {e}")
        else:
            self.vocoder.eval()
            print(f"Warning: Vocoder checkpoint {vocoder_ckpt} not found.")

    def synthesize(self, text: str, speed: float = 1.0, vocoder: str = None) -> np.ndarray:
        """Convert Khmer text into audio waveform numpy array."""
        text = text.strip() if text else ""
        if not text:
            raise ValueError("សូមបញ្ចូលអត្ថបទ (Please enter text to synthesize)")

        voc_type = vocoder or self.vocoder_type
        token_ids = self.tokenizer.text_to_ids(text)
        if not token_ids:
            raise ValueError(f"Could not convert text into valid tokens: '{text}'")

        phon_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            speed = float(speed)
            pace = 1.0 / max(speed, 0.2)

            enc_out = self.acoustic.encoder(phon_tensor)
            log_dur_pred = self.acoustic.duration_predictor(enc_out)
            durations = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1) * pace), min=1).long()

            expanded, out_lens = self.acoustic.length_regulator(enc_out, durations)
            mel_pred = self.acoustic.decoder(expanded)

            mel_len = max(1, int(out_lens[0].item()))

            if voc_type == "griffin_lim":
                mel_linear = torch.exp(mel_pred[0, :mel_len, :].transpose(0, 1))
                spec = self.inv_mel(mel_linear)
                wav_tensor = self.gl(spec)
                waveform = wav_tensor.squeeze().cpu().numpy()
            else:
                mel_for_vocoder = mel_pred[:, :mel_len, :].transpose(1, 2)
                wav_pred = self.vocoder(mel_for_vocoder)
                waveform = wav_pred.squeeze().cpu().numpy()

        if len(waveform) > 0:
            # Remove DC offset & peak normalize for clean, audible playback
            waveform = waveform - np.mean(waveform)
            peak = np.max(np.abs(waveform))
            if peak > 1e-4:
                waveform = (waveform / peak) * 0.95

        return waveform.astype(np.float32)

    def synthesize_to_file(self, text: str, out_path: str, speed: float = 1.0, vocoder: str = None):
        """Synthesize text and save directly to WAV file."""
        wav = self.synthesize(text, speed=speed, vocoder=vocoder)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) if os.path.dirname(out_path) else ".", exist_ok=True)
        sf.write(out_path, wav, SAMPLE_RATE)
        duration = len(wav) / SAMPLE_RATE
        print(f"Saved audio ({duration:.2f}s) to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Input Khmer text to synthesize")
    ap.add_argument("--out", default="synthesized_output.wav", help="Output WAV path")
    ap.add_argument("--acoustic_ckpt", default="checkpoints/acoustic/best_acoustic.pt")
    ap.add_argument("--vocoder_ckpt", default="checkpoints/vocoder/best_vocoder.pt")
    ap.add_argument("--vocoder", default="griffin_lim", choices=["griffin_lim", "hifigan"], help="Sound generator / vocoder engine")
    ap.add_argument("--vocab", default="web/models/vocab.json")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    pipeline = KhmerTTSPipeline(
        acoustic_ckpt=args.acoustic_ckpt,
        vocoder_ckpt=args.vocoder_ckpt,
        vocab_path=args.vocab,
        vocoder_type=args.vocoder
    )
    pipeline.synthesize_to_file(args.text, args.out, speed=args.speed, vocoder=args.vocoder)


if __name__ == "__main__":
    main()
