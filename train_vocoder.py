"""
Train HiFiGAN-tiny Vocoder on resampled wav audio from OpenSLR 42 / km_kh_male.
Features:
- Adversarial training (Generator vs Multi-Scale Discriminator)
- Feature Matching Loss + Mel Reconstruction Loss
- GPU/CPU device-safe MelSpectrogram transform
- Robust directory resolution and checkpoint resuming

Usage:
  python train_vocoder.py --data_dir km_kh_male --epochs 200 --batch_size 16
"""
import argparse
import os
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms as T

from model.vocoder import Generator, Discriminator

SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_MELS = 80
N_FFT = 1024
WIN_LENGTH = 1024
FMIN = 0
FMAX = 8000
SEGMENT_FRAMES = 32  # 32 frames * 256 hop = 8,192 audio samples


def resolve_proc_dir(data_dir: str) -> str:
    """Find the processed data directory reliably."""
    candidates = [
        os.path.join(data_dir, "processed"),
        data_dir,
        "processed",
        os.path.join("km_kh_male", "processed"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "mels")) and os.path.exists(os.path.join(c, "wavs_22k")):
            return c
    return os.path.join(data_dir, "processed")


class MelLoss(nn.Module):
    """Device-safe Mel-Spectrogram reconstruction loss for HiFi-GAN training."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=1.0,
        ).to(device)

    def forward(self, wav: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        pred = torch.log(torch.clamp(self.mel_transform(wav), min=1e-5))
        frames = min(pred.size(-1), target_mel.size(-1))
        return F.l1_loss(pred[..., :frames], target_mel[..., :frames])


class VocoderDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.proc = resolve_proc_dir(data_dir)
        manifest_file = os.path.join(self.proc, f"{split}_manifest.txt")
        if not os.path.exists(manifest_file):
            manifest_file = os.path.join(self.proc, "manifest.txt")

        with open(manifest_file, encoding="utf-8") as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self.wav_dir = os.path.join(self.proc, "wavs_22k")
        self.mel_dir = os.path.join(self.proc, "mels")

        # Filter IDs that have both WAV and Mel files
        self.ids = [
            i for i in all_ids
            if os.path.exists(os.path.join(self.wav_dir, f"{i}.wav"))
            and os.path.exists(os.path.join(self.mel_dir, f"{i}.npy"))
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        wav_file = os.path.join(self.wav_dir, f"{uid}.wav")
        mel_file = os.path.join(self.mel_dir, f"{uid}.npy")

        wav, _ = sf.read(wav_file)
        mel = np.load(mel_file)  # shape (80, T)

        expected_wav_len = mel.shape[1] * HOP_LENGTH
        if len(wav) < expected_wav_len:
            wav = np.pad(wav, (0, expected_wav_len - len(wav)))
        elif len(wav) > expected_wav_len:
            wav = wav[:expected_wav_len]

        # Pad if mel is shorter than required segment length
        if mel.shape[1] < SEGMENT_FRAMES:
            pad = SEGMENT_FRAMES - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)))
            wav = np.pad(wav, (0, pad * HOP_LENGTH))

        # Random segment extraction
        max_start = max(0, mel.shape[1] - SEGMENT_FRAMES)
        start = random.randint(0, max_start)
        mel_seg = mel[:, start : start + SEGMENT_FRAMES]
        wav_seg = wav[start * HOP_LENGTH : (start + SEGMENT_FRAMES) * HOP_LENGTH]

        return torch.from_numpy(mel_seg).float(), torch.from_numpy(wav_seg).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="km_kh_male")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=0, help="Optional max steps to run")
    ap.add_argument("--out_dir", default="checkpoints/vocoder")
    ap.add_argument("--resume", default="", help="Checkpoint path to resume from")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Vocoder Training Device: {device}")

    train_ds = VocoderDataset(args.data_dir, split="train")
    val_ds = VocoderDataset(args.data_dir, split="val")
    print(f"Vocoder Dataset: {len(train_ds)} train utterances, {len(val_ds)} val utterances (from {train_ds.proc})")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99))
    mel_criterion = MelLoss(device=device)
    writer = SummaryWriter(os.path.join(args.out_dir, "logs"))

    start_epoch = 0
    step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        gen.load_state_dict(ckpt.get("gen", ckpt))
        if "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_loss = float(ckpt.get("val_loss", float("inf")))
        print(f"Resumed vocoder checkpoint: {args.resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        gen.train()
        disc.train()

        for mel, wav in train_dl:
            mel, wav = mel.to(device), wav.to(device).unsqueeze(1)  # (B, 1, T)

            wav_gen = gen(mel)

            # 1. Discriminator Step
            d_real, _ = disc(wav)
            d_fake, _ = disc(wav_gen.detach())
            d_loss = torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # 2. Generator Step
            d_fake_for_g, feats_fake = disc(wav_gen)
            _, feats_real = disc(wav)
            adv_loss = torch.mean((d_fake_for_g - 1) ** 2)
            fm_loss = sum(F.l1_loss(ff, fr.detach()) for ff, fr in zip(feats_fake, feats_real))
            recon_loss = mel_criterion(wav_gen, mel)
            g_loss = adv_loss + 2.0 * fm_loss + 45.0 * recon_loss

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            if step % args.log_interval == 0:
                print(f"epoch {epoch:03d} step {step:05d} | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f} | recon: {recon_loss.item():.4f}", flush=True)
                writer.add_scalar("train/d_loss", d_loss.item(), step)
                writer.add_scalar("train/g_loss", g_loss.item(), step)
                writer.add_scalar("train/mel_recon_loss", recon_loss.item(), step)
            step += 1

            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

        val_loss = float("inf")
        if len(val_ds) > 0 and (epoch % 5 == 0 or epoch == args.epochs - 1):
            gen.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for mel, wav in val_dl:
                    mel, wav = mel.to(device), wav.to(device).unsqueeze(1)
                    total += mel_criterion(gen(mel), mel).item()
                    count += 1
            val_loss = total / max(count, 1)
            writer.add_scalar("val/mel_recon_loss", val_loss, epoch)
            print(f"--- Epoch {epoch:03d} Val mel loss: {val_loss:.4f} ---")

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
            torch.save({
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.out_dir, "best_vocoder.pt")
            torch.save({
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            print(f"Saved new best vocoder -> {best_path}")

    best_path = os.path.join(args.out_dir, "best_vocoder.pt")
    if not os.path.exists(best_path):
        torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(), "epoch": epoch if 'epoch' in locals() else 0}, best_path)
        print(f"Saved fallback best vocoder -> {best_path}")

    print("Vocoder training finished.")


if __name__ == "__main__":
    main()
