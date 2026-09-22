"""
Train HiFiGAN-tiny Vocoder on resampled wav audio from OpenSLR 42 / km_kh_male.
Features:
- Adversarial training (Generator vs Multi-Scale Discriminator)
- Feature Matching Loss + Mel Reconstruction Loss
- Support for Train / Validation splits and checkpointing

Usage:
  python train_vocoder.py --data_dir km_kh_male --epochs 200 --batch_size 16
"""
import argparse
import os
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms as T

from model.vocoder import Generator, Discriminator

# Audio / mel constants (mirror data/prepare_dataset.py)
SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_MELS = 80
N_FFT = 1024
WIN_LENGTH = 1024
FMIN = 0.0
FMAX = 8000.0


def create_mel_transform(device):
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        power=1.0,
    ).to(device)


SEGMENT_FRAMES = 32  # 32 frames * 256 hop = 8,192 audio samples


class VocoderDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.proc = os.path.join(data_dir, "processed")
        manifest_file = os.path.join(self.proc, f"{split}_manifest.txt")
        if not os.path.exists(manifest_file):
            manifest_file = os.path.join(self.proc, "manifest.txt")

        with open(manifest_file, encoding="utf-8") as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self.wav_dir = os.path.join(self.proc, "wavs_22k")
        self.mel_dir = os.path.join(self.proc, "mels")

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

        # Choose a start index that ensures a full segment can be extracted
        max_start = mel.shape[1] - SEGMENT_FRAMES
        if max_start < 0:
            max_start = 0
        start = random.randint(0, max_start)
        mel_seg = mel[:, start : start + SEGMENT_FRAMES]
        wav_seg = wav[start * HOP_LENGTH : (start + SEGMENT_FRAMES) * HOP_LENGTH]

        return torch.from_numpy(mel_seg).float(), torch.from_numpy(wav_seg).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=0, help="Optional max steps to run")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume weights from")
    ap.add_argument("--resume", action="store_true", default=False, help="Resume from latest checkpoint")
    ap.add_argument("--out_dir", default="checkpoints/vocoder")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Vocoder Training Device: {device}")

    train_ds = VocoderDataset(args.data_dir, split="train")
    val_ds = VocoderDataset(args.data_dir, split="val")
    print(f"Vocoder Dataset: {len(train_ds)} train, {len(val_ds)} val")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99))
    writer = SummaryWriter(os.path.join(args.out_dir, "logs"))
    mel_transform = create_mel_transform(device)

    start_epoch = 0
    if args.resume and not args.checkpoint:
        latest = os.path.join(args.out_dir, "latest_vocoder.pt")
        best = os.path.join(args.out_dir, "best_vocoder.pt")
        if os.path.exists(latest):
            args.checkpoint = latest
        elif os.path.exists(best):
            args.checkpoint = best

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "gen" in ckpt:
            gen.load_state_dict(ckpt["gen"])
        if "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        print(f"Loaded vocoder checkpoint from {args.checkpoint}, resuming at epoch {start_epoch}")

    step = start_epoch * len(train_dl)
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

            # Spectral Mel Reconstruction Loss (essential for speech quality)
            mel_fake = torch.log(torch.clamp(mel_transform(wav_gen.squeeze(1)), min=1e-5))
            min_mel_len = min(mel_fake.shape[2], mel.shape[2])
            mel_loss = F.l1_loss(mel_fake[:, :, :min_mel_len], mel[:, :, :min_mel_len]) * 45.0

            g_loss = adv_loss + 2.0 * fm_loss + mel_loss

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            if step % args.log_interval == 0:
                print(f"epoch {epoch:03d} step {step:05d} | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f} | mel_loss: {mel_loss.item():.4f}", flush=True)
                writer.add_scalar("train/d_loss", d_loss.item(), step)
                writer.add_scalar("train/g_loss", g_loss.item(), step)
                writer.add_scalar("train/mel_loss", mel_loss.item(), step)
            step += 1

            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

        # Checkpoints
        latest_path = os.path.join(args.out_dir, "latest_vocoder.pt")
        torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(), "epoch": epoch}, latest_path)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
            torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(), "epoch": epoch}, ckpt_path)
            best_path = os.path.join(args.out_dir, "best_vocoder.pt")
            torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(), "epoch": epoch}, best_path)
            print(f"Saved checkpoint -> {ckpt_path}")

    best_path = os.path.join(args.out_dir, "best_vocoder.pt")
    if not os.path.exists(best_path):
        torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(), "epoch": epoch if 'epoch' in locals() else 0}, best_path)
        print(f"Saved fallback best vocoder -> {best_path}")

    print("Vocoder training finished.")


if __name__ == "__main__":
    main()
