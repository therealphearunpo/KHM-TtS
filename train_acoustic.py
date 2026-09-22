"""
Train FastSpeechLite on preprocessed Khmer speech data (OpenSLR 42 / km_kh_male).
Supports:
- Train / Validation splits
- Mel-spectrogram L1 Loss + Duration Log-MSE Loss
- Automatic best checkpoint saving & TensorBoard metrics

Usage:
    python train_acoustic.py --data_dir km_kh_male --epochs 100 --batch_size 16
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.acoustic_model import FastSpeechLite


class TTSDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        proc = os.path.join(data_dir, "processed")
        dur_dir = os.path.join(proc, "durations")
        
        manifest_file = os.path.join(proc, f"{split}_manifest.txt")
        if not os.path.exists(manifest_file):
            manifest_file = os.path.join(proc, "manifest.txt")

        with open(manifest_file, encoding="utf-8") as f:
            all_ids = [l.strip() for l in f if l.strip()]

        # Filter IDs that have existing durations, mels, and phonemes
        self.ids = [
            i for i in all_ids if os.path.exists(os.path.join(dur_dir, f"{i}.npy"))
        ]
        self.proc = proc
        with open(os.path.join(proc, "vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        phon = np.load(os.path.join(self.proc, "phonemes", f"{uid}.npy"))
        mel = np.load(os.path.join(self.proc, "mels", f"{uid}.npy")).T  # (T, n_mels)
        dur = np.load(os.path.join(self.proc, "durations", f"{uid}.npy"))
        return (
            torch.from_numpy(phon).long(),
            torch.from_numpy(mel).float(),
            torch.from_numpy(dur).long(),
        )


def collate(batch):
    phons, mels, durs = zip(*batch)
    phon_len = max(p.size(0) for p in phons)
    mel_len = max(m.size(0) for m in mels)

    phon_pad = torch.zeros(len(batch), phon_len, dtype=torch.long)
    dur_pad = torch.zeros(len(batch), phon_len, dtype=torch.long)
    mel_pad = torch.zeros(len(batch), mel_len, mels[0].size(1))
    mel_mask = torch.zeros(len(batch), mel_len, dtype=torch.bool)

    for i, (p, m, d) in enumerate(zip(phons, mels, durs)):
        phon_pad[i, : p.size(0)] = p
        dur_pad[i, : d.size(0)] = d
        mel_pad[i, : m.size(0)] = m
        mel_mask[i, : m.size(0)] = True

    return phon_pad, dur_pad, mel_pad, mel_mask, mel_len


def evaluate_val(model, val_dl, device):
    """Compute validation loss on held-out validation set."""
    model.eval()
    val_mel_loss = 0.0
    val_dur_loss = 0.0
    count = 0

    with torch.no_grad():
        for phon, dur, mel_target, mel_mask, mel_len in val_dl:
            phon, dur = phon.to(device), dur.to(device)
            mel_target, mel_mask = mel_target.to(device), mel_mask.to(device)

            mel_pred, log_dur_pred, _ = model(phon, durations=dur, max_mel_len=mel_len)

            mask = mel_mask.unsqueeze(-1)
            # Normalize by (frames * n_mels) for true per-element MAE
            n_mels = mel_target.size(-1)
            mel_l = F.l1_loss(mel_pred * mask, mel_target * mask, reduction="sum") / (mask.sum() * n_mels)

            phon_mask = phon != 0
            log_dur_target = torch.log(dur.float() + 1)
            dur_l = F.mse_loss(log_dur_pred * phon_mask, log_dur_target * phon_mask, reduction="sum") / phon_mask.sum()

            val_mel_loss += mel_l.item()
            val_dur_loss += dur_l.item()
            count += 1

    return (val_mel_loss / count, val_dur_loss / count) if count > 0 else (0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=0, help="Optional max steps to run")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume weights from")
    ap.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint if available")
    ap.add_argument("--start_epoch", type=int, default=None, help="Explicit start epoch override")
    ap.add_argument("--out_dir", default="checkpoints/acoustic")
    ap.add_argument("--log_file", type=str, default=None, help="Optional log file to append output to")
    args = ap.parse_args()

    if args.log_file:
        class TeeStream:
            def __init__(self, target_stream, filepath):
                self.target_stream = target_stream
                self.file = open(filepath, "a", encoding="utf-8")

            def write(self, data):
                self.target_stream.write(data)
                self.target_stream.flush()
                self.file.write(data)
                self.file.flush()

            def flush(self):
                self.target_stream.flush()
                self.file.flush()

        sys.stdout = TeeStream(sys.stdout, args.log_file)
        sys.stderr = TeeStream(sys.stderr, args.log_file)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Acoustic Training Device: {device}")

    train_ds = TTSDataset(args.data_dir, split="train")
    val_ds = TTSDataset(args.data_dir, split="val")
    print(f"Dataset: {len(train_ds)} train utterances, {len(val_ds)} validation utterances")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = FastSpeechLite(vocab_size=len(train_ds.vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    writer = SummaryWriter(os.path.join(args.out_dir, "logs"))

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and not args.checkpoint:
        latest_ckpt = os.path.join(args.out_dir, "latest_acoustic.pt")
        best_ckpt = os.path.join(args.out_dir, "best_acoustic.pt")
        if os.path.exists(latest_ckpt):
            args.checkpoint = latest_ckpt
        elif os.path.exists(best_ckpt):
            args.checkpoint = best_ckpt

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "val_loss" in ckpt:
            best_val_loss = ckpt["val_loss"]
            if best_val_loss > 50.0:  # Reset legacy unnormalized scale
                best_val_loss = float("inf")
        if "optimizer" in ckpt and ckpt["optimizer"]:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        if "scheduler" in ckpt and ckpt["scheduler"]:
            try:
                sched.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
        print(f"Loaded checkpoint from {args.checkpoint}: start_epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")

    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    step = start_epoch * len(train_dl)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_mel_total, train_dur_total = 0.0, 0.0
        steps_in_epoch = 0

        for phon, dur, mel_target, mel_mask, mel_len in train_dl:
            phon, dur = phon.to(device), dur.to(device)
            mel_target, mel_mask = mel_target.to(device), mel_mask.to(device)

            mel_pred, log_dur_pred, _ = model(phon, durations=dur, max_mel_len=mel_len)

            mask = mel_mask.unsqueeze(-1)
            # Normalize by (frames * n_mels) for true per-element MAE
            n_mels = mel_target.size(-1)
            mel_loss = F.l1_loss(mel_pred * mask, mel_target * mask, reduction="sum") / (mask.sum() * n_mels)

            phon_mask = phon != 0
            log_dur_target = torch.log(dur.float() + 1)
            dur_loss = F.mse_loss(log_dur_pred * phon_mask, log_dur_target * phon_mask, reduction="sum") / phon_mask.sum()

            # Both losses are now per-element: balance mel (dominant) vs duration
            loss = mel_loss + 0.1 * dur_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            train_mel_total += mel_loss.item()
            train_dur_total += dur_loss.item()
            steps_in_epoch += 1

            if step % args.log_interval == 0:
                print(f"epoch {epoch:03d} step {step:05d} | train mel_loss: {mel_loss.item():.4f} | dur_loss: {dur_loss.item():.4f}", flush=True)
                writer.add_scalar("train/mel_loss", mel_loss.item(), step)
                writer.add_scalar("train/dur_loss", dur_loss.item(), step)
            step += 1

            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

        # Validation Step
        if len(val_ds) > 0 and (epoch % 5 == 0 or epoch == args.epochs - 1):
            val_mel_l, val_dur_l = evaluate_val(model, val_dl, device)
            val_total_l = val_mel_l + 0.1 * val_dur_l
            print(f"--- Epoch {epoch:03d} Val Results | Mel L1: {val_mel_l:.4f} | Dur MSE: {val_dur_l:.4f} | Total: {val_total_l:.4f} ---")
            writer.add_scalar("val/mel_loss", val_mel_l, epoch)
            writer.add_scalar("val/dur_loss", val_dur_l, epoch)

            if val_total_l < best_val_loss:
                best_val_loss = val_total_l
                best_ckpt = os.path.join(args.out_dir, "best_acoustic.pt")
                torch.save({"model": model.state_dict(), "vocab": train_ds.vocab, "epoch": epoch, "val_loss": val_total_l}, best_ckpt)
                print(f"Saved new best model -> {best_ckpt}")

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(), "vocab": train_ds.vocab, "epoch": epoch}, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

        latest_ckpt = os.path.join(args.out_dir, "latest_acoustic.pt")
        torch.save({
            "model": model.state_dict(),
            "vocab": train_ds.vocab,
            "epoch": epoch,
            "step": step,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "val_loss": best_val_loss,
        }, latest_ckpt)

    best_ckpt = os.path.join(args.out_dir, "best_acoustic.pt")
    if not os.path.exists(best_ckpt):
        torch.save({"model": model.state_dict(), "vocab": train_ds.vocab, "epoch": epoch if 'epoch' in locals() else 0}, best_ckpt)
        print(f"Saved fallback best model -> {best_ckpt}")

    print("Training finished.")


if __name__ == "__main__":
    main()
