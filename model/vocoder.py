"""
HiFiGAN-tiny: a shrunk HiFi-GAN generator + a single-scale discriminator.

This is a *simplified* version of HiFi-GAN (paper uses multi-period +
multi-scale discriminators; this uses one discriminator to keep training
simpler and faster on a single-speaker dataset). Swap in the full
multi-discriminator setup later if you want to close the quality gap.

Generator: upsamples mel frames (hop=256) back to waveform via transposed
convs interleaved with residual dilated-conv blocks (the "MRF" idea from
the paper, simplified to one kernel size per stage instead of three).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=(kernel_size - 1) * d // 2,
                    )
                )
                for d in dilations
            ]
        )

    def forward(self, x):
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c in self.convs:
            remove_weight_norm(c)


class Generator(nn.Module):
    """
    Default config upsamples by 8*8*2*2 = 256, matching hop_length=256 in
    the mel front-end used in data/prepare_dataset.py.
    """

    def __init__(
        self,
        n_mels=80,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel=256,
        resblock_kernel_sizes=(3, 7, 11),
    ):
        super().__init__()
        assert len(upsample_rates) == len(upsample_kernel_sizes)
        self.n_upsamples = len(upsample_rates)
        self.n_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(
            nn.Conv1d(n_mels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for rate, k in zip(upsample_rates, upsample_kernel_sizes):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        ch, ch // 2, k, stride=rate, padding=(k - rate) // 2
                    )
                )
            )
            ch //= 2

        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for _ in upsample_rates:
            ch //= 2
            for k in resblock_kernel_sizes:
                self.resblocks.append(ResBlock(ch, kernel_size=k))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, mel):
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            rb_sum = None
            for j in range(self.n_kernels):
                rb_out = self.resblocks[i * self.n_kernels + j](x)
                rb_sum = rb_out if rb_sum is None else rb_sum + rb_out
            x = rb_sum / self.n_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)  # (B, 1, T_wav)

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Discriminator(nn.Module):
    """Single-scale waveform discriminator (simplified vs. HiFi-GAN's MPD+MSD)."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, 16, 15, stride=1, padding=7)),
                weight_norm(nn.Conv1d(16, 64, 41, stride=4, groups=4, padding=20)),
                weight_norm(nn.Conv1d(64, 256, 41, stride=4, groups=16, padding=20)),
                weight_norm(nn.Conv1d(256, 512, 41, stride=4, groups=16, padding=20)),
                weight_norm(nn.Conv1d(512, 512, 5, stride=1, padding=2)),
            ]
        )
        self.conv_post = weight_norm(nn.Conv1d(512, 1, 3, stride=1, padding=1))

    def forward(self, x):
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        return x.flatten(1, -1), feats


if __name__ == "__main__":
    gen = Generator()
    disc = Discriminator()
    mel = torch.randn(1, 80, 50)
    wav = gen(mel)
    print("generated waveform:", wav.shape)  # expect (1, 1, 50*256)
    score, feats = disc(wav)
    print("discriminator score:", score.shape)
    n_params_g = sum(p.numel() for p in gen.parameters())
    n_params_d = sum(p.numel() for p in disc.parameters())
    print(
        f"generator params: {n_params_g/1e6:.2f}M, discriminator params: {n_params_d/1e6:.2f}M"
    )
