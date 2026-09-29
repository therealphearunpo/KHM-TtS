"""
FastSpeechLite: a small non-autoregressive text-to-mel model.

Encoder (phonemes -> hidden) -> Duration Predictor -> Length Regulator
(expand hidden states to frame-rate) -> Decoder (hidden -> mel).

No autoregressive loop anywhere, which is the point: inference cost is one
forward pass regardless of output length, so it's a good fit for low-latency
browser use.

Sized for a small single-speaker dataset (~4M params at these defaults) --
scale d_model / n_layers up if you have >~5 hours of audio and want more
headroom, or down further if you need an even smaller ONNX artifact.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SelfAttention(nn.Module):
    """Clean, fully ONNX-exportable Multi-Head Self Attention matching state dict keys."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        B, T = x.size(0), x.size(1)
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        return self.out_proj(out), None


class FFTBlock(nn.Module):
    """Transformer block: self-attention + conv1d feed-forward (as in FastSpeech)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        ff = x.transpose(1, 2)
        ff = self.conv2(self.act(self.conv1(ff)))
        ff = ff.transpose(1, 2)
        x = self.norm2(x + self.dropout(ff))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 256, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_hidden, kernel_size, padding=kernel_size // 2)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel_size, padding=kernel_size // 2)
        self.ln2 = nn.LayerNorm(d_hidden)
        self.linear = nn.Linear(d_hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, d_model)
        h = self.act(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        h = self.dropout(self.ln1(h))
        h = self.act(self.conv2(h.transpose(1, 2))).transpose(1, 2)
        h = self.dropout(self.ln2(h))
        log_duration = self.linear(h).squeeze(-1)  # (B, T), predicts log(duration+1)
        return log_duration


class LengthRegulator(nn.Module):
    """
    Expands each encoder frame by its (integer) predicted duration.
    Fully vectorized, ONNX-exportable implementation without Python loops or .item() calls.
    """

    def forward(self, x, durations, max_len: int = None):
        # x: (B, T_text, C), durations: (B, T_text) int
        durations = torch.clamp(durations, min=0)
        lengths = durations.sum(dim=-1)

        if max_len is not None:
            target_len = max_len
        else:
            target_len = lengths.max()

        cum_durs = torch.cumsum(durations, dim=-1)
        starts = cum_durs - durations

        grid = torch.arange(target_len, device=x.device, dtype=durations.dtype).unsqueeze(0).unsqueeze(-1)
        mask = ((grid >= starts.unsqueeze(1)) & (grid < cum_durs.unsqueeze(1))).to(x.dtype)

        expanded = torch.bmm(mask, x)
        return expanded, lengths


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_heads=2, d_ff=768, n_layers=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [FFTBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, phoneme_ids, key_padding_mask=None):
        x = self.embed(phoneme_ids)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model=192, n_heads=2, d_ff=768, n_layers=4, n_mels=80, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [FFTBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.mel_proj = nn.Linear(d_model, n_mels)

    def forward(self, x, key_padding_mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.mel_proj(x)


class FastSpeechLite(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_heads=2, d_ff=768,
                 n_enc_layers=4, n_dec_layers=4, n_mels=80, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, d_ff, n_enc_layers, dropout)
        self.duration_predictor = DurationPredictor(d_model)
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder(d_model, n_heads, d_ff, n_dec_layers, n_mels, dropout)

    def forward(self, phoneme_ids, durations=None, max_mel_len=None, speed: float = 1.0):
        """
        Training mode (durations given): teacher-forces the length regulator with real durations,
        returns predicted mel + predicted log-durations.

        Inference mode (durations=None): predicts durations itself, rounds to nearest int,
        and generates mel end-to-end.
        """
        phon_mask = phoneme_ids.eq(0)
        enc_out = self.encoder(phoneme_ids, key_padding_mask=phon_mask)
        log_duration_pred = self.duration_predictor(enc_out)

        if durations is None:
            if speed <= 0:
                raise ValueError("speed must be greater than zero")
            durations = torch.clamp(
                torch.round((torch.exp(log_duration_pred) - 1) / speed), min=1
            ).long()
            durations = durations.masked_fill(phon_mask, 0)

        expanded, out_lens = self.length_regulator(enc_out, durations, max_len=max_mel_len)
        mel_mask = torch.arange(expanded.size(1), device=expanded.device)[None, :] >= out_lens[:, None]
        mel = self.decoder(expanded, key_padding_mask=mel_mask)
        mel = mel.masked_fill(mel_mask.unsqueeze(-1), 0.0)
        return mel, log_duration_pred, out_lens


if __name__ == "__main__":
    model = FastSpeechLite(vocab_size=100)
    phon = torch.randint(1, 100, (2, 20))
    durs = torch.randint(1, 5, (2, 20))
    mel, log_dur, lens = model(phon, durations=durs)
    print("train-mode mel:", mel.shape, "lens:", lens)
    mel2, log_dur2, lens2 = model(phon)
    print("infer-mode mel:", mel2.shape, "lens:", lens2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")
