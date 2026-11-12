# models/backbones/tf_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn


class AttnPoolTokens(nn.Module):
    """Attention pooling over tokens: (B,N,d) -> (B,d)"""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d))
        self.proj = nn.Linear(d, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        q = self.q / (self.q.norm() + 1e-6)
        scores = torch.einsum("bnd,d->bn", z, q)  # (B, N)
        w = torch.softmax(scores, dim=1)  # (B, N)
        pooled = torch.einsum("bnd,bn->bd", z, w)  # (B, d)
        return self.proj(pooled)


class TFEncoder(nn.Module):
    """
    TF (CWT) encoder MVP (ViT-like):
      tf(B, C, F, T) -> patchify (Conv2d) -> tokens (B, N, d)
      -> Transformer blocks -> pooled embedding (B, d)

    Supports input shapes:
      - (B, C, F, T)  [recommended]
      - (B, F, T)     [auto adds C=1]
      - (C, F, T) or (F, T) [auto adds B=1]
    """

    def __init__(
            self,
            in_ch: int,
            d: int = 256,
            patch_freq: int = 8,
            patch_time: int = 8,
            depth: int = 6,
            n_heads: int = 8,
            mlp_ratio: int = 4,
            dropout: float = 0.1,
            use_attn_pool: bool = True,
    ):
        super().__init__()

        self.patch = nn.Conv2d(
            in_channels=in_ch,
            out_channels=d,
            kernel_size=(patch_freq, patch_time),
            stride=(patch_freq, patch_time),
            bias=True,
        )
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.out_ln = nn.LayerNorm(d)
        self.pool = AttnPoolTokens(d) if use_attn_pool else None

    @staticmethod
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        # Accept (F,T), (C,F,T), (B,F,T), (B,C,F,T)
        if x.ndim == 2:  # (F,T)
            x = x.unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
        elif x.ndim == 3:
            # Could be (C,F,T) or (B,F,T). We assume if first dim <= 128 it's likely C.
            # To avoid guessing wrong, treat as (C,F,T) and add B=1.
            x = x.unsqueeze(0)  # (1,C,F,T)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported tf shape: {tuple(x.shape)}")
        return x

    def forward(self, tf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          z_tf: (B, N, d) tokens
          e_tf: (B, d) pooled embedding
        """
        tf = self._to_bchw(tf)  # (B,C,F,T)
        z = self.patch(tf)  # (B,d,F',T')
        z = z.flatten(2).transpose(1, 2)  # (B,N,d)
        z = self.drop(z)

        z = self.encoder(z)  # (B,N,d)
        z = self.out_ln(z)

        e = self.pool(z) if self.pool is not None else z.mean(dim=1)
        return z, e
