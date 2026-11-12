# models/raw_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool1D(nn.Module):
    """Attention pooling over token dimension: (B,N,d) -> (B,d)"""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d))
        self.proj = nn.Linear(d, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, d)
        q = self.q / (self.q.norm() + 1e-6)  # (d,)
        scores = torch.einsum("bnd,d->bn", z, q)  # (B, N)
        w = torch.softmax(scores, dim=1)  # (B, N)
        pooled = torch.einsum("bnd,bn->bd", z, w)  # (B, d)
        return self.proj(pooled)


class RawPatchEmbed1D(nn.Module):
    """
    Patchify along time using Conv1d:
      (B,C,T) -> Conv1d(C -> d, kernel=patch, stride=patch) -> (B,d,N) -> (B,N,d)
    """

    def __init__(self, in_ch: int, d: int, patch_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_ch, d, kernel_size=patch_size, stride=patch_size, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        z = self.proj(x)  # (B, d, N)
        z = z.transpose(1, 2)  # (B, N, d)
        return self.drop(z)


class MambaBlock(nn.Module):
    """Pre-LN residual Mamba block: z -> z + Mamba(LN(z))"""

    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()

        # ✅ Bypass mamba_ssm/__init__.py to avoid transformers dependency chain
        from mamba_ssm.modules.mamba_simple import Mamba

        self.ln = nn.LayerNorm(d)
        self.mamba = Mamba(
            d_model=d,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mamba(self.ln(z))  # (B, N, d)
        return z + self.drop(h)


class RawEncoder(nn.Module):
    """
    Raw EEG encoder MVP:
      x(B,C,T) -> z_raw(B,N,d) -> e_raw(B,d)
    """

    def __init__(
            self,
            in_ch: int,
            d: int = 256,
            patch_size: int = 64,
            depth: int = 6,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.1,
            use_attn_pool: bool = True,
    ):
        super().__init__()
        self.patch = RawPatchEmbed1D(in_ch=in_ch, d=d, patch_size=patch_size, dropout=dropout)

        self.blocks = nn.ModuleList([
            MambaBlock(d=d, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(depth)
        ])

        self.out_ln = nn.LayerNorm(d)
        self.pool = AttnPool1D(d) if use_attn_pool else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          z_raw: (B, N, d) tokens (for SSL losses)
          e_raw: (B, d) pooled embedding
        """
        z = self.patch(x)  # (B, N, d)
        for blk in self.blocks:
            z = blk(z)
        z = self.out_ln(z)

        if self.pool is not None:
            e = self.pool(z)  # (B, d)
        else:
            e = z.mean(dim=1)  # (B, d)

        return z, e
