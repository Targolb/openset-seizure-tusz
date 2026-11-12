# models/fusion.py
from __future__ import annotations
import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """
    concat(e_raw, e_tf) -> MLP -> fused embedding
    """

    def __init__(self, d: int = 256, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mult * d, d),
        )
        self.out_ln = nn.LayerNorm(d)

    def forward(self, e_raw: torch.Tensor, e_tf: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e_raw, e_tf], dim=-1)  # (B, 2d)
        z = self.net(x)  # (B, d)
        return self.out_ln(z)
