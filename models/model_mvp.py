# models/model_mvp.py
from __future__ import annotations
import torch
import torch.nn as nn

from models.backbones.raw_encoder import RawEncoder
from models.backbones.tf_encoder import TFEncoder
from models.fusion.fusion import FusionMLP


class EEGFMMVP(nn.Module):
    """
    Week-2 MVP:
      raw -> e_raw
      tf  -> e_tf
      fusion -> e_fused
    """

    def __init__(
            self,
            raw_in_ch: int,
            tf_in_ch: int,
            d: int = 256,
            raw_patch: int = 64,
            raw_depth: int = 6,
            tf_patch_f: int = 8,
            tf_patch_t: int = 64,
            tf_depth: int = 6,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.raw_enc = RawEncoder(
            in_ch=raw_in_ch,
            d=d,
            patch_size=raw_patch,
            depth=raw_depth,
            dropout=dropout,
        )
        self.tf_enc = TFEncoder(
            in_ch=tf_in_ch,
            d=d,
            patch_freq=tf_patch_f,
            patch_time=tf_patch_t,
            depth=tf_depth,
            dropout=dropout,
        )
        self.fusion = FusionMLP(d=d, hidden_mult=2, dropout=dropout)

    def forward(self, x_raw: torch.Tensor, x_tf: torch.Tensor):
        z_raw, e_raw = self.raw_enc(x_raw)
        z_tf, e_tf = self.tf_enc(x_tf)
        e_fused = self.fusion(e_raw, e_tf)
        return {
            "z_raw": z_raw,
            "e_raw": e_raw,
            "z_tf": z_tf,
            "e_tf": e_tf,
            "e_fused": e_fused,
        }
