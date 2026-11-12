import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_mvp import EEGFMMVP


class EEGFMMVPMixed(nn.Module):
    """
    Wraps EEGFMMVP with dataset-specific input adapters so we can mix datasets
    with different channel counts (e.g., SEED=62, TUH=19) while sharing encoders.
    """

    def __init__(
            self,
            seed_ch: int = 62,
            tuh_ch: int = 19,
            common_ch: int = 62,
            **mvp_kwargs,
    ):
        super().__init__()

        self.common_ch = common_ch
        self.seed_ch = seed_ch
        self.tuh_ch = tuh_ch

        # --- Adapters ---
        # raw: (B,C,T) -> Conv1d(Cin, Ccommon, 1)
        self.raw_adapt_seed = nn.Identity() if seed_ch == common_ch else nn.Conv1d(seed_ch, common_ch, 1)
        self.raw_adapt_tuh = nn.Conv1d(tuh_ch, common_ch, 1) if tuh_ch != common_ch else nn.Identity()

        # tf: (B,C,F,T) -> Conv2d(Cin, Ccommon, 1) treating (F,T) as image
        self.tf_adapt_seed = nn.Identity() if seed_ch == common_ch else nn.Conv2d(seed_ch, common_ch, 1)
        self.tf_adapt_tuh = nn.Conv2d(tuh_ch, common_ch, 1) if tuh_ch != common_ch else nn.Identity()

        # Shared backbone sees common_ch for both modalities
        self.backbone = EEGFMMVP(raw_in_ch=common_ch, tf_in_ch=common_ch, **mvp_kwargs)

    def _adapt(self, x_raw, x_tf, dataset: str):
        if dataset.lower() in ["seed", "seed_eeg", "seed_eeg"]:
            x_raw = self.raw_adapt_seed(x_raw)  # (B,62,T)
            x_tf = self.tf_adapt_seed(x_tf)  # (B,62,F,T)
        elif dataset.lower() in ["tuh", "tuh_eeg"]:
            x_raw = self.raw_adapt_tuh(x_raw)  # (B,62,T)
            x_tf = self.tf_adapt_tuh(x_tf)  # (B,62,F,T)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        return x_raw, x_tf

    def forward(self, x_raw, x_tf, dataset: str):
        x_raw, x_tf = self._adapt(x_raw, x_tf, dataset)
        return self.backbone(x_raw, x_tf)
