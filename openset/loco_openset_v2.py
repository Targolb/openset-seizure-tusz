#!/usr/bin/env python3
"""
loco_openset_v2.py
==================
Leave-One-Class-Out (LOCO) open-set evaluation for seizure type
classification. Improvements over v1:

  1. CCR@FPR: Correct Classification Rate at controlled False Positive
     Rate (FPR=0.05, 0.10, 0.20). This is the OSCR metric from
     Scheirer et al. — measures what fraction of accepted known windows
     are correctly classified, at a threshold where X% of unknowns
     are incorrectly accepted.

  2. Early stopping: patience=7 epochs on internal val accuracy.
     Prevents overfitting (rounds 0-2) and allows round 3 (tcsz) to
     train longer than 30 epochs if still improving.

  3. Full epoch curves saved per round for plotting.

  4. PosixPath JSON serialization fixed.

Usage
-----
  python loco_openset_v2.py \
      --index_csv  /NCL_VAST_Datasets/Targol/eegfm_outputs/TUH/openset/embed_index.csv \
      --ckpt       /NCL_VAST_Datasets/Targol/eegfm_checkpoints/week3_mixed_tuh_seed/week3_mixed_step20000.pt \
      --out_dir    /NCL_VAST_Datasets/Targol/eegfm_outputs/TUH/openset/loco_v2 \
      --max_epochs 50 \
      --patience   7 \
      --batch_size 64 \
      --lr         3e-4 \
      --device     cuda

  # Skip retraining — use existing checkpoints from loco/ folder:
  python loco_openset_v2.py ... --eval_only \
      --ckpt_dir /NCL_VAST_Datasets/Targol/eegfm_outputs/TUH/openset/loco
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pywt
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_KNOWN = ["fnsz", "gnsz", "cpsz", "tcsz"]
TRUE_UNKNOWNS = ["absz", "tnsz", "spsz", "mysz"]

FS_TARGET = 256
FMIN_HZ = 0.5
FMAX_HZ = 70.0
N_FREQS = 64
WAVELET = "morl"
WINDOW_SAMP = 15 * FS_TARGET
KEEP_CH_INDICES = list(range(19))
N_CH = 19
TARGET_FPR = 0.05

_FREQS_HZ = np.linspace(FMIN_HZ, FMAX_HZ, N_FREQS)
_CENTRAL_FREQ = pywt.central_frequency(WAVELET)
_SCALES = (_CENTRAL_FREQ * FS_TARGET) / _FREQS_HZ

FPR_OPERATING_POINTS = [0.05, 0.10, 0.20]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TUSZWindowDataset(Dataset):
    def __init__(self, rows, le, augment=False):
        self.rows = rows
        self.le = le
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        raw_23 = np.load(row["path"]).astype(np.float32)
        raw_19 = raw_23[KEEP_CH_INDICES, :]

        if self.augment:
            mask = np.random.rand(N_CH) > 0.10
            raw_19 = raw_19 * mask[:, None]

        tf = np.zeros((N_CH, N_FREQS, WINDOW_SAMP), dtype=np.float32)
        for c in range(N_CH):
            coeffs, _ = pywt.cwt(raw_19[c], _SCALES, WAVELET,
                                 sampling_period=1.0 / FS_TARGET)
            tf[c] = np.abs(coeffs).astype(np.float32)

        sztype = row["seizure_type"]
        label = int(self.le.transform([sztype])[0]) \
            if sztype in self.le.classes_ else -1

        return (torch.from_numpy(raw_19),
                torch.from_numpy(tf),
                torch.tensor(label, dtype=torch.long))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SeizureTypeHead(nn.Module):
    def __init__(self, d_in=256, d_hidden=128, n_classes=3, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        h = self.drop(self.act(self.fc1(x)))
        return self.fc2(h), h


class FineTuneModel(nn.Module):
    def __init__(self, backbone_ckpt, n_classes, device):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.model_mvp_mixed import EEGFMMVPMixed

        ck = torch.load(str(backbone_ckpt), map_location="cpu")
        state = ck["model"] if "model" in ck else ck
        tuh_ch = state["raw_adapt_tuh.weight"].shape[1]
        common_ch = state["raw_adapt_tuh.weight"].shape[0]

        self.backbone = EEGFMMVPMixed(tuh_ch=tuh_ch, common_ch=common_ch)
        self.backbone.load_state_dict(state, strict=False)
        self.head = SeizureTypeHead(d_in=256, d_hidden=128, n_classes=n_classes)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for blk_idx in [4, 5]:
            for p in self.backbone.backbone.raw_enc.blocks[blk_idx].parameters():
                p.requires_grad_(True)
        tf_layers = self.backbone.backbone.tf_enc.encoder.layers
        for layer in tf_layers[-2:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in self.backbone.backbone.tf_enc.out_ln.parameters():
            p.requires_grad_(True)
        if self.backbone.backbone.tf_enc.pool is not None:
            for p in self.backbone.backbone.tf_enc.pool.parameters():
                p.requires_grad_(True)
        for p in self.backbone.backbone.fusion.parameters():
            p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    def forward(self, x_raw, x_tf):
        out = self.backbone(x_raw, x_tf, dataset="tuh")
        e_fused = out["e_fused"]
        logits, h = self.head(e_fused)
        return logits, h

    @torch.no_grad()
    def get_penultimate(self, x_raw, x_tf):
        out = self.backbone(x_raw, x_tf, dataset="tuh")
        e_fused = out["e_fused"]
        _, h = self.head(e_fused)
        return h


# ---------------------------------------------------------------------------
# Training with early stopping
# ---------------------------------------------------------------------------

def train_one_round(model, train_rows, val_rows, le, class_weights,
                    max_epochs, patience, batch_size, lr, device, ckpt_path):
    train_ds = TUSZWindowDataset(train_rows, le, augment=True)
    val_ds = TUSZWindowDataset(val_rows, le, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.05)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_acc = -1.0
    epochs_no_imp = 0
    epoch_curves = []  # store per-epoch metrics for plotting

    for epoch in range(max_epochs):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for x_raw, x_tf, labels in train_dl:
            x_raw, x_tf, labels = x_raw.to(device), x_tf.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_raw, x_tf)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
            loss_sum += loss.item() * len(labels)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_raw, x_tf, labels in val_dl:
                x_raw, x_tf, labels = x_raw.to(device), x_tf.to(device), labels.to(device)
                logits, _ = model(x_raw, x_tf)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        train_acc = correct / max(total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = loss_sum / max(total, 1)

        epoch_curves.append({
            "epoch": epoch + 1,
            "train_acc": round(float(train_acc), 4),
            "val_acc": round(float(val_acc), 4),
            "loss": round(float(avg_loss), 4),
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{max_epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.3f}  "
                  f"val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_imp = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": val_acc}, str(ckpt_path))
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"  Best val_acc = {best_val_acc:.3f}")
    return best_val_acc, epoch_curves


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, rows, le, batch_size, device):
    model.eval()
    ds = TUSZWindowDataset(rows, le, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs = []
    for x_raw, x_tf, _ in dl:
        x_raw, x_tf = x_raw.to(device), x_tf.to(device)
        embs.append(model.get_penultimate(x_raw, x_tf).cpu().numpy())
    return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Open-set scoring
# ---------------------------------------------------------------------------

def energy_scores(logits, T=1.0):
    return T * np.log(np.exp(logits / T).sum(axis=1))


def softmax_scores(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs.max(axis=1)


def auroc(pos_scores, neg_scores):
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([pos_scores, neg_scores])
    return float(roc_auc_score(y, s))


def ccr_at_fpr(known_scores, known_true_labels, known_pred_labels,
               unk_scores, fpr_target):
    """
    Correct Classification Rate at a controlled False Positive Rate.

    FPR = fraction of unknown windows accepted (score >= tau).
    CCR = fraction of known windows that are BOTH accepted AND correctly
          classified.

    tau is swept to find the value where FPR = fpr_target on unknowns.
    At that tau, CCR is computed on known windows.

    Args:
        known_scores      : (N_known,) energy scores for known windows
        known_true_labels : (N_known,) integer true class labels
        known_pred_labels : (N_known,) integer predicted class labels
        unk_scores        : (N_unk,)   energy scores for unknown windows
        fpr_target        : float, e.g. 0.05 / 0.10 / 0.20

    Returns:
        ccr   : float
        tau   : float (threshold used)
    """
    if len(unk_scores) == 0 or len(known_scores) == 0:
        return float("nan"), float("nan")

    # tau = percentile of unknown scores such that fpr_target fraction
    # of unknowns are ABOVE tau (incorrectly accepted)
    # i.e. tau = (1 - fpr_target) * 100th percentile of unk scores
    tau = float(np.percentile(unk_scores, (1.0 - fpr_target) * 100))

    # CCR = fraction of known windows accepted AND correctly classified
    accepted = known_scores >= tau
    correct = known_true_labels == known_pred_labels
    ccr = float((accepted & correct).sum() / max(len(known_scores), 1))

    return ccr, tau


def oscr_curve(known_scores, known_true_labels, known_pred_labels,
               unk_scores, n_points=200):
    """
    Compute the full OSCR curve: CCR vs FPR across all thresholds.
    Returns (fpr_vals, ccr_vals) for plotting.
    """
    all_scores = np.concatenate([known_scores, unk_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_points)

    fpr_vals, ccr_vals = [], []
    for tau in thresholds:
        fpr = float((unk_scores >= tau).mean())
        accepted = known_scores >= tau
        correct = known_true_labels == known_pred_labels
        ccr = float((accepted & correct).sum() / max(len(known_scores), 1))
        fpr_vals.append(fpr)
        ccr_vals.append(ccr)

    return fpr_vals, ccr_vals


# ---------------------------------------------------------------------------
# Evaluate one LOCO round
# ---------------------------------------------------------------------------

def evaluate_round(round_id, held_out_class, train_rows, loco_unk_rows,
                   true_unk_rows, backbone_ckpt, round_ckpt, out_dir,
                   max_epochs, patience, batch_size, lr, device, seed,
                   eval_only=False):
    train_classes = [c for c in ALL_KNOWN if c != held_out_class]
    print(f"\n{'=' * 60}")
    print(f"LOCO Round {round_id}: held-out = {held_out_class}")
    print(f"  Train classes : {train_classes}")
    print(f"  Train windows : {len(train_rows)}")
    print(f"  LOCO unk      : {len(loco_unk_rows)} ({held_out_class})")
    print(f"  True unk      : {len(true_unk_rows)}")
    print(f"{'=' * 60}")

    le = LabelEncoder()
    le.fit(train_classes)

    counts = Counter(r["seizure_type"] for r in train_rows)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(train_classes) * counts[t]) for t in le.classes_],
        dtype=torch.float32)
    print(f"  Class weights: { {t: f'{w:.2f}' for t, w in zip(le.classes_, weights.tolist())} }")

    epoch_curves = []

    if not eval_only:
        model = FineTuneModel(backbone_ckpt, n_classes=len(train_classes), device=device)
        model = model.to(device)

        rng = np.random.default_rng(seed)
        all_subjects = list({r["subject"] for r in train_rows})
        rng.shuffle(all_subjects)
        n_val_subj = max(1, int(0.2 * len(all_subjects)))
        val_subjects = set(all_subjects[:n_val_subj])

        internal_train = [r for r in train_rows if r["subject"] not in val_subjects]
        internal_val = [r for r in train_rows if r["subject"] in val_subjects]
        print(f"  Internal train/val: {len(internal_train)}/{len(internal_val)}")

        best_val_acc, epoch_curves = train_one_round(
            model, internal_train, internal_val, le, weights,
            max_epochs, patience, batch_size, lr, device, round_ckpt)

        best_state = torch.load(str(round_ckpt), map_location=device)
        model.load_state_dict(best_state["model"])
        print(f"  Loaded best (epoch {best_state['epoch'] + 1}, "
              f"val_acc={best_state['val_acc']:.3f})")
    else:
        # Load existing checkpoint for eval-only mode
        from models.model_mvp_mixed import EEGFMMVPMixed
        best_state = torch.load(str(round_ckpt), map_location=device)
        model = FineTuneModel(backbone_ckpt, n_classes=len(train_classes), device=device)
        model = model.to(device)
        model.load_state_dict(best_state["model"])
        best_val_acc = float(best_state["val_acc"])
        print(f"  Loaded existing checkpoint (epoch {best_state['epoch'] + 1}, "
              f"val_acc={best_val_acc:.3f})")

    # ── Extract embeddings ────────────────────────────────────────────────
    print("  Extracting embeddings...")
    train_emb = extract_embeddings(model, train_rows, le, batch_size, device)
    loco_unk_emb = extract_embeddings(model, loco_unk_rows, le, batch_size, device)
    true_unk_emb = extract_embeddings(model, true_unk_rows, le, batch_size, device) \
        if true_unk_rows else np.zeros((0, 128))

    # ── Linear probe + open-set scoring ──────────────────────────────────
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=seed)
    clf.fit(train_emb, le.transform([r["seizure_type"] for r in train_rows]))

    train_logits = clf.decision_function(train_emb)
    loco_unk_logits = clf.decision_function(loco_unk_emb)
    true_unk_logits = clf.decision_function(true_unk_emb) \
        if len(true_unk_emb) > 0 else np.zeros((0, len(train_classes)))

    train_energy = energy_scores(train_logits)
    loco_unk_energy = energy_scores(loco_unk_logits)
    true_unk_energy = energy_scores(true_unk_logits) \
        if len(true_unk_logits) > 0 else np.array([])

    # Tau at target FPR on training known windows
    tau = float(np.percentile(train_energy, TARGET_FPR * 100))

    # Predictions for CCR computation
    train_true_labels = le.transform([r["seizure_type"] for r in train_rows])
    train_pred_labels = clf.predict(train_emb)

    # ── Metrics ───────────────────────────────────────────────────────────
    res = {
        "round": round_id,
        "held_out_class": held_out_class,
        "train_classes": train_classes,
        "best_val_acc": float(best_val_acc),
        "tau": float(tau),
        "n_train": len(train_rows),
        "n_loco_unk": len(loco_unk_rows),
        "n_true_unk": len(true_unk_rows),
        "epoch_curves": epoch_curves,
    }

    # AUROC
    res["auroc_energy_loco"] = auroc(train_energy, loco_unk_energy)
    res["auroc_softmax_loco"] = auroc(softmax_scores(train_logits),
                                      softmax_scores(loco_unk_logits))
    res["auroc_energy_true_unk"] = auroc(train_energy, true_unk_energy) \
        if len(true_unk_energy) > 0 else float("nan")

    true_unk_types = np.array([r["seizure_type"] for r in true_unk_rows])
    for ut in TRUE_UNKNOWNS:
        ut_mask = true_unk_types == ut
        res[f"auroc_energy_{ut}"] = auroc(train_energy, true_unk_energy[ut_mask]) \
            if ut_mask.sum() > 0 else float("nan")

    # CCR@FPR for LOCO unknowns
    for fpr_t in FPR_OPERATING_POINTS:
        ccr, ccr_tau = ccr_at_fpr(
            train_energy, train_true_labels, train_pred_labels,
            loco_unk_energy, fpr_t)
        res[f"ccr_at_fpr{int(fpr_t * 100):02d}_loco"] = float(ccr)

    # CCR@FPR for true unknowns
    if len(true_unk_energy) > 0:
        for fpr_t in FPR_OPERATING_POINTS:
            ccr, _ = ccr_at_fpr(
                train_energy, train_true_labels, train_pred_labels,
                true_unk_energy, fpr_t)
            res[f"ccr_at_fpr{int(fpr_t * 100):02d}_true_unk"] = float(ccr)

    # OSCR curve (for plotting)
    if len(loco_unk_energy) > 0:
        fpr_v, ccr_v = oscr_curve(train_energy, train_true_labels,
                                  train_pred_labels, loco_unk_energy)
        res["oscr_loco"] = {"fpr": fpr_v, "ccr": ccr_v}

    # Rejection rates
    res["rejection_rate_known"] = float((train_energy < tau).mean())
    res["rejection_rate_loco_unk"] = float((loco_unk_energy < tau).mean())
    if len(true_unk_energy) > 0:
        res["rejection_rate_true_unk"] = float((true_unk_energy < tau).mean())
        for ut in TRUE_UNKNOWNS:
            ut_mask = true_unk_types == ut
            res[f"rejection_rate_{ut}"] = float((true_unk_energy[ut_mask] < tau).mean()) \
                if ut_mask.sum() > 0 else float("nan")

    # Closed-set F1 on non-rejected known
    accepted = train_energy >= tau
    if accepted.sum() > 0:
        f1 = float(f1_score(train_true_labels[accepted],
                            train_pred_labels[accepted],
                            average="weighted", zero_division=0))
        res["closed_set_f1"] = f1
        res["n_train_accepted"] = int(accepted.sum())
    else:
        res["closed_set_f1"] = float("nan")
        res["n_train_accepted"] = 0

    # Class assignment of non-rejected LOCO unknowns
    loco_accepted = loco_unk_energy >= tau
    if loco_accepted.sum() > 0:
        loco_pred = le.inverse_transform(clf.predict(loco_unk_emb[loco_accepted]))
        res["loco_unk_assignment"] = dict(Counter(loco_pred))
        res["n_loco_unk_accepted"] = int(loco_accepted.sum())
    else:
        res["loco_unk_assignment"] = {}
        res["n_loco_unk_accepted"] = 0

    # Class assignment of non-rejected true unknowns
    if len(true_unk_energy) > 0:
        true_accepted = true_unk_energy >= tau
        if true_accepted.sum() > 0:
            true_pred = le.inverse_transform(clf.predict(true_unk_emb[true_accepted]))
            true_types_acc = true_unk_types[true_accepted]
            assignment = defaultdict(Counter)
            for tt, pt in zip(true_types_acc, true_pred):
                assignment[tt][pt] += 1
            res["true_unk_assignment"] = {k: dict(v) for k, v in assignment.items()}
        else:
            res["true_unk_assignment"] = {}

    # Print round summary
    print(f"\n  --- Round {round_id} ({held_out_class} held out) ---")
    print(f"  AUROC energy (LOCO unk)          : {res['auroc_energy_loco']:.3f}")
    print(f"  AUROC energy (true unk)          : {res.get('auroc_energy_true_unk', float('nan')):.3f}")
    print(f"  Rejection known / LOCO / true    : "
          f"{res['rejection_rate_known']:.3f} / "
          f"{res['rejection_rate_loco_unk']:.3f} / "
          f"{res.get('rejection_rate_true_unk', float('nan')):.3f}")
    print(f"  Closed-set F1 (non-rejected)     : {res['closed_set_f1']:.3f}")
    print(f"  CCR@FPR (LOCO unk):")
    for fpr_t in FPR_OPERATING_POINTS:
        k = f"ccr_at_fpr{int(fpr_t * 100):02d}_loco"
        print(f"    FPR={fpr_t:.2f}: CCR={res.get(k, float('nan')):.3f}")
    print(f"  CCR@FPR (true unk):")
    for fpr_t in FPR_OPERATING_POINTS:
        k = f"ccr_at_fpr{int(fpr_t * 100):02d}_true_unk"
        print(f"    FPR={fpr_t:.2f}: CCR={res.get(k, float('nan')):.3f}")
    if res["loco_unk_assignment"]:
        total_acc = sum(res["loco_unk_assignment"].values())
        print(f"  LOCO unk ({held_out_class}) assigned to:")
        for pred_t, cnt in Counter(res["loco_unk_assignment"]).most_common():
            print(f"    -> {pred_t:<8}: {cnt:>4} ({100 * cnt / total_acc:.1f}%)")

    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load existing checkpoints")
    parser.add_argument("--ckpt_dir", type=Path, default=None,
                        help="Where to find existing checkpoints (eval_only mode)")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir else args.out_dir

    with open(args.index_csv, newline="") as f:
        all_rows = list(csv.DictReader(f))
    print(f"Loaded {len(all_rows)} windows")

    seizure_types = np.array([r["seizure_type"] for r in all_rows])
    roles = np.array([r["role"] for r in all_rows])
    contaminated = np.array([r["contaminated"].lower() == "true" for r in all_rows])

    true_unk_rows = [r for r, t in zip(all_rows, seizure_types)
                     if t in TRUE_UNKNOWNS]
    print(f"True unknowns: {len(true_unk_rows)} windows "
          f"({dict(Counter(r['seizure_type'] for r in true_unk_rows))})")

    all_results = []

    for round_id, held_out in enumerate(ALL_KNOWN):
        train_classes = [c for c in ALL_KNOWN if c != held_out]
        train_rows = [r for r, t, ro, co in
                      zip(all_rows, seizure_types, roles, contaminated)
                      if t in train_classes and ro == "known" and not co]
        loco_unk_rows = [r for r, t, ro, co in
                         zip(all_rows, seizure_types, roles, contaminated)
                         if t == held_out and ro == "known" and not co]
        round_ckpt = ckpt_dir / f"loco_round{round_id}_best.pt"

        res = evaluate_round(
            round_id=round_id,
            held_out_class=held_out,
            train_rows=train_rows,
            loco_unk_rows=loco_unk_rows,
            true_unk_rows=true_unk_rows,
            backbone_ckpt=args.ckpt,
            round_ckpt=round_ckpt,
            out_dir=args.out_dir,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            eval_only=args.eval_only,
        )
        all_results.append(res)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LOCO OPEN-SET SUMMARY")
    print("=" * 80)
    print(f"\n{'Round':<6} {'Held-out':<10} {'AUROC(LOCO)':<13} "
          f"{'AUROC(True)':<13} {'CCR@FPR5(LOCO)':<16} "
          f"{'CCR@FPR10(LOCO)':<17} {'F1'}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['round']:<6} {r['held_out_class']:<10} "
              f"{r['auroc_energy_loco']:<13.3f} "
              f"{r.get('auroc_energy_true_unk', float('nan')):<13.3f} "
              f"{r.get('ccr_at_fpr05_loco', float('nan')):<16.3f} "
              f"{r.get('ccr_at_fpr10_loco', float('nan')):<17.3f} "
              f"{r['closed_set_f1']:.3f}")
    print("-" * 80)

    loco_aurocs = [r["auroc_energy_loco"] for r in all_results]
    true_aurocs = [r.get("auroc_energy_true_unk", float("nan")) for r in all_results]
    ccr5_loco = [r.get("ccr_at_fpr05_loco", float("nan")) for r in all_results]
    ccr10_loco = [r.get("ccr_at_fpr10_loco", float("nan")) for r in all_results]

    print(f"\nMean AUROC  (LOCO)      : {np.mean(loco_aurocs):.3f} ± {np.std(loco_aurocs):.3f}")
    print(f"Mean AUROC  (true unk)  : {np.nanmean(true_aurocs):.3f} ± {np.nanstd(true_aurocs):.3f}")
    print(f"Mean CCR@FPR5  (LOCO)   : {np.nanmean(ccr5_loco):.3f} ± {np.nanstd(ccr5_loco):.3f}")
    print(f"Mean CCR@FPR10 (LOCO)   : {np.nanmean(ccr10_loco):.3f} ± {np.nanstd(ccr10_loco):.3f}")

    print("\n--- LOCO class assignments ---")
    for r in all_results:
        ho = r["held_out_class"]
        asgn = r.get("loco_unk_assignment", {})
        n_acc = r.get("n_loco_unk_accepted", 0)
        n_tot = r["n_loco_unk"]
        if asgn:
            top = Counter(asgn).most_common(2)
            top_str = ", ".join(f"{t}({c})" for t, c in top)
            print(f"  {ho:<8}: {n_acc}/{n_tot} non-rejected -> {top_str}")
        else:
            print(f"  {ho:<8}: 0/{n_tot} (all rejected)")
    print("=" * 80)

    # ── Save results (JSON-safe) ──────────────────────────────────────────
    class _NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray):  return o.tolist()
            if isinstance(o, Path):        return str(o)
            return super().default(o)

    # Serialize args safely
    args_dict = {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()}

    out_path = args.out_dir / "loco_results.json"
    with open(out_path, "w") as f:
        json.dump({"rounds": all_results, "config": args_dict},
                  f, indent=2, cls=_NE)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
