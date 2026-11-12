#!/usr/bin/env python3
"""
unified_openset_eval.py  (camera-ready)
=======================================
One script -> one JSON that every number/figure in the paper is generated from.

Fixes vs. the submitted version
  * SUBJECT-DISJOINT partition: per seed, subjects are split 60/20/20 into
    train / val / test (a subject lives in exactly one part).
  * tau is calibrated on VAL-subject known windows (never on training windows).
  * Known scores are computed on TEST-subject known windows (not training windows).
  * LOCO unknown = held-out class windows. Reported on (a) test subjects only
    (subject-disjoint, primary) and (b) all subjects, split by whether the
    subject was seen in fine-tuning (subject-novelty control).
  * True unknowns (absz/tnsz/spsz/mysz) are evaluated against the same known-test set.
  * 3 seeds; scorers: Energy, MSP, MaxLogit, Mahalanobis, kNN, GHOST, COSTARR.
  * Metrics: AUROC, AUOSCR, CCR@FPR{5,10,20}, F@C95, TPR@tau, per-subject
    TPR@FPR10 + COV. Raw scores + subject ids saved for plots / Wilcoxon.

Usage
-----
  python unified_openset_eval.py \
    --index_csv /NCL_VAST_Datasets/Targol/eegfm_outputs/TUH/openset/embed_index.csv \
    --backbone  /NCL_VAST_Datasets/Targol/eegfm_checkpoints/week3_mixed_tuh_seed/week3_mixed_step20000.pt \
    --out_dir   /NCL_VAST_Datasets/Targol/eegfm_outputs/TUH/openset/camera_ready \
    --seeds 42 43 44 --batch_size 16 --device cuda
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from loco_openset_v2 import (TUSZWindowDataset, FineTuneModel,  # noqa: E402
                             train_one_round)

ALL_KNOWN = ["fnsz", "gnsz", "cpsz", "tcsz"]
TRUE_UNK = ["absz", "tnsz", "spsz", "mysz"]
MIN_SUBJ_WINDOWS = 5
SCORERS = ["Energy", "MSP", "MaxLogit", "Mahalanobis", "kNN", "GHOST", "COSTARR"]


# ----------------------------------------------------------------- partition
def subject_partition(rows, seed, frac=(0.6, 0.2, 0.2)):
    """Subject-disjoint train/val/test, stratified per class (rarest first)."""
    elig = [r for r in rows if r["role"] == "known"
            and r["contaminated"].lower() != "true"]
    by_cls = defaultdict(set)
    for r in elig:
        by_cls[r["seizure_type"]].add(r["subject"])
    rng = np.random.default_rng(seed)
    assign = {}
    for cls in sorted(by_cls, key=lambda c: len(by_cls[c])):
        subs = sorted(s for s in by_cls[cls] if s not in assign)
        rng.shuffle(subs)
        n = len(subs)
        n_te = max(1, round(frac[2] * n)) if n >= 3 else 0
        n_va = max(1, round(frac[1] * n)) if n >= 3 else 0
        for i, s in enumerate(subs):
            assign[s] = "test" if i < n_te else ("val" if i < n_te + n_va else "train")
    return assign


# ------------------------------------------------------------------ features
@torch.no_grad()
def feats_logits(model, rows, le, bs, device):
    if not rows:
        return np.zeros((0, 128), np.float32), np.zeros((0, len(le.classes_)), np.float32)
    model.eval()
    dl = DataLoader(TUSZWindowDataset(rows, le, augment=False),
                    batch_size=bs, shuffle=False, num_workers=0)
    F, L = [], []
    for xr, xt, _ in dl:
        lg, h = model(xr.to(device), xt.to(device))
        F.append(h.cpu().numpy()); L.append(lg.cpu().numpy())
    return np.concatenate(F), np.concatenate(L)


# ------------------------------------------------------------------- scorers
class Scorers:
    """Fit on training features; score(F, logits) -> dict name -> (N,) (high=known)."""
    def __init__(self, trF, trL, trY, W):
        self.K = trL.shape[1]
        ok = trL.argmax(1) == trY
        self.W = W
        self.mu, self.sd, self.muC = {}, {}, {}
        for k in range(self.K):
            m = ok & (trY == k)
            if m.sum() < 2:
                m = trY == k
            Fk = trF[m].astype(np.float64)
            self.mu[k] = Fk.mean(0)
            self.sd[k] = Fk.std(0) + 1e-6
            Ck = np.concatenate([Fk, Fk * W[k][None]], 1)
            self.muC[k] = Ck.mean(0)
        # shared-covariance Mahalanobis
        cen = np.concatenate([trF[trY == k] - self.mu[k] for k in range(self.K)]).astype(np.float64)
        cov = cen.T @ cen / max(len(cen), 1) + 1e-3 * np.eye(trF.shape[1])
        self.P = np.linalg.inv(cov)
        # kNN bank (cosine)
        self.bank = trF / (np.linalg.norm(trF, axis=1, keepdims=True) + 1e-8)
        cl = trL[ok]
        self.lmin, self.lmax = float(cl.min()), float(cl.max())

    def score(self, F, L):
        N = len(F)
        if N == 0:
            return {s: np.array([]) for s in SCORERS}
        out = {}
        out["Energy"] = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
        e = np.exp(L - L.max(1, keepdims=True))
        out["MSP"] = (e / e.sum(1, keepdims=True)).max(1)
        out["MaxLogit"] = L.max(1)
        kh = L.argmax(1)
        F64 = F.astype(np.float64)
        # Mahalanobis: -min_k d^2
        d = np.stack([np.einsum("nd,de,ne->n", F64 - self.mu[k], self.P, F64 - self.mu[k])
                      for k in range(self.K)], 1)
        out["Mahalanobis"] = -d.min(1)
        # kNN: -(distance to 10th nearest train neighbour)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        knn = np.zeros(N, np.float32)
        for i in range(0, N, 512):
            sim = Fn[i:i + 512] @ self.bank.T
            kth = np.partition(-sim, 9, axis=1)[:, 9]
            knn[i:i + 512] = kth          # = -sim_k ; high = close
        out["kNN"] = -(-knn) * 0 + (-(-knn))  # placeholder replaced below
        out["kNN"] = knn * -1 * -1  # keep dtype
        out["kNN"] = -knn * -1  # sim of 10th neighbour is -kth
        out["kNN"] = -knn  # high similarity -> known  (kth = -sim)  => score = sim
        out["kNN"] = -knn if False else (-knn) * -1 * -1
        # (explicit, unambiguous version:)
        sims = []
        for i in range(0, N, 512):
            s = Fn[i:i + 512] @ self.bank.T
            sims.append(np.sort(s, axis=1)[:, -10])
        out["kNN"] = np.concatenate(sims)
        # GHOST
        zk = L[np.arange(N), kh]
        g = np.array([zk[i] / (np.sum(np.abs(F64[i] - self.mu[kh[i]]) / self.sd[kh[i]]) + 1e-8)
                      for i in range(N)], np.float32)
        out["GHOST"] = g
        # COSTARR
        gnl = np.clip((zk - self.lmin) / max(self.lmax - self.lmin, 1e-8), 0, 1)
        c = np.zeros(N, np.float32)
        for i in range(N):
            k = kh[i]
            Ci = np.concatenate([F[i], F[i] * self.W[k]])
            mu = self.muC[k]
            sim = 0.5 * (1 + Ci @ mu / (np.linalg.norm(Ci) * np.linalg.norm(mu) + 1e-8))
            c[i] = gnl[i] * sim
        out["COSTARR"] = c
        return out


# ------------------------------------------------------------------- metrics
def auroc(kn, unk):
    if len(kn) == 0 or len(unk) == 0:
        return float("nan")
    y = np.r_[np.ones(len(kn)), np.zeros(len(unk))]
    return float(roc_auc_score(y, np.r_[kn, unk]))


def oscr_points(kn, kn_ok, unk):
    """Exact OSCR sweep over all thresholds. returns fpr (sorted asc), ccr."""
    th = np.unique(np.r_[kn, unk])[::-1]
    th = np.r_[th[0] + 1e-9, th]
    fpr = np.array([(unk >= t).mean() for t in th])
    ccr = np.array([((kn >= t) & kn_ok).sum() / len(kn) for t in th])
    return fpr, ccr


def auoscr(fpr, ccr):
    o = np.argsort(fpr, kind="stable")
    return float(np.trapz(ccr[o], fpr[o]))


def ccr_at_fpr(fpr, ccr, target):
    m = fpr <= target
    return float(ccr[m].max()) if m.any() else 0.0


def f_at_c95(fpr, ccr, closed_acc):
    m = ccr >= 0.95 * closed_acc
    return float(fpr[m].min()) if m.any() else 1.0


def per_subject_tpr(subjects, scores, tau):
    subjects = np.asarray(subjects)
    out = {}
    for s in np.unique(subjects):
        m = subjects == s
        if m.sum() >= MIN_SUBJ_WINDOWS:
            out[str(s)] = float((scores[m] < tau).mean())
    return out


def cov(d):
    v = np.array(list(d.values()))
    return float(v.std() / max(v.mean(), 1e-8)) if len(v) else float("nan")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=Path)
    ap.add_argument("--backbone", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse_ckpt", action="store_true",
                    help="skip training if checkpoint for (seed,round) exists")
    a = ap.parse_args()
    dev = torch.device(a.device)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "ckpt").mkdir(exist_ok=True)

    rows = list(csv.DictReader(open(a.index_csv, newline="")))
    true_unk_rows = [r for r in rows if r["seizure_type"] in TRUE_UNK]
    results = {"config": vars(a) | {"index_csv": str(a.index_csv),
                                    "backbone": str(a.backbone),
                                    "out_dir": str(a.out_dir)},
               "runs": []}

    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        part = subject_partition(rows, seed)
        json.dump(part, open(a.out_dir / f"split_seed{seed}.json", "w"), indent=1)

        def sel(types, which):
            return [r for r in rows if r["seizure_type"] in types
                    and r["role"] == "known" and r["contaminated"].lower() != "true"
                    and part.get(r["subject"]) in which]

        for rid, held in enumerate(ALL_KNOWN):
            tcls = [c for c in ALL_KNOWN if c != held]
            le = LabelEncoder().fit(tcls)
            tr = sel(tcls, {"train"})
            va = sel(tcls, {"val"})
            te = sel(tcls, {"test"})
            lo_all = sel([held], {"train", "val", "test"})
            lo_test = [r for r in lo_all if part[r["subject"]] == "test"]
            train_subj = {r["subject"] for r in sel(tcls, {"train"})}
            seen_flag = np.array([r["subject"] in train_subj for r in lo_all])
            print(f"\n=== seed {seed} round {rid} held={held}: train {len(tr)} "
                  f"val {len(va)} test {len(te)} | loco(all) {len(lo_all)} "
                  f"loco(test-subj) {len(lo_test)} seen-in-FT {int(seen_flag.sum())}")

            cnt = Counter(r["seizure_type"] for r in tr)
            w = torch.tensor([len(tr) / (len(tcls) * cnt[t]) for t in le.classes_],
                             dtype=torch.float32)
            ck = a.out_dir / "ckpt" / f"s{seed}_r{rid}.pt"
            model = FineTuneModel(a.backbone, n_classes=len(tcls), device=dev).to(dev)
            if a.reuse_ckpt and ck.exists():
                model.load_state_dict(torch.load(ck, map_location=dev)["model"])
            else:
                train_one_round(model, tr, va, le, w, a.max_epochs, a.patience,
                                a.batch_size, a.lr, dev, ck)
                model.load_state_dict(torch.load(ck, map_location=dev)["model"])
            model.eval()
            W = model.head.fc2.weight.detach().cpu().numpy()

            trF, trL = feats_logits(model, tr, le, a.batch_size, dev)
            vaF, vaL = feats_logits(model, va, le, a.batch_size, dev)
            teF, teL = feats_logits(model, te, le, a.batch_size, dev)
            loF, loL = feats_logits(model, lo_all, le, a.batch_size, dev)
            tuF, tuL = feats_logits(model, true_unk_rows, le, a.batch_size, dev)
            trY = np.array([le.transform([r["seizure_type"]])[0] for r in tr])
            teY = np.array([le.transform([r["seizure_type"]])[0] for r in te])
            closed_acc = float((teL.argmax(1) == teY).mean())
            te_ok = teL.argmax(1) == teY

            S = Scorers(trF, trL, trY, W)
            sc_va, sc_te = S.score(vaF, vaL), S.score(teF, teL)
            sc_lo, sc_tu = S.score(loF, loL), S.score(tuF, tuL)

            lo_subj = np.array([r["subject"] for r in lo_all])
            tu_subj = np.array([r["subject"] for r in true_unk_rows])
            te_subj = np.array([r["subject"] for r in te])
            is_te_lo = np.array([part[s] == "test" for s in lo_subj])

            rd = {"seed": seed, "round": rid, "held_out": held,
                  "n": {"train": len(tr), "val": len(va), "known_test": len(te),
                        "loco_all": len(lo_all), "loco_test_subj": int(is_te_lo.sum()),
                        "loco_seen_in_ft": int(seen_flag.sum()),
                        "true_unk": len(true_unk_rows),
                        "subjects_known_test": len(set(te_subj)),
                        "subjects_loco_all": len(set(lo_subj)),
                        "subjects_loco_seen": len(set(lo_subj[seen_flag])),
                        "subjects_true_unk": len(set(tu_subj))},
                  "closed_set_acc_test": closed_acc, "scorers": {}, "raw": {}}

            for name in SCORERS:
                tau = float(np.percentile(sc_va[name], 5))      # 95% of VAL-known accepted
                entry = {"tau": tau,
                         "known_acc_at_tau": float((sc_te[name] >= tau).mean())}
                for tag, uscore in [("loco_test", sc_lo[name][is_te_lo]),
                                    ("loco_all", sc_lo[name]),
                                    ("loco_seen", sc_lo[name][seen_flag]),
                                    ("loco_unseen", sc_lo[name][~seen_flag]),
                                    ("true", sc_tu[name])]:
                    if len(uscore) == 0:
                        continue
                    fpr, ccr = oscr_points(sc_te[name], te_ok, uscore)
                    entry[tag] = {
                        "auroc": auroc(sc_te[name], uscore),
                        "auoscr": auoscr(fpr, ccr),
                        "ccr@5": ccr_at_fpr(fpr, ccr, .05),
                        "ccr@10": ccr_at_fpr(fpr, ccr, .10),
                        "ccr@20": ccr_at_fpr(fpr, ccr, .20),
                        "f@c95": f_at_c95(fpr, ccr, closed_acc),
                        "tpr@tau": float((uscore < tau).mean()),
                        "n_unk": int(len(uscore))}
                # per-subject fairness on subject-disjoint LOCO unknowns, tau at FPR=10% on known-test
                tau10 = float(np.percentile(sc_te[name], 10))
                for tag, subj, usc in [("loco_test", lo_subj[is_te_lo], sc_lo[name][is_te_lo]),
                                       ("true", tu_subj, sc_tu[name])]:
                    if len(usc):
                        d = per_subject_tpr(subj, usc, tau10)
                        entry[f"fair_{tag}"] = {"cov": cov(d), "n_subj": len(d),
                                                "mean_tpr": float(np.mean(list(d.values()))) if d else None,
                                                "per_subject_tpr": d}
                rd["scorers"][name] = entry
                rd["raw"][name] = {"known_test": sc_te[name].tolist(),
                                   "loco_all": sc_lo[name].tolist(),
                                   "true": sc_tu[name].tolist()}
            rd["raw"]["_meta"] = {"known_test_correct": te_ok.tolist(),
                                  "known_test_subj": te_subj.tolist(),
                                  "loco_subj": lo_subj.tolist(),
                                  "loco_seen": seen_flag.tolist(),
                                  "loco_is_test_subj": is_te_lo.tolist(),
                                  "true_subj": tu_subj.tolist()}
            results["runs"].append(rd)
            print(f"  closed acc {closed_acc:.3f} | AUROC loco_test / true:")
            for name in SCORERS:
                e = rd["scorers"][name]
                print(f"   {name:<12} {e.get('loco_test', {}).get('auroc', float('nan')):.3f} "
                      f"/ {e.get('true', {}).get('auroc', float('nan')):.3f}")
            # checkpoint JSON progressively
            json.dump(results, open(a.out_dir / "unified_results_43.json", "w"))
            del model; torch.cuda.empty_cache()

    print("\nDONE ->", a.out_dir / "unified_results_43.json")


if __name__ == "__main__":
    main()
