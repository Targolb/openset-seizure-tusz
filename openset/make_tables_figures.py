#!/usr/bin/env python3
"""
make_tables_figures.py
======================
Reproduces the paper's result tables and Figure 1 from the per-seed
unified_results.json files written by unified_openset_eval.py.

  * Closed-set accuracy per LOCO round            (Table: closed-set)
  * Open-set metrics, 7 scorers, true unknowns     (Table: open-set, top)
    and LOCO unknowns from unseen subjects         (Table: open-set, bottom)
  * Subject-confound table: seen vs unseen subjects (Table: seen/unseen)
  * Per-subject COV + Wilcoxon, GHOST vs Energy     (Table: fairness)
  * LOCO OSCR curves (log FPR)                       (Figure 1)

Usage
-----
  python make_tables_figures.py \
      --results results/seed42/unified_results.json \
                results/seed43/unified_results.json \
                results/seed44/unified_results.json \
      --out_dir results/paper
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

SCORERS = ["Energy", "MSP", "MaxLogit", "Mahalanobis", "kNN", "GHOST", "COSTARR"]
HELD = ["fnsz", "gnsz", "cpsz", "tcsz"]
METRICS = ["auroc", "auoscr", "ccr@5", "ccr@10", "ccr@20", "f@c95"]
MIN_SUBJ_WINDOWS = 5
FIG_SCORERS = {"Energy": ("#c0392b", "-"), "MSP": ("#7f8c8d", (0, (1, 1))),
               "Mahalanobis": ("#8e44ad", "-."), "GHOST": ("#27ae60", "--"),
               "COSTARR": ("#2471a3", ":")}


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std():.3f}"


def closed_set(runs, out):
    out.append("## Closed-set accuracy on known test subjects (mean ± std over seeds)")
    for rid, h in enumerate(HELD):
        rr = [r for r in runs if r["round"] == rid]
        out.append(f"{h:6} {ms([r['closed_set_acc_test'] for r in rr])}  (n_seeds={len(rr)})")


def openset_table(runs, tag, out):
    out.append(f"\n## Open-set metrics: {tag} (round average per seed, then mean ± std over seeds)")
    out.append(f"{'scorer':12} " + " ".join(f"{m:>15}" for m in METRICS))
    for s in SCORERS:
        cells = []
        for m in METRICS:
            per_seed = defaultdict(list)
            for r in runs:
                if tag in r["scorers"][s]:
                    per_seed[r["seed"]].append(r["scorers"][s][tag][m])
            cells.append(ms([np.mean(v) for v in per_seed.values()]))
        out.append(f"{s:12} " + " ".join(f"{c:>15}" for c in cells))


def seen_unseen(runs, out):
    out.append("\n## Subject confound: LOCO windows seen/unseen in fine-tuning (mean over seeds)")
    out.append("held   all_win  seen_win  seen_subj  AUROC_unseen(E/G)  AUROC_seen(E/G)")
    for rid, h in enumerate(HELD):
        rr = [r for r in runs if r["round"] == rid]
        g = lambda tag, s: np.mean([r["scorers"][s][tag]["auroc"] for r in rr
                                    if tag in r["scorers"][s]])
        out.append(f"{h:6} {rr[0]['n']['loco_all']:7d} "
                   f"{np.mean([r['n']['loco_seen_in_ft'] for r in rr]):9.0f} "
                   f"{np.mean([r['n']['subjects_loco_seen'] for r in rr]):10.1f}  "
                   f"{g('loco_unseen', 'Energy'):.2f}/{g('loco_unseen', 'GHOST'):.2f}"
                   f"            {g('loco_seen', 'Energy'):.2f}/{g('loco_seen', 'GHOST'):.2f}")


def fairness(runs, out):
    out.append("\n## Per-subject fairness on LOCO unseen-subject windows "
               "(tau = 10th pct of known-test scores; subjects with >=5 windows)")
    out.append("COV mean ± std over seeds; Wilcoxon: two-sided signed-rank, "
               "per-subject rejection rate averaged over seeds, GHOST vs Energy")
    for rid, h in enumerate(HELD):
        rr = [r for r in runs if r["round"] == rid]
        covs = defaultdict(list)
        subj_tpr = defaultdict(lambda: defaultdict(list))
        for r in rr:
            meta = r["raw"]["_meta"]
            subj = np.array(meta["loco_subj"])
            seen = np.array(meta["loco_seen"])
            for s in SCORERS:
                kn = np.array(r["raw"][s]["known_test"])
                lo = np.array(r["raw"][s]["loco_all"])
                tau = np.percentile(kn, 10)
                tprs = {}
                for u in np.unique(subj[~seen]):
                    m = (subj == u) & (~seen)
                    if m.sum() >= MIN_SUBJ_WINDOWS:
                        tprs[u] = float((lo[m] < tau).mean())
                v = np.array(list(tprs.values()))
                covs[s].append(v.std() / max(v.mean(), 1e-8) if len(v) else np.nan)
                for u, t in tprs.items():
                    subj_tpr[s][u].append(t)
        e = {u: np.mean(v) for u, v in subj_tpr["Energy"].items()}
        g = {u: np.mean(v) for u, v in subj_tpr["GHOST"].items()}
        us = sorted(set(e) & set(g))
        try:
            p = wilcoxon([g[u] for u in us], [e[u] for u in us],
                         alternative="two-sided").pvalue
        except ValueError:
            p = float("nan")
        out.append(f"{h:6} n_subj={len(us):3d}  Energy COV {ms(covs['Energy'])}  "
                   f"GHOST COV {ms(covs['GHOST'])}  p={p:.3g}")


def oscr_figure(runs, out_dir):
    def curve(kn, ok, unk):
        th = np.quantile(np.r_[kn, unk], np.linspace(0, 1, 400))[::-1]
        return (np.array([(unk >= t).mean() for t in th]),
                np.array([((kn >= t) & ok).mean() for t in th]))

    grid = np.logspace(-3, 0, 200)
    fig, axs = plt.subplots(2, 2, figsize=(7, 6.2), sharex=True, sharey=True)
    for ax, (rid, h) in zip(axs.flat, enumerate(HELD)):
        rr = [r for r in runs if r["round"] == rid]
        for s, (c, ls) in FIG_SCORERS.items():
            curves, aus = [], []
            for r in rr:
                m = r["raw"]["_meta"]
                unseen = ~np.array(m["loco_seen"])
                kn = np.array(r["raw"][s]["known_test"])
                ok = np.array(m["known_test_correct"])
                unk = np.array(r["raw"][s]["loco_all"])[unseen]
                f, cc = curve(kn, ok, unk)
                o = np.argsort(f, kind="stable")
                curves.append(np.interp(grid, f[o], cc[o]))
                aus.append(r["scorers"][s]["loco_unseen"]["auoscr"])
            ax.plot(grid, np.mean(curves, 0), color=c, ls=ls, lw=1.6,
                    label=f"{s} ({np.mean(aus):.2f})")
        ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1)
        ax.axvline(0.05, color="gray", lw=.7, ls="--"); ax.grid(alpha=.3)
        ax.set_title(f"Held-out: {h}" + (f" ({len(rr)} seeds)" if len(rr) < 3 else ""),
                     fontsize=10)
        ax.legend(fontsize=6.5, loc="upper left", title="AUOSCR", title_fontsize=6.5)
    for ax in axs[1]:
        ax.set_xlabel("FPR (held-out class, unseen subjects)", fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("CCR (known test subjects)", fontsize=9)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"loco_oscr.{ext}", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for p in a.results:
        runs += json.load(open(p))["runs"]
    out = [f"Seeds: {sorted({r['seed'] for r in runs})}, runs: {len(runs)}\n"]
    closed_set(runs, out)
    openset_table(runs, "true", out)
    openset_table(runs, "loco_unseen", out)
    seen_unseen(runs, out)
    fairness(runs, out)
    text = "\n".join(out)
    print(text)
    (a.out_dir / "paper_tables.txt").write_text(text)
    oscr_figure(runs, a.out_dir)
    print(f"\nSaved: {a.out_dir/'paper_tables.txt'}, {a.out_dir/'loco_oscr.pdf'}")


if __name__ == "__main__":
    main()
