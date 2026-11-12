#!/usr/bin/env python3
"""
partition_counts.py
===================
Subject and window counts per known class in the fine-tune / val / test
partition, averaged over seeds. Prints LaTeX rows for tab:partition.

Usage (from the openset output folder on VAST):
  python partition_counts.py embed_index.csv \
      camera_ready_s42/split_seed42.json \
      camera_ready_s43/split_seed43.json \
      camera_ready_s44/split_seed44.json
"""
import csv
import json
import sys
from collections import defaultdict

KNOWN = ["fnsz", "gnsz", "cpsz", "tcsz"]
PARTS = ["train", "val", "test"]


def main():
    idx = list(csv.DictReader(open(sys.argv[1], newline="")))
    split_paths = sys.argv[2:]
    acc = defaultdict(list)

    for sp in split_paths:
        part = json.load(open(sp))
        subj = defaultdict(set)
        win = defaultdict(int)
        for r in idx:
            if (r["seizure_type"] in KNOWN and r["role"] == "known"
                    and r["contaminated"].lower() != "true"):
                p = part.get(r["subject"])
                if p:
                    subj[(r["seizure_type"], p)].add(r["subject"])
                    win[(r["seizure_type"], p)] += 1
        print(f"\n{sp}")
        for c in KNOWN:
            print(f"  {c:<5}" + "  ".join(
                f"{p}: {len(subj[(c, p)])} subj / {win[(c, p)]} win" for p in PARTS))
            for p in PARTS:
                acc[(c, p, "s")].append(len(subj[(c, p)]))
                acc[(c, p, "w")].append(win[(c, p)])

    n = len(split_paths)
    print(f"\nLaTeX rows (mean over {n} seeds):")
    for c in KNOWN:
        row = [f"{sum(acc[(c, p, t)]) / n:.0f}" for p in PARTS for t in "sw"]
        print(f"{c} & " + " & ".join(row) + r" \\")


if __name__ == "__main__":
    main()
