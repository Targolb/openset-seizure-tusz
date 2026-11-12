#!/usr/bin/env python3
"""
check_pretrain_overlap.py
=========================
Checks how many TUSZ open-set subjects (and sessions) also appear in the
TUH EEG windows used to pretrain week3_mixed_tuh_seed.

Pretraining (models/pretrain_week3_mixed_tuh_seed.py) globbed every window under
  TUH_ROOT = /scratch0/Targol/eegfm_outputs/TUH/tuh_windows_out/TUH_EEG
whose layout is  TUH_EEG/<subject>/<session>/<recording>/<montage>/win_xxxxxx/
(written by preprocessing/window_extractor_tuh.py on the first 10,000 EDFs of
/NCL_VAST_Datasets/tuh_eeg_full/edf). TUSZ uses the same TUEG patient IDs,
so overlap is checked by subject ID and by (subject, session) pair.

Usage:
  python check_pretrain_overlap.py \
      --index_csv /NCL_VAST_Datasets/Targol/eegfm_outputs/TUH/openset/embed_index.csv \
      --tuh_root  /scratch0/Targol/eegfm_outputs/TUH/tuh_windows_out/TUH_EEG \
      --out_json  /NCL_VAST_Datasets/Targol/eegfm_outputs/TUH/openset/pretrain_overlap.json
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

KNOWN = ["fnsz", "gnsz", "cpsz", "tcsz"]
TRUE_UNK = ["absz", "tnsz", "spsz", "mysz"]
SESS_RE = re.compile(r"(s\d{3}_\d{4})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--tuh_root", required=True)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()

    # --- pretraining subjects / sessions (directory names only, fast) ---
    pre_subj = set()
    pre_sess = set()
    for subj in os.listdir(a.tuh_root):
        p = os.path.join(a.tuh_root, subj)
        if not os.path.isdir(p):
            continue
        pre_subj.add(subj)
        for sess in os.listdir(p):
            m = SESS_RE.search(sess)
            if m:
                pre_sess.add((subj, m.group(1)))
    print(f"Pretraining TUH: {len(pre_subj)} subjects, {len(pre_sess)} sessions")

    # --- TUSZ open-set subjects / sessions ---
    rows = list(csv.DictReader(open(a.index_csv, newline="")))
    cols = rows[0].keys()
    print(f"Index columns: {list(cols)}")
    subj_cls = defaultdict(set)
    sess_cls = defaultdict(set)
    win_cls = defaultdict(int)
    win_cls_overlap_subj = defaultdict(int)
    win_cls_overlap_sess = defaultdict(int)
    for r in rows:
        if r.get("contaminated", "").lower() == "true":
            continue
        c = r["seizure_type"]
        s = r["subject"]
        blob = " ".join(str(r.get(k, "")) for k in cols)
        m = SESS_RE.search(blob)
        sess = (s, m.group(1)) if m else None
        subj_cls[c].add(s)
        if sess:
            sess_cls[c].add(sess)
        win_cls[c] += 1
        if s in pre_subj:
            win_cls_overlap_subj[c] += 1
        if sess and sess in pre_sess:
            win_cls_overlap_sess[c] += 1

    out = {"pretrain_subjects": len(pre_subj), "pretrain_sessions": len(pre_sess),
           "per_class": {}}
    print(f"\n{'class':<6}{'role':<9}{'subj':>6}{'subj in PT':>12}"
          f"{'sess':>7}{'sess in PT':>12}{'win':>7}{'win subj-in-PT':>16}")
    for c in KNOWN + TRUE_UNK:
        if c not in subj_cls:
            continue
        S = subj_cls[c]
        SS = sess_cls[c]
        so = len(S & pre_subj)
        sso = len(SS & pre_sess)
        role = "known" if c in KNOWN else "unknown"
        print(f"{c:<6}{role:<9}{len(S):>6}{so:>12}{len(SS):>7}{sso:>12}"
              f"{win_cls[c]:>7}{win_cls_overlap_subj[c]:>16}")
        out["per_class"][c] = {
            "role": role, "subjects": len(S), "subjects_in_pretrain": so,
            "sessions": len(SS), "sessions_in_pretrain": sso,
            "windows": win_cls[c],
            "windows_subject_in_pretrain": win_cls_overlap_subj[c],
            "windows_session_in_pretrain": win_cls_overlap_sess[c],
            "overlapping_subjects": sorted(S & pre_subj)}

    all_s = set().union(*subj_cls.values())
    unk_s = set().union(*(subj_cls[c] for c in TRUE_UNK if c in subj_cls))
    all_ss = set().union(*sess_cls.values()) if sess_cls else set()
    print(f"\nAll open-set subjects: {len(all_s)}, in pretraining: {len(all_s & pre_subj)}")
    print(f"True-unknown subjects: {len(unk_s)}, in pretraining: {len(unk_s & pre_subj)}")
    print(f"All open-set sessions: {len(all_ss)}, in pretraining: {len(all_ss & pre_sess)}")
    if not all_ss:
        print("[WARN] No session IDs parsed from the index; only subject-level overlap reported.")
    out["summary"] = {"subjects": len(all_s), "subjects_in_pretrain": len(all_s & pre_subj),
                      "true_unk_subjects": len(unk_s),
                      "true_unk_subjects_in_pretrain": len(unk_s & pre_subj),
                      "sessions": len(all_ss), "sessions_in_pretrain": len(all_ss & pre_sess)}
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2)
        print(f"Saved: {a.out_json}")


if __name__ == "__main__":
    main()
