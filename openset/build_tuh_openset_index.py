#!/usr/bin/env python3
"""
build_tuh_openset_index.py
==========================
Builds the TUH seizure-type window index for open-set learning.

Reads TUSZ v2.0.3 multi-class CSV annotations (.csv files, per-channel,
"csv_v1.0.0" format), aggregates to a single term-level seizure-type label
per time segment (majority vote across channels), derives the 23-channel
CHB-MIT-compatible bipolar montage from TUSZ's -LE referential channels,
segments into non-overlapping 15-second windows at 256 Hz, assigns
seizure-type labels, and outputs an index CSV ready for the EFM open-set
pipeline.

Based on your report (WACV2027.pdf):
  Known classes  (train + test) : fnsz, gnsz, cpsz, tcsz  [>=12 subjects]
  Unknown classes (test only)   : absz, tnsz, spsz, mysz
  Contaminated subjects         : absz & tnsz patients who also appear
                                  in the known-class pool -> removed from
                                  training (see Table 7 in your report)

Dataset layout (confirmed on hulkbuster):
  /NCL_VAST_Datasets/tuh/TUSZ/edf/{train,dev,eval}/<subject>/<session>/<montage>/<rec>.edf
                                                                                  <rec>.csv      <- multi-class, per-channel
                                                                                  <rec>.csv_bi   <- binary, term-level (seiz/bckg)

TUSZ EDF channels (41 total, -LE referential montage, 250 Hz):
  ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE',
   'EEG A1-LE', 'EEG A2-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
   'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE',
   'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE', 'EEG OZ-LE', 'EEG PG1-LE', 'EEG PG2-LE',
   'EEG EKG-LE', 'EEG 26-LE'..'EEG 32-LE', 'PHOTIC PH', 'DC1-DC'..'DC8-DC']

CHB-MIT 23-channel target montage (from your meta.json selected_channels):
  ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
   'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
   'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8']

  Old<->new 10-20 nomenclature: T7=T3, T8=T4, P7=T5, P8=T6.
  20/23 channels are derivable as differences of TUSZ -LE referential
  channels. FT9/FT10 are non-standard 10-10 positions not present in
  TUSZ's -LE montage -> the 3 channels involving them (T7-FT9, FT9-FT10,
  FT10-T8) are zero-filled, following the same fallback convention already
  used in utils/eval_chbmit_foundation_protocol.py for missing
  FOUNDATION_CHANNELS.

Output CSV columns:
  path            - absolute path to the saved .npy window file (raw, 23ch x 3840)
  subject         - subject ID string (e.g. "aaaaaflc")
  seizure_type    - fnsz / gnsz / cpsz / tcsz / absz / tnsz / spsz / mysz
  role            - "known" | "unknown"
  split           - "train" | "dev" | "eval"  (TUSZ official split)
  fold            - 0-4 (5-fold CV within train+dev for known classes; -1 for unknown/eval)
  edf_path        - source EDF file
  start_sec       - window start time in seconds within the EDF
  end_sec         - window end time in seconds
  contaminated    - True if subject is contaminated (excluded from training)

Usage
-----
  python build_tuh_openset_index.py \
      --tuh_root  /NCL_VAST_Datasets/tuh/TUSZ \
      --out_dir   /NCL_VAST_Datasets/Targol/eegfm_outputs/TUH/openset \
      --n_folds   5 \
      --seed      42

  # Dry run (parse annotations only, no windowing or .npy saving):
  python build_tuh_openset_index.py --tuh_root ... --out_dir ... --dry_run
"""

import argparse
import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants - match your existing EFM preprocessing pipeline exactly
# ---------------------------------------------------------------------------
WINDOW_SEC = 15  # seconds per window
SAMPLE_RATE = 256  # Hz - resample target (matches CHB-MIT native format)
WINDOW_SAMP = WINDOW_SEC * SAMPLE_RATE  # 3840 samples

# Seizure type taxonomy from your report
KNOWN_TYPES = {"fnsz", "gnsz", "cpsz", "tcsz"}
UNKNOWN_TYPES = {"absz", "tnsz", "spsz", "mysz"}
ALL_TYPES = KNOWN_TYPES | UNKNOWN_TYPES

# Minimum subjects per class to be "known" - your report threshold
MIN_SUBJECTS_KNOWN = 12

# Subjects contaminated by unknown-class events (Table 7 in your report).
# absz and tnsz are contaminated; spsz and mysz are clean.
CONTAMINATED_UNKNOWN_TYPES = {"absz", "tnsz"}

# Background (non-seizure) label
BCKG_LABEL = "bckg"

# TUSZ official splits to walk
SPLITS = ("train", "dev", "eval")

# ---------------------------------------------------------------------------
# Channel mapping: CHB-MIT 23-channel target montage
# ---------------------------------------------------------------------------
# Exact order from meta.json selected_channels (CHB-MIT native format)
TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8",
]

# Map each target bipolar channel to a (anode, cathode) pair of TUSZ -LE
# electrode names (old 10-20 nomenclature: T7=T3, T8=T4, P7=T5, P8=T6).
# None entries cannot be derived from TUSZ's -LE montage (FT9/FT10 missing)
# and are zero-filled, matching the fallback convention in
# utils/eval_chbmit_foundation_protocol.py.
TUSZ_DERIVATION = {
    "FP1-F7": ("FP1", "F7"),
    "F7-T7": ("F7", "T3"),
    "T7-P7": ("T3", "T5"),
    "P7-O1": ("T5", "O1"),
    "FP1-F3": ("FP1", "F3"),
    "F3-C3": ("F3", "C3"),
    "C3-P3": ("C3", "P3"),
    "P3-O1": ("P3", "O1"),
    "FP2-F4": ("FP2", "F4"),
    "F4-C4": ("F4", "C4"),
    "C4-P4": ("C4", "P4"),
    "P4-O2": ("P4", "O2"),
    "FP2-F8": ("FP2", "F8"),
    "F8-T8": ("F8", "T4"),
    "T8-P8": ("T4", "T6"),
    "P8-O2": ("T6", "O2"),
    "FZ-CZ": ("FZ", "CZ"),
    "CZ-PZ": ("CZ", "PZ"),
    "P7-T7": ("T5", "T3"),
    "T7-FT9": None,  # FT9 not in TUSZ -LE montage -> zero-fill
    "FT9-FT10": None,  # FT9/FT10 not in TUSZ -LE montage -> zero-fill
    "FT10-T8": None,  # FT10 not in TUSZ -LE montage -> zero-fill
    # Note: "T8-P8" appears twice in TARGET_CHANNELS (indices 14 and 22),
    # matching CHB-MIT's own meta.json (intentional duplicate in original
    # montage). Both map to the same TUSZ derivation T4-T6.
}

N_TARGET_CHANNELS = len(TARGET_CHANNELS)  # 23


# ---------------------------------------------------------------------------
# TUSZ multi-class CSV parser
# ---------------------------------------------------------------------------

def parse_tusz_csv(csv_path: Path) -> list:
    """
    Parse a TUSZ v2.0.3 multi-class .csv annotation file.

    Format:
      # version = csv_v1.0.0
      # bname = ...
      # duration = <secs>
      # montage_file = ...
      #
      channel,start_time,stop_time,label,confidence
      FP1-F7,0.0000,351.9844,bckg,1.0000
      FP1-F7,351.9844,407.0234,gnsz,1.0000
      ...

    The file is per-channel: every channel repeats (approximately) the same
    set of time segments. We aggregate to a single term-level timeline by
    majority vote across channels at each distinct interval defined by the
    union of all channel boundary points.

    Returns a list of merged term-level events:
      [{"start": float, "end": float, "label": str}, ...]
    sorted by start time, covering [0, duration].
    """
    duration = None
    rows = []  # (channel, start, end, label)

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "duration" in line:
                    try:
                        duration = float(line.split("=")[1].strip().split()[0])
                    except (IndexError, ValueError):
                        pass
                continue
            if line.startswith("channel,"):
                continue  # header
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                ch = parts[0]
                start = float(parts[1])
                end = float(parts[2])
                label = parts[3].lower()
                rows.append((ch, start, end, label))
            except ValueError:
                continue

    if not rows:
        return []

    if duration is None:
        duration = max(r[2] for r in rows)

    # Collect all distinct boundary points across all channels
    boundaries = set()
    for _, start, end, _ in rows:
        boundaries.add(round(start, 4))
        boundaries.add(round(end, 4))
    boundaries.add(0.0)
    boundaries.add(round(duration, 4))
    boundaries = sorted(boundaries)

    # For each sub-interval between consecutive boundaries, take majority
    # label across channels that cover that interval.
    merged_events = []
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        if seg_end - seg_start < 1e-6:
            continue
        seg_mid = (seg_start + seg_end) / 2.0

        label_votes = Counter()
        for ch, start, end, label in rows:
            if start <= seg_mid < end:
                label_votes[label] += 1

        if not label_votes:
            majority_label = BCKG_LABEL
        else:
            majority_label = label_votes.most_common(1)[0][0]

        merged_events.append({"start": seg_start, "end": seg_end, "label": majority_label})

    # Merge consecutive segments with the same label
    collapsed = []
    for ev in merged_events:
        if collapsed and collapsed[-1]["label"] == ev["label"] and abs(collapsed[-1]["end"] - ev["start"]) < 1e-6:
            collapsed[-1]["end"] = ev["end"]
        else:
            collapsed.append(dict(ev))

    return collapsed


def label_at_time(events: list, t_start: float, t_end: float) -> str:
    """
    Return the seizure-type label that covers the majority of [t_start, t_end].
    Returns BCKG_LABEL if the window is predominantly background or the
    overlapping label is not a recognized seizure type.
    """
    best_label = BCKG_LABEL
    best_overlap = 0.0

    for ev in events:
        if ev["label"] == BCKG_LABEL:
            continue
        overlap = max(0.0, min(ev["end"], t_end) - max(ev["start"], t_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = ev["label"]

    if best_overlap > (t_end - t_start) * 0.5:
        return best_label
    return BCKG_LABEL


# ---------------------------------------------------------------------------
# EDF loading + 23-channel bipolar derivation
# ---------------------------------------------------------------------------

def _norm_tusz_ch_name(raw_name: str) -> str:
    """
    Normalize a TUSZ EDF channel name to its bare 10-20 electrode label.
    'EEG FP1-LE' -> 'FP1'
    'EEG T3-LE'  -> 'T3'
    """
    name = raw_name.strip().upper()
    if name.startswith("EEG "):
        name = name[4:]
    if name.endswith("-LE"):
        name = name[:-3]
    elif name.endswith("-REF"):
        name = name[:-4]
    return name.strip()


def load_edf_raw_23ch(edf_path: Path, target_sr: int = SAMPLE_RATE):
    """
    Load a TUSZ EDF, bandpass filter, resample to target_sr, and derive the
    23-channel CHB-MIT-compatible bipolar montage (TARGET_CHANNELS) via
    differences of TUSZ -LE referential channels.

    Channels that cannot be derived (FT9/FT10-involving) are zero-filled.

    Returns:
      data_23: np.ndarray, shape (23, n_samples)
      missing: list of TARGET_CHANNELS entries that were zero-filled
    """
    import mne
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    orig_sr = int(raw.info["sfreq"])

    # Build lookup: normalized electrode name -> row index in raw data
    ch_lookup = {}
    for i, ch in enumerate(raw.ch_names):
        norm = _norm_tusz_ch_name(ch)
        if norm and norm not in ch_lookup:
            ch_lookup[norm] = i

    # Bandpass 0.5-70 Hz (matches CHB-MIT pipeline)
    raw.filter(l_freq=0.5, h_freq=70.0, method="iir", verbose=False)

    if orig_sr != target_sr:
        raw.resample(target_sr, verbose=False)

    full_data = raw.get_data()  # (n_channels_orig, n_samples)
    n_samples = full_data.shape[1]

    data_23 = np.zeros((N_TARGET_CHANNELS, n_samples), dtype=np.float64)
    missing = []

    for i, target_ch in enumerate(TARGET_CHANNELS):
        deriv = TUSZ_DERIVATION.get(target_ch)
        if deriv is None:
            missing.append(target_ch)
            continue
        anode, cathode = deriv
        if anode not in ch_lookup or cathode not in ch_lookup:
            missing.append(target_ch)
            continue
        data_23[i, :] = full_data[ch_lookup[anode], :] - full_data[ch_lookup[cathode], :]

    return data_23, missing


def z_score_normalize(data: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalization - matches CHB-MIT pipeline."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (data - mean) / std


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_windows(
        data: np.ndarray,
        events: list,
        edf_path: Path,
        subject: str,
        split_tag: str,
        out_dir: Path,
        dry_run: bool = False,
) -> list:
    """
    Slide a non-overlapping 15s window over the recording.
    Label each window; save .npy (23ch x 3840) for seizure-type windows only
    (background windows are skipped entirely - not needed for open-set
    seizure-type classification).
    Returns list of index row dicts.
    """
    n_ch, n_samp = data.shape
    rows = []

    n_windows = n_samp // WINDOW_SAMP
    for i in range(n_windows):
        t_start = i * WINDOW_SEC
        t_end = t_start + WINDOW_SEC
        s_start = i * WINDOW_SAMP
        s_end = s_start + WINDOW_SAMP

        label = label_at_time(events, t_start, t_end)
        if label == BCKG_LABEL or label not in ALL_TYPES:
            continue

        window = data[:, s_start:s_end]
        window = z_score_normalize(window)

        role = "known" if label in KNOWN_TYPES else "unknown"

        rel_path = Path(split_tag) / subject / edf_path.stem / f"w{i:06d}_{label}.npy"
        abs_path = out_dir / "windows" / rel_path

        if not dry_run:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(abs_path), window.astype(np.float32))

        rows.append({
            "path": str(abs_path),
            "subject": subject,
            "seizure_type": label,
            "role": role,
            "split": split_tag,
            "fold": -1,
            "edf_path": str(edf_path),
            "start_sec": round(t_start, 3),
            "end_sec": round(t_end, 3),
            "contaminated": False,
        })

    return rows


# ---------------------------------------------------------------------------
# 5-fold subject-stratified CV splitter (known classes only, train+dev split)
# ---------------------------------------------------------------------------

def assign_cv_folds(rows: list, n_folds: int = 5, seed: int = 42) -> list:
    """
    Assign fold 0-(n_folds-1) to known-class windows from the 'train' and 'dev'
    TUSZ splits, using subject-stratified, class-balanced greedy assignment.

    Problem with naive per-class round-robin: many subjects have events from
    MULTIPLE known classes (e.g. fnsz+gnsz co-occur in 64 subjects - see
    Table 5 in your report). A subject's fold is decided by whichever class
    is processed FIRST in a dict iteration, which can starve rare classes
    (e.g. tcsz, only 15 subjects) of balanced representation - some folds
    end up with almost no tcsz windows at all.

    Fix: process classes from RAREST to MOST COMMON (by subject count), and
    for each subject, assign it to whichever fold currently has the fewest
    WINDOWS of that subject's primary class (greedy balancing), not just
    index parity. Once a subject is assigned, it's locked for all its
    classes.

    'eval' split and unknown-class windows keep fold = -1 and are reserved
    as held-out open-set test data.
    """
    eligible = [r for r in rows
                if r["role"] == "known"
                and not r["contaminated"]
                and r["split"] in ("train", "dev")]

    # subject -> set of classes it has events for
    subject_classes = defaultdict(set)
    # subject -> {class: n_windows} for greedy balancing weight
    subject_class_windows = defaultdict(lambda: defaultdict(int))
    for r in eligible:
        subject_classes[r["subject"]].add(r["seizure_type"])
        subject_class_windows[r["subject"]][r["seizure_type"]] += 1

    # Process classes from RAREST to MOST COMMON (by subject count)
    subjects_by_class = defaultdict(set)
    for r in eligible:
        subjects_by_class[r["seizure_type"]].add(r["subject"])
    class_order = sorted(subjects_by_class.keys(),
                         key=lambda c: len(subjects_by_class[c]))

    rng = random.Random(seed)
    subject_to_fold = {}
    # Track per-fold, per-class window totals for greedy balancing
    fold_class_windows = defaultdict(lambda: defaultdict(int))

    for cls in class_order:
        subj_list = sorted(s for s in subjects_by_class[cls] if s not in subject_to_fold)
        rng.shuffle(subj_list)

        for subj in subj_list:
            # Assign to the fold with the FEWEST windows of this class so far
            # (greedy balancing, not naive round-robin)
            best_fold = min(range(n_folds), key=lambda f: fold_class_windows[f][cls])
            subject_to_fold[subj] = best_fold

            # Update running totals for ALL classes this subject contributes to
            # (since the subject is now locked into best_fold for everything)
            for c, n_win in subject_class_windows[subj].items():
                fold_class_windows[best_fold][c] += n_win

    for r in rows:
        if r["role"] == "known" and not r["contaminated"] and r["split"] in ("train", "dev"):
            r["fold"] = subject_to_fold.get(r["subject"], -1)

    return rows


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------

def mark_contaminated_subjects(rows: list):
    """
    Mark subjects contaminated if they appear in BOTH a known class
    AND a contaminated unknown class (absz or tnsz - Table 7 in your report).
    """
    subjects_with_unknown = defaultdict(set)
    for r in rows:
        if r["seizure_type"] in CONTAMINATED_UNKNOWN_TYPES:
            subjects_with_unknown[r["subject"]].add(r["seizure_type"])

    contaminated_subjects = set()
    for r in rows:
        if r["role"] == "known" and r["subject"] in subjects_with_unknown:
            r["contaminated"] = True
            contaminated_subjects.add(r["subject"])

    return rows, contaminated_subjects


# ---------------------------------------------------------------------------
# TUSZ directory walker
# ---------------------------------------------------------------------------

def iter_tusz_edfs(tuh_root: Path):
    """
    Walk the TUSZ v2.0.3 directory tree:
      <tuh_root>/edf/{train,dev,eval}/<subject>/<session>/<montage>/<rec>.edf
                                                                     <rec>.csv

    Yields: (edf_path, csv_path, subject_id, split_tag)
    """
    edf_root = tuh_root / "edf"

    for split_tag in SPLITS:
        split_dir = edf_root / split_tag
        if not split_dir.exists():
            continue
        for edf_path in sorted(split_dir.rglob("*.edf")):
            csv_path = edf_path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            try:
                subject_id = edf_path.relative_to(split_dir).parts[0]
            except ValueError:
                subject_id = "unknown"
            yield edf_path, csv_path, subject_id, split_tag


# ---------------------------------------------------------------------------
# Summary / sanity check printer
# ---------------------------------------------------------------------------

def print_summary(rows: list, contaminated_subjects: set, missing_channel_counts: Counter):
    type_counts = Counter(r["seizure_type"] for r in rows)
    type_subjects = defaultdict(set)
    for r in rows:
        type_subjects[r["seizure_type"]].add(r["subject"])

    print("\n" + "=" * 70)
    print("TUH Open-Set Index Summary")
    print("=" * 70)
    print(f"{'Type':<8} {'Role':<8} {'Windows':>8} {'Subjects':>9} {'Contaminated?':>14}")
    print("-" * 70)
    for t in sorted(ALL_TYPES, key=lambda x: -type_counts.get(x, 0)):
        role = "known" if t in KNOWN_TYPES else "unknown"
        n_win = type_counts.get(t, 0)
        n_subj = len(type_subjects.get(t, set()))
        contam = "YES" if t in CONTAMINATED_UNKNOWN_TYPES else "-"
        print(f"{t:<8} {role:<8} {n_win:>8,} {n_subj:>9} {contam:>14}")
    print("-" * 70)
    print(f"Total windows : {len(rows):,}")
    print(f"Total subjects: {len({r['subject'] for r in rows}):,}")
    print(f"Contaminated subjects (excluded from known training): "
          f"{len(contaminated_subjects)}")
    known_clean = [r for r in rows
                   if r["role"] == "known" and not r["contaminated"]]
    print(f"Known-class windows available for training: {len(known_clean):,}")

    # Split breakdown
    split_counts = Counter((r["seizure_type"], r["split"]) for r in rows)
    print("\nPer-split breakdown:")
    for t in sorted(ALL_TYPES, key=lambda x: -type_counts.get(x, 0)):
        line = f"  {t:<8}"
        for s in SPLITS:
            line += f" {s}={split_counts.get((t, s), 0):>6,}"
        print(line)

    # Channel derivation summary (per-EDF missing channel sets)
    print("\nChannel derivation (zero-filled channels per EDF):")
    if not missing_channel_counts:
        print("  All 20 derivable channels present in all processed EDFs.")
        print("  3 channels always zero-filled (FT9/FT10 not in TUSZ montage): "
              "T7-FT9, FT9-FT10, FT10-T8")
    else:
        for ch, cnt in missing_channel_counts.most_common():
            print(f"  {ch:<10} zero-filled in {cnt} EDFs")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build TUH open-set index")
    parser.add_argument("--tuh_root", required=True, type=Path,
                        help="Root of TUSZ v2.0.3 dataset (contains edf/{train,dev,eval})")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Output directory for index CSV and windows/")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Parse annotations only; do not load EDFs or save .npy files")
    parser.add_argument("--max_edfs", type=int, default=None,
                        help="Cap number of EDFs processed (for debugging)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    n_edfs = 0
    errors = []
    missing_channel_counts = Counter()  # which TARGET_CHANNELS got zero-filled, beyond the 3 known FT9/FT10 ones

    print(f"Scanning TUSZ at: {args.tuh_root}")
    print(f"Output dir      : {args.out_dir}")
    print(f"Dry run         : {args.dry_run}")
    print(f"Target montage  : {N_TARGET_CHANNELS} channels (CHB-MIT compatible)\n")

    for edf_path, csv_path, subject, split_tag in iter_tusz_edfs(args.tuh_root):
        if args.max_edfs and n_edfs >= args.max_edfs:
            break
        n_edfs += 1

        try:
            events = parse_tusz_csv(csv_path)

            seizure_labels = {e["label"] for e in events if e["label"] != BCKG_LABEL}
            if not seizure_labels.intersection(ALL_TYPES):
                continue  # no relevant seizure types in this recording

            if args.dry_run:
                for ev in events:
                    if ev["label"] not in ALL_TYPES:
                        continue
                    role = "known" if ev["label"] in KNOWN_TYPES else "unknown"
                    seg_len = ev["end"] - ev["start"]
                    n_windows = max(1, int(seg_len // WINDOW_SEC))
                    for i in range(n_windows):
                        t_start = ev["start"] + i * WINDOW_SEC
                        all_rows.append({
                            "path": f"DRY_RUN/{split_tag}/{subject}/{edf_path.stem}/w{i:06d}_{ev['label']}.npy",
                            "subject": subject,
                            "seizure_type": ev["label"],
                            "role": role,
                            "split": split_tag,
                            "fold": -1,
                            "edf_path": str(edf_path),
                            "start_sec": round(t_start, 3),
                            "end_sec": round(t_start + WINDOW_SEC, 3),
                            "contaminated": False,
                        })
            else:
                data_23, missing = load_edf_raw_23ch(edf_path)
                # Track any unexpected missing channels (beyond the known
                # FT9/FT10 trio, which are always missing by design)
                for ch in missing:
                    if TUSZ_DERIVATION.get(ch) is not None:
                        missing_channel_counts[ch] += 1

                rows = extract_windows(data_23, events, edf_path, subject, split_tag,
                                       args.out_dir, dry_run=False)
                all_rows.extend(rows)

        except Exception as e:
            errors.append({"edf": str(edf_path), "error": str(e)})
            print(f"  [ERROR] {edf_path.name}: {e}")

        if n_edfs % 50 == 0:
            print(f"  Processed {n_edfs} EDFs, {len(all_rows):,} windows so far...")

    print(f"\nDone scanning. {n_edfs} EDFs processed, {len(all_rows):,} windows.")

    # --- Contamination check ---
    all_rows, contaminated_subjects = mark_contaminated_subjects(all_rows)

    # --- CV fold assignment (train+dev only) ---
    all_rows = assign_cv_folds(all_rows, n_folds=args.n_folds, seed=args.seed)

    # --- Summary ---
    print_summary(all_rows, contaminated_subjects, missing_channel_counts)

    # --- Write index CSV ---
    csv_out_path = args.out_dir / "tuh_openset_index.csv"
    fieldnames = ["path", "subject", "seizure_type", "role", "split", "fold",
                  "edf_path", "start_sec", "end_sec", "contaminated"]
    with open(csv_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Index CSV written to: {csv_out_path}")

    # --- Write contaminated subject list ---
    contam_path = args.out_dir / "contaminated_subjects.json"
    with open(contam_path, "w") as f:
        json.dump(sorted(contaminated_subjects), f, indent=2)
    print(f"Contaminated subjects written to: {contam_path}")

    # --- Write channel mapping metadata (for downstream reproducibility) ---
    chmap_path = args.out_dir / "channel_mapping.json"
    with open(chmap_path, "w") as f:
        json.dump({
            "target_channels": TARGET_CHANNELS,
            "derivation": {k: (list(v) if v else None) for k, v in TUSZ_DERIVATION.items()},
            "always_zero_filled": ["T7-FT9", "FT9-FT10", "FT10-T8"],
            "sample_rate_hz": SAMPLE_RATE,
            "window_sec": WINDOW_SEC,
        }, f, indent=2)
    print(f"Channel mapping written to: {chmap_path}")

    # --- Write error log ---
    if errors:
        err_path = args.out_dir / "build_errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"[WARNING] {len(errors)} errors logged to: {err_path}")

    # --- Write fold summary ---
    fold_summary = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        if r["role"] == "known" and not r["contaminated"] and r["fold"] >= 0:
            fold_summary[r["fold"]][r["seizure_type"]] += 1
    fold_path = args.out_dir / "fold_summary.json"
    with open(fold_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(fold_summary.items())}, f, indent=2)
    print(f"Fold summary written to: {fold_path}")


if __name__ == "__main__":
    main()
