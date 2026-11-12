# Open-Set EEG Seizure Type Classification on TUSZ

Code, subject splits, and evaluation scripts for

> **A Benchmark Protocol and Fairness Analysis Towards Open-Set EEG Seizure Type Classification**
> Targol Bakhtiarvand et al. WACV 2027 Workshops (CVPath-Vision4Neuro, Evaluations and Datasets Track).

The protocol asks a seizure-type classifier, trained on **known** types, to also
**reject** seizure types it never saw. It is built on TUSZ v2.0.3 and a pretrained
multimodal EEG Foundation Model (EFM)
([Bakhtiarvand et al., IEEE SMC 2026](https://arxiv.org/abs/2607.21384)).

| | Seizure types |
|---|---|
| Known (≥12 subjects) | `fnsz`, `gnsz`, `cpsz`, `tcsz` |
| True unknown | `absz`, `tnsz`, `spsz`, `mysz` |

* **Subject-disjoint partition**: per seed, known-class subjects are split 60/20/20
  into fine-tune / validation / test. Rejection thresholds are calibrated on the
  validation subjects; known-class scores come from the test subjects.
* **Leave-one-class-out (LOCO)**: 4 rounds, each holding out one known type and
  fine-tuning a fresh model on the other three. The held-out type is evaluated
  separately for subjects **unseen** (primary) and **seen** during fine-tuning.
* **Scorers**: MSP, MaxLogit, Energy, Mahalanobis, kNN, GHOST, COSTARR (all post-hoc).
* **Metrics**: AUROC, AUOSCR, CCR@FPR{5,10,20}%, F@C95, per-subject rejection-rate COV,
  Wilcoxon signed-rank tests. Three seeds (42, 43, 44).

## Repository layout

```
models/                      EFM architecture (Mamba raw encoder + Transformer TF encoder + fusion)
openset/
  build_tuh_openset_index.py  TUSZ EDF -> 15 s windows (.npy) + index CSV
  loco_openset_v2.py          dataset, partial fine-tuning model, training loop
  unified_openset_eval.py     full camera-ready evaluation -> unified_results.json
  make_tables_figures.py      paper tables + Figure 1 from the result JSONs
  partition_counts.py         subject/window counts per class and partition
  check_pretrain_overlap.py   overlap between TUSZ subjects and EFM pretraining data
splits/
  split_seed{42,43,44}.json   subject -> train / val / test (TUSZ subject IDs)
```

## 1. Data

TUSZ is distributed by the Neural Engineering Data Consortium and requires
registration: <https://isip.piconepress.com/projects/nedc/html/tuh_eeg/>.
We do **not** redistribute any EEG. Download TUSZ v2.0.3 yourself.

## 2. EFM checkpoint

The pretrained EFM checkpoint is available on request.

## 3. Environment

```bash
docker build -t openset-tusz .
docker run --rm -it --gpus '"device=0"' --shm-size 16g \
  -v $(pwd):/workspace/openset-seizure-tusz \
  -v /path/to/TUSZ:/data/TUSZ -v /path/to/outputs:/outputs \
  openset-tusz bash
```

A CUDA GPU is required (mamba-ssm). Experiments used one 16 GB V100 per seed.

## 4. Reproduce

```bash
# (a) windows + index (15 s, 256 Hz, 23-ch bipolar montage)
python openset/build_tuh_openset_index.py --tuh_root /data/TUSZ --out_dir /outputs/index

# (b) evaluation, one seed per run (≈ 1 day per seed on one V100)
for s in 42 43 44; do
  python -u openset/unified_openset_eval.py \
    --index_csv /outputs/index/tuh_openset_index.csv \
    --backbone  checkpoints/week3_mixed_step20000.pt \
    --out_dir   /outputs/seed$s --seeds $s --batch_size 16 --device cuda
done

# (c) tables + Figure 1
python openset/make_tables_figures.py \
  --results /outputs/seed42/unified_results.json \
            /outputs/seed43/unified_results.json \
            /outputs/seed44/unified_results.json \
  --out_dir /outputs/paper

# (d) partition table (uses the released splits)
python openset/partition_counts.py /outputs/index/tuh_openset_index.csv \
  splits/split_seed42.json splits/split_seed43.json splits/split_seed44.json
```

`unified_openset_eval.py` regenerates `split_seed<seed>.json` deterministically;
the files in `splits/` are the exact partitions used in the paper.

## Citation

```bibtex
@inproceedings{bakhtiarvand2027openset,
  title     = {A Benchmark Protocol and Fairness Analysis Towards Open-Set {EEG} Seizure Type Classification},
  author    = {Bakhtiarvand, Targol and others},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}

@inproceedings{bakhtiarvand2026efm,
  title     = {Multimodal Pretraining for Generalizable {EEG} Representation Learning},
  author    = {Bakhtiarvand, Targol and Kalita, Jugal and Atyabi, Adham},
  booktitle = {Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year      = {2026},
  note      = {To appear. arXiv:2607.21384}
}
```

## License

MIT (code). TUSZ data are subject to the NEDC data use agreement.
