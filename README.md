# OOS-KCC: reproducible submission implementation

This repository contains the main OOS-KCC algorithm from **“Scalable Out-of-Sample K-Means Consensus Clustering via Sparse Consensus Transfer”**, the prepared MFeat dataset, its exact five train/test splits, and per-run reference results. The included MFeat experiment runs end to end.

## Reproduce the bundled MFeat results

Use Python 3.12 (the recorded run used 3.12.14). From this folder:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

The script checks the MFeat data and split hashes, fits a fresh model for each of seeds 0–4, predicts the held-out 70%, and writes `results/per_seed.csv` and `results/summary.csv`. By default it compares every result with the MFeat rows in `reference/paper_seven_cohort_per_seed.csv` and stops if a metric or the predicted **partition** differs. Cluster ID names are arbitrary.

On Windows, `python` can be replaced with the full path to a Python 3.12 executable. To run one split: `python scripts/reproduce.py --seeds 0`.

## Recorded results (ACC / NMI / ARI, percent)

Each cell is the mean ± sample standard deviation (`ddof=1`) over the five saved paired 30/70 splits.

| Dataset | N | Views | ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|
| Augsburg | 3,191 | 3 | 53.48 ± 7.03 | 33.44 ± 2.71 | 28.10 ± 7.35 |
| MFeat | 2,000 | 6 | 93.34 ± 0.97 | 87.63 ± 1.09 | 86.02 ± 1.77 |
| MNIST10K | 10,000 | 3 | 76.54 ± 4.14 | 68.89 ± 1.29 | 63.32 ± 2.71 |

These values come from actual per-seed runs. The MFeat row can be reproduced directly from this repository. `reference/verified_reproduction_per_seed.csv` records the earlier reproduction of all three datasets, and `reference/verified_reproduction_summary.csv` records its aggregation.

## Protocol

- For each seed 0–4, use MFeat's saved stratified 30% training / 70% future index arrays. The labels are used to create the benchmark split and determine the published number of clusters `K`; they are not passed to model fitting.
- Fit `StandardScaler` for each view on training data only.
- Fit three base `KMeans` models per view, each with `K` prototypes, `n_init=2`, `max_iter=150`, seed `split_seed*1000 + view*100 + member`.
- Fit KCC consensus on the concatenated training base partitions with `n_init=8`, `max_iter=200`, seed `777+split_seed`. Compile each transfer matrix `Q` from the training samples' KCC consensus assignments, without using their true class labels.
- Predict each future object using two nearest prototypes, the frozen `Q` matrices, equal member/view aggregation, then `argmax`; `batch_size=32768` affects only chunking.
- ACC uses a rectangular Hungarian one-to-one matching of predicted and true cluster labels. NMI is scikit-learn `normalized_mutual_info_score` (arithmetic normalization, its default), and ARI is `adjusted_rand_score`. Results are multiplied by 100 only for display.

The bundled `oos_kcc/core.py` and `oos_kcc/metrics.py` are byte-for-byte copies of the project implementation used for this release. The fixed protocol is explicitly selected by the reproduction script; other optional modes in the source are not used here.

## Folder layout

```text
oos_kcc/                     Main method and metric definitions
scripts/reproduce.py         Standalone five-seed MFeat runner and reference check
data/MFeat/                  Prepared views, labels, metadata and checksums
reference/splits/MFeat/     Exact ordered train/test indices, seeds 0–4
reference/*.csv              Original run-level ledger and verified reproduction
requirements.txt             Recorded dependency versions
DATA_PROVENANCE.md           Sources, transforms and data availability
```

