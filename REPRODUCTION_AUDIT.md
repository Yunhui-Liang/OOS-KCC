# Release verification

The original three-dataset release package was verified on 2026-10-20 with Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.7.0, and one numerical thread. This public repository includes MFeat for direct reproduction and preserves the three-dataset run records.

Command used on the original three-dataset package: `python scripts/reproduce.py`. In this public repository, the same command runs the bundled MFeat dataset.

- 15/15 packaged dataset×seed runs completed.
- 15/15 ACC, NMI and ARI triples reproduced the original run-level ledger exactly at the reported floating-point precision (maximum absolute metric error zero on this host).
- 15/15 predicted partitions matched the original saved predictions after canonicalizing arbitrary cluster IDs.
- 14/15 raw integer label vectors matched byte-for-byte. Augsburg seed 0 has different cluster ID names but the same partition and metrics.
- Three five-run means and sample standard deviations reproduce the displayed paper table values in `README.md`.

The comparison was made against original run-level records and saved predictions, not manually copied means. The corresponding metrics and prediction-partition hashes are preserved in `reference/`.

