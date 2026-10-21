# Dataset provenance

This repository includes the **exact prepared MFeat arrays** used for the paper's seven-dataset main-cohort results. The public `metadata.json` retains portable metadata, per-file SHA-256 hashes, and the recorded source description.

| Dataset | Original main-cohort ID | Source | Contents |
|---|---|---|---|
| MFeat | `prepared_mfeat_mvc` | [Robert Duin (1998), Multiple Features, UCI Machine Learning Repository, DOI 10.24432/C5HC70](https://archive.ics.uci.edu/dataset/72/multiple+features) | Included: six aligned float32 views, each source row L2-normalized; original row order. Source license: CC BY 4.0. |

**MFeat representation matters:** an older, separate five-dataset supplement used a different unnormalized `MFeat.npz` and yielded a different NMI. This repository uses the prepared main-cohort version corresponding to the paper's 87.63 ± 1.09 NMI.

The original run-level results for all three datasets are recorded in `reference/`.
