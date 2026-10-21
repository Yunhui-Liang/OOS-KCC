"""Reproduce OOS-KCC results on the bundled MFeat dataset.

The data files and exact five stratified 30/70 index lists are packaged here.
Reference metrics come from the original exhaustive campaign, not the paper table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oos_kcc import OOSKCC, all_metrics  # noqa: E402

DATASETS = ("MFeat",)
METRICS = ("ACC", "NMI", "ARI")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def partition_sha(pred: np.ndarray) -> str:
    """Cluster IDs have no intrinsic meaning; hash the partition after canonical relabeling."""
    _, first, inverse = np.unique(pred, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=np.int32)
    remap[order] = np.arange(len(order), dtype=np.int32)
    return sha_array(remap[inverse])


def load_dataset(name: str):
    folder = ROOT / "data" / name
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    label_path = folder / meta["labels"]["file"]
    if sha_file(label_path) != meta["labels"]["sha256_file"]:
        raise RuntimeError(f"{name}: labels file checksum mismatch")
    y = np.load(label_path, mmap_mode="r", allow_pickle=False)
    views = []
    for view in meta["views"]:
        path = folder / view["file"]
        if sha_file(path) != view["sha256_file"]:
            raise RuntimeError(f"{name}: {view['file']} checksum mismatch")
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(x.shape) != view["shape"] or str(x.dtype) != view["dtype"]:
            raise RuntimeError(f"{name}: {view['file']} shape/dtype mismatch")
        views.append(x)
    if len(y) != meta["n_samples"] or len(views) != meta["n_views"]:
        raise RuntimeError(f"{name}: dataset metadata mismatch")
    return views, np.asarray(y, dtype=np.int32), meta


def read_reference():
    with (ROOT / "reference" / "paper_seven_cohort_per_seed.csv").open(newline="", encoding="utf-8") as f:
        return {(r["dataset"], int(r["seed"])): r for r in csv.DictReader(f)}


def reproduce(names: list[str], seeds: list[int], verify: bool):
    reference = read_reference()
    rows = []
    with threadpool_limits(limits=1):
        for name in names:
            views, y, meta = load_dataset(name)
            k = int(meta["n_classes"])
            for seed in seeds:
                old = reference[(name, seed)]
                split_path = ROOT / "reference" / "splits" / name / f"seed_{seed:02d}.npz"
                with np.load(split_path, allow_pickle=False) as split:
                    train = np.asarray(split["train_idx"], dtype=np.int64)
                    test = np.asarray(split["test_idx"], dtype=np.int64)
                if sha_array(train) != old["train_index_sha256"] or sha_array(test) != old["test_index_sha256"]:
                    raise RuntimeError(f"{name} seed {seed}: split checksum mismatch")
                xtrain = [np.ascontiguousarray(x[train], dtype=np.float32) for x in views]
                xtest = [np.ascontiguousarray(x[test], dtype=np.float32) for x in views]
                model = OOSKCC(
                    k, base_models_per_view=3, base_n_init=2, base_max_iter=150,
                    kcc_n_init=8, kcc_max_iter=200, random_state=seed,
                    standardize=True, base_protocol="fixed",
                )
                model.fit(xtrain)
                pred = model.predict(xtest, mode="soft", batch_size=32768)
                metrics = all_metrics(y[test], pred)
                pred_sha = sha_array(np.asarray(pred, dtype=np.int32))
                part_sha = partition_sha(pred)
                metric_error = max(abs(metrics[m] - float(old[m])) for m in METRICS)
                exact = pred_sha == old["prediction_array_sha256"]
                same_partition = part_sha == old["prediction_partition_sha256"]
                row = {"dataset": name, "seed": seed, "n_train": len(train), "n_test": len(test),
                       **metrics, "prediction_array_sha256": pred_sha,
                       "prediction_partition_sha256": part_sha,
                       "reference_metric_max_abs_error": metric_error,
                       "reference_labels_exact": exact,
                       "reference_partition_exact": same_partition}
                rows.append(row)
                print(f"{name:9s} seed={seed} ACC={metrics['ACC']:.6f} NMI={metrics['NMI']:.6f} "
                      f"ARI={metrics['ARI']:.6f} partition_exact={same_partition} labels_exact={exact} "
                      f"error={metric_error:.3g}", flush=True)
                if verify and (metric_error > 1e-10 or not same_partition):
                    raise RuntimeError(f"{name} seed {seed}: mismatch with original run; see saved partial results")
    return rows


def save(rows: list[dict], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for name in DATASETS:
        group = [r for r in rows if r["dataset"] == name]
        if not group:
            continue
        record = {"dataset": name, "n_runs": len(group)}
        for metric in METRICS:
            values = np.array([r[metric] for r in group], dtype=float)
            record[f"{metric}_mean_pct"] = 100 * values.mean()
            record[f"{metric}_sample_sd_pct"] = 100 * values.std(ddof=1) if len(values) > 1 else float("nan")
        summary.append(record)
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"Saved {outdir / 'per_seed.csv'} and {outdir / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--no-verify", action="store_true", help="Compute results without failing on reference mismatch")
    args = parser.parse_args()
    rows = reproduce(args.datasets, args.seeds, verify=not args.no_verify)
    save(rows, args.output)


if __name__ == "__main__":
    main()
