"""Select projections on train/validation, then evaluate a frozen run."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform
import time
from urllib.request import urlopen

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from reducers import grid_laplacian, orient, principal_axes, robust_axes

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits"
DATA_SHA = {
    "tra": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
    "tes": "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
}
SEEDS = (11, 22, 33)
CONDITIONS = ("clean", "contaminated")
METHODS = ("Pixels", "PCA", "PCA-L1", "RSPCA", "RSMPCA", "RSSPCA")
RANK = 16
GRID = tuple(itertools.product((0.01, 0.1, 1.0), (0.1, 1.0, 10.0)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digits(folder, suffix):
    path = folder / f"optdigits.{suffix}"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with urlopen(f"{BASE_URL}/{path.name}", timeout=25) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != DATA_SHA[suffix]:
            raise ValueError("Unexpected data digest from UCI")
        with path.open("xb") as handle:
            handle.write(content)
    if digest(path) != DATA_SHA[suffix]:
        raise ValueError(f"Unexpected digest for {path.name}")
    values = np.loadtxt(path, delimiter=",", dtype=np.int64)
    n = 3823 if suffix == "tra" else 1797
    if values.shape != (n, 65):
        raise ValueError("Unexpected Optdigits shape")
    return values[:, :64] / 16.0, values[:, 64]


def partitions(labels, seed):
    fit, val = train_test_split(
        np.arange(len(labels)), test_size=765, random_state=seed, stratify=labels
    )
    fit, val = np.sort(fit), np.sort(val)
    if np.intersect1d(fit, val).size or fit.size + val.size != len(labels):
        raise ValueError("Invalid validation split")
    return fit, val


def contaminate(gallery, seed):
    """Training-only salt-and-pepper replacement at the block-count level."""
    rng = np.random.default_rng(seed + 1000)
    changed = gallery.copy()
    images = rng.choice(len(gallery), size=round(0.30 * len(gallery)), replace=False)
    n_cells = round(0.35 * gallery.shape[1])
    for image in images:
        cells = rng.choice(gallery.shape[1], n_cells, replace=False)
        changed[image, cells] = rng.integers(0, 2, n_cells)
    return changed, {"images_targeted": int(images.size), "cells_per_image": n_cells,
                     "cells_changed": int(np.count_nonzero(changed != gallery))}


def predict(fit, queries, labels, axes):
    if axes is not None:
        fit, queries = fit @ axes, queries @ axes
    # cdist plus argmin yields deterministic first-index tie breaking.
    return np.concatenate([
        labels[cdist(queries[i:i + 256], fit, metric="sqeuclidean").argmin(axis=1)]
        for i in range(0, len(queries), 256)
    ])


def fingerprint():
    paths = ["src/reducers.py", "src/experiment.py", "docs/PROTOCOL.md"]
    return {path: digest(ROOT / path) for path in paths}


def record_basis(axes, lap, iterations, changes):
    return {
        "columns": int(axes.shape[1]),
        "numerical_rank": int(np.linalg.matrix_rank(axes)),
        "mean_l1": float(np.abs(axes).sum(0).mean()),
        "mean_roughness": float(np.sum(axes * (lap @ axes), axis=0).mean()),
        "mean_support_001": float(np.count_nonzero(np.abs(axes) >= .01, axis=0).mean()),
        "steps": iterations,
        "last_relative_changes": changes,
        "above_tolerance_at_cap": sum(c > 1e-4 for c in changes),
    }


def select(data, run):
    start = time.perf_counter()
    x, y = digits(data, "tra")
    run.mkdir(parents=True, exist_ok=False)
    plan = fingerprint()
    lap = grid_laplacian(8, 8)
    frozen = []
    for seed in SEEDS:
        fit, val = partitions(y, seed)
        for condition in CONDITIONS:
            original = x[fit]
            if condition == "clean":
                gallery = original
                damage = {"images_targeted": 0, "cells_per_image": 0, "cells_changed": 0}
            else:
                gallery, damage = contaminate(original, seed)
            mean = gallery.mean(axis=0)
            centered, val_x = gallery - mean, x[val] - mean
            ordinary = np.column_stack([orient(w) for w in principal_axes(centered, RANK).T])
            trial_results, best_count, best = [], -1, None
            for eta1, eta2 in GRID:
                axes, steps, changes = robust_axes(centered, ordinary, lap, eta1, eta2)
                n_correct = int(np.sum(predict(centered, val_x, y[fit], axes) == y[val]))
                trial_results.append({"eta1": eta1, "eta2": eta2, "correct": n_correct})
                if n_correct > best_count:
                    best_count = n_correct
                    best = (axes, steps, changes, eta1, eta2)
            axes, steps, changes, eta1, eta2 = best
            models = {"PCA": (ordinary, [], []), "RSSPCA": (axes, steps, changes)}
            for name, a, b in (("PCA-L1", 0, 0), ("RSPCA", eta1, 0), ("RSMPCA", 0, eta2)):
                models[name] = robust_axes(centered, ordinary, lap, a, b)
            checks = {
                name: record_basis(basis, lap, steps, changes)
                for name, (basis, steps, changes) in models.items()
            }
            val_correct = {
                name: int(np.sum(predict(centered, val_x, y[fit],
                    None if name == "Pixels" else models[name][0]) == y[val]))
                for name in METHODS
            }
            model_path = run / f"seed_{seed}_{condition}.npz"
            np.savez_compressed(model_path, mean=mean, fit=fit, validation=val,
                                **{name: params[0] for name, params in models.items()})
            frozen.append({
                "seed": seed, "condition": condition, "fit_size": len(fit), "validation_size": len(val),
                "corruption": damage, "eta1": eta1, "eta2": eta2, "trials": trial_results,
                "validation_correct": val_correct, "projection_checks": checks,
                "model": model_path.name, "model_sha256": digest(model_path),
            })
            print(f"seed {seed} {condition:12} selected ({eta1:g}, {eta2:g}), "
                  f"validation {best_count}/{len(val)}", flush=True)
    if fingerprint() != plan:
        raise RuntimeError("Source or protocol changed while selecting")
    json_write(run / "selection.json", {
        "fit_sha256": DATA_SHA["tra"], "sources": plan, "rank": RANK, "rows": frozen,
        "seconds": time.perf_counter() - start, "python": platform.python_version(),
        "versions": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        "threads": 1,
    })
    print("Frozen selection saved. Test data have not been read by this command.", flush=True)


def evaluate(data, run):
    start = time.perf_counter()
    sel_path = run / "selection.json"
    selected = json_read(sel_path)
    if (run / "predictions.csv").exists() or (run / "results.json").exists():
        raise FileExistsError("Final outputs already exist in this run")
    if selected["sources"] != fingerprint():
        raise ValueError("Frozen code or protocol was changed")
    for row in selected["rows"]:
        if digest(run / row["model"]) != row["model_sha256"]:
            raise ValueError("Frozen projection was changed")
    x, y = digits(data, "tra")
    test_x, test_y = digits(data, "tes")
    results = []
    with (run / "predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["seed", "condition", "method", "test_index", "true", "predicted"])
        for row in selected["rows"]:
            seed, condition = row["seed"], row["condition"]
            with np.load(run / row["model"], allow_pickle=False) as frozen:
                fit = frozen["fit"]
                gallery = x[fit] if condition == "clean" else contaminate(x[fit], seed)[0]
                mean = frozen["mean"]
                if not np.array_equal(mean, gallery.mean(axis=0)):
                    raise ValueError("Fit preprocessing changed after selection")
                for method in METHODS:
                    axes = None if method == "Pixels" else frozen[method]
                    p = predict(gallery - mean, test_x - mean, y[fit], axes)
                    num_correct = int(np.sum(p == test_y))
                    results.append({
                        "seed": seed, "condition": condition, "method": method, "correct": num_correct,
                        "test_size": len(test_y), "accuracy": 100 * num_correct / len(test_y),
                        "macro_f1": 100 * f1_score(test_y, p, average="macro"),
                    })
                    out.writerows((seed, condition, method, i, a, b)
                                  for i, (a, b) in enumerate(zip(test_y, p)))
                    print(f"seed {seed} {condition:12} {method:7} {num_correct}/{len(test_y)}", flush=True)
    json_write(run / "results.json", {
        "selection_sha256": digest(sel_path), "test_sha256": DATA_SHA["tes"],
        "predictions_sha256": digest(run / "predictions.csv"), "rows": results,
        "seconds": time.perf_counter() - start,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "input")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "processed" / "experiment")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        if args.stage == "select":
            select(args.data_dir, args.run_dir)
        else:
            evaluate(args.data_dir, args.run_dir)


if __name__ == "__main__":
    main()
