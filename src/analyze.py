"""Verify measured outputs and generate summary data and report figures."""

import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "temp" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from experiment import CONDITIONS, METHODS, SEEDS, digest, json_read, json_write


def sample_summary(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def analyze(run):
    selected = json_read(run / "selection.json")
    result = json_read(run / "results.json")
    if result["selection_sha256"] != digest(run / "selection.json"):
        raise ValueError("Selection digest mismatch")
    if result["predictions_sha256"] != digest(run / "predictions.csv"):
        raise ValueError("Prediction digest mismatch")
    groups = defaultdict(list)
    with (run / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["condition"], row["method"])
            groups[key].append([int(row[name]) for name in ("test_index", "true", "predicted")])
    expected_keys = {(s, c, m) for s in SEEDS for c in CONDITIONS for m in METHODS}
    if groups.keys() != expected_keys:
        raise ValueError("Missing or extra prediction groups")
    truth = np.array(groups[next(iter(groups))])[:, 1]
    lookup = {}
    for row in result["rows"]:
        key = (row["seed"], row["condition"], row["method"])
        index, y, p = np.array(groups[key]).T
        correct = int(np.count_nonzero(y == p))
        if len(y) != 1797 or not np.array_equal(index, np.arange(len(truth))) or not np.array_equal(y, truth):
            raise ValueError("Prediction groups use different test samples")
        if correct != row["correct"] or not np.isclose(100 * correct / len(y), row["accuracy"]):
            raise ValueError("Accuracy mismatch")
        if not np.isclose(100 * f1_score(y, p, average="macro"), row["macro_f1"]):
            raise ValueError("Macro-F1 mismatch")
        lookup[key] = row

    aggregate, contrast, loading_checks = {}, {}, {}
    for condition in CONDITIONS:
        aggregate[condition] = {
            m: {metric: sample_summary([lookup[s, condition, m][metric] for s in SEEDS])
                for metric in ("accuracy", "macro_f1")}
            for m in METHODS
        }
        contrast[condition] = {}
        for baseline in ("PCA", "PCA-L1", "RSPCA", "RSMPCA"):
            delta = [lookup[s, condition, "RSSPCA"]["accuracy"] -
                     lookup[s, condition, baseline]["accuracy"] for s in SEEDS]
            contrast[condition][baseline] = dict(by_seed=delta, **sample_summary(delta))
        rows = [row for row in selected["rows"] if row["condition"] == condition]
        loading_checks[condition] = {}
        for method in METHODS[1:]:
            checks = [row["projection_checks"][method] for row in rows]
            if any(c["numerical_rank"] != c["columns"] for c in checks):
                raise ValueError("Fitted basis is rank deficient")
            loading_checks[condition][method] = {
                field: float(np.mean([m[field] for m in checks]))
                for field in ("mean_l1", "mean_roughness", "mean_support_001")
            }
            loading_checks[condition][method]["capped_loadings"] = sum(c["above_tolerance_at_cap"] for c in checks)
    seed, condition = SEEDS[0], "contaminated"
    yp = np.asarray(groups[seed, condition, "RSSPCA"])[:, 2]
    pca = np.asarray(groups[seed, condition, "PCA"])[:, 2]
    cm = confusion_matrix(truth, yp, labels=np.arange(10))
    offdiag = cm.copy()
    np.fill_diagonal(offdiag, 0)
    a, b = map(int, np.unravel_index(offdiag.argmax(), offdiag.shape))
    analysis = {
        "aggregate": aggregate, "contrasts": contrast, "basis_statistics": loading_checks,
        "representative": {
            "seed": seed, "condition": condition,
            "rss_correct_only": int(np.count_nonzero((yp == truth) & (pca != truth))),
            "pca_correct_only": int(np.count_nonzero((pca == truth) & (yp != truth))),
            "largest_rss_confusion": {"true": a, "predicted": b, "count": int(cm[a, b])},
        },
        "verified_predictions": sum(map(len, groups.values())),
    }
    json_write(run / "summary.json", analysis)
    return aggregate, lookup


def figures(run, out, aggregate, lookup):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.6, 3.35), layout="constrained")
    colors = ["#7C8694", "#375F89", "#638EAD", "#668672", "#B38B47", "#BA4936"]
    for condition, ax in zip(CONDITIONS, axs):
        for i, (method, color) in enumerate(zip(METHODS, colors)):
            vals = [lookup[s, condition, method]["accuracy"] for s in SEEDS]
            ax.scatter(vals, [i - .12, i, i + .12], s=15, color=color, alpha=.60)
            ax.scatter(aggregate[condition][method]["accuracy"]["mean"],
                       i, s=48, marker="D", color=color, edgecolor="white", linewidth=.55)
        ax.set_yticks(range(len(METHODS)), METHODS)
        ax.set_title("Clean fit images" if condition == "clean" else "Contaminated fit images")
        ax.set_xlabel("Held-out accuracy (%)")
        ax.grid(axis="x", alpha=.25)
    axs[0].invert_yaxis()
    fig.savefig(out / "accuracy.pdf")
    plt.close(fig)

    with np.load(run / "seed_11_contaminated.npz", allow_pickle=False) as model:
        images = [model[m][:, :2].T.reshape(2, 8, 8) for m in METHODS[1:]]
    vmax = max(np.abs(im).max() for im in images)
    fig, axs = plt.subplots(2, 5, figsize=(8.6, 3.0), layout="constrained")
    for col, (method, im) in enumerate(zip(METHODS[1:], images)):
        for row in range(2):
            grid = axs[row, col].imshow(im[row], vmin=-vmax, vmax=vmax, cmap="RdBu_r",
                                       interpolation="nearest")
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
        axs[0, col].set_title(method)
    axs[0, 0].set_ylabel("Component 1")
    axs[1, 0].set_ylabel("Component 2")
    bar = fig.colorbar(grid, ax=axs, pad=.025, fraction=.035, shrink=.85)
    bar.set_label("Loading")
    fig.savefig(out / "loadings.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "processed" / "experiment")
    parser.add_argument("--figures", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    aggregate, lookup = analyze(args.run_dir)
    figures(args.run_dir, args.figures, aggregate, lookup)
    print(f"Verified {len(lookup)} prediction groups and wrote aggregate results.")


if __name__ == "__main__":
    main()
