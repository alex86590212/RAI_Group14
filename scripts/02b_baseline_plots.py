import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from rai.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IN_DIR = RESULTS_DIR / "baseline"
OUT_DIR = IN_DIR / "plots"
GRID = np.linspace(0, 1, 101)


def plot_overview(predictions: pd.DataFrame) -> None:
    seeds = sorted(predictions["seed"].unique())
    base_rate = predictions["y_true"].mean()
    fig, (roc_ax, pr_ax, dist_ax) = plt.subplots(1, 3, figsize=(16, 4.8))

    tprs, precisions = [], []
    for seed in seeds:
        d = predictions[predictions["seed"] == seed]
        fpr, tpr, _ = roc_curve(d["y_true"], d["score"])
        roc_ax.plot(fpr, tpr, color="tab:blue", alpha=0.15, lw=1)
        tprs.append(np.interp(GRID, fpr, tpr))
        prec, rec, _ = precision_recall_curve(d["y_true"], d["score"])
        keep = rec[:-1] > 0
        prec, rec = prec[:-1][keep], rec[:-1][keep]
        pr_ax.plot(rec, prec, color="tab:blue", alpha=0.15, lw=1)
        precisions.append(np.interp(GRID, rec[::-1], prec[::-1]))
    roc_ax.plot(GRID, np.mean(tprs, axis=0), color="tab:blue", lw=2, label="mean over seeds")
    roc_ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    roc_ax.legend()
    pr_ax.plot(GRID, np.mean(precisions, axis=0), color="tab:blue", lw=2, label="mean over seeds")
    pr_ax.axhline(base_rate, color="k", ls="--", lw=1, label=f"base rate {base_rate:.3f}")
    pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    pr_ax.legend()

    first = predictions[predictions["seed"] == seeds[0]]
    bins = np.linspace(0, 1, 41)
    for label, color, name in [(0, "tab:gray", "not readmitted <30d"), (1, "tab:red", "readmitted <30d")]:
        dist_ax.hist(first.loc[first["y_true"] == label, "score"], bins=bins, alpha=0.6, color=color, density=True, label=name)
    dist_ax.set(xlabel="Predicted score", ylabel="Density", title=f"Score distribution by outcome (seed {seeds[0]})")
    dist_ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_overview.png", dpi=150)
    plt.close(fig)


def plot_metrics_across_seeds(metrics: pd.DataFrame) -> None:
    names = ["auc", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([metrics[n] for n in names], tick_labels=names, showmeans=True)
    rng = np.random.default_rng(0)
    for i, n in enumerate(names, start=1):
        ax.scatter(rng.normal(i, 0.04, len(metrics)), metrics[n], s=12, color="tab:blue", alpha=0.6, zorder=3)
    ax.set(ylabel="Test-set value", title=f"Baseline metrics across {len(metrics)} patient-level splits")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metrics_across_seeds.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(IN_DIR / "predictions.csv", usecols=["seed", "y_true", "score"])
    metrics = pd.read_csv(IN_DIR / "metrics_per_seed.csv")
    plot_overview(predictions)
    plot_metrics_across_seeds(metrics)
    logger.info("saved plots to %s", OUT_DIR)


if __name__ == "__main__":
    main()
