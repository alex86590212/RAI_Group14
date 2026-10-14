from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from rai.config import PRIMARY_ATTR, SECONDARY_ATTR
from rai.fairness import (
    METRICS,
    bootstrap_by_patient,
    gaps,
    gaps_from_rates,
    rates_table,
)

GROUP_SPECS = {
    PRIMARY_ATTR: [PRIMARY_ATTR],
    SECONDARY_ATTR: [SECONDARY_ATTR],
    f"{PRIMARY_ATTR}_{SECONDARY_ATTR}": [PRIMARY_ATTR, SECONDARY_ATTR],
}
PLOTTED_GAPS = ["demographic_parity_gap", "tpr_gap", "fpr_gap"]


def per_seed_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows, gap_rows = [], []
    for seed, seed_df in predictions.groupby("seed"):
        for attribute, cols in GROUP_SPECS.items():
            table = rates_table(seed_df, cols)
            group_rows.append(table.reset_index().assign(seed=seed, attribute=attribute))
            gap_rows.append({"seed": seed, "attribute": attribute, **gaps(table).to_dict()})
    return pd.concat(group_rows, ignore_index=True), pd.DataFrame(gap_rows)


def summarise_across_seeds(
    group_df: pd.DataFrame, gap_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["n", "positives", *METRICS]
    group_summary = group_df.groupby(["attribute", "group"])[group_cols].agg(["mean", "std"])
    gap_cols = [c for c in gap_df.columns if c not in ("seed", "attribute")]
    gap_summary = gap_df.groupby("attribute")[gap_cols].agg(["mean", "std", "min", "max"])
    for summary in (group_summary, gap_summary):
        summary.columns = ["_".join(col) for col in summary.columns]
    return group_summary.reset_index(), gap_summary.reset_index()


def bootstrap_tables(
    predictions: pd.DataFrame, seed: int, n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_df = predictions[predictions["seed"] == seed]
    group_rows, gap_rows = [], []
    for attribute, cols in GROUP_SPECS.items():
        table = rates_table(seed_df, cols)
        draws = bootstrap_by_patient(seed_df, cols, n_boot, seed)
        for metric in METRICS:
            low, high = np.nanpercentile(draws[metric], [2.5, 97.5], axis=0)
            group_rows.append(
                pd.DataFrame(
                    {
                        "attribute": attribute,
                        "group": table.index,
                        "n": table["n"].to_numpy(),
                        "positives": table["positives"].to_numpy(),
                        "metric": metric,
                        "estimate": table[metric].to_numpy(),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            )
        point = gaps(table)
        boot_gaps = gaps_from_rates(draws)
        for name in boot_gaps.columns:
            low, high = boot_gaps[name].quantile([0.025, 0.975])
            gap_rows.append(
                {
                    "attribute": attribute,
                    "gap": name,
                    "estimate": point[name],
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.concat(group_rows, ignore_index=True), pd.DataFrame(gap_rows)


def plot_group_metrics(ci: pd.DataFrame, attribute: str, seed: int, path: Path) -> None:
    data = ci[ci["attribute"] == attribute]
    groups = list(data["group"].unique())
    sizes = data.drop_duplicates("group").set_index("group").loc[groups]
    labels = [f"{g} (n={n}, pos={p})" for g, n, p in zip(groups, sizes["n"], sizes["positives"])]
    y = np.arange(len(groups))

    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(4 * len(METRICS), 0.45 * len(groups) + 2), sharey=True
    )
    for ax, metric in zip(axes, METRICS):
        sub = data[data["metric"] == metric].set_index("group").loc[groups]
        err = [sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]]
        ax.errorbar(sub["estimate"], y, xerr=err, fmt="o", capsize=3, color="tab:blue")
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(f"{attribute}: per-group metrics, patient-level bootstrap 95% CI (seed {seed})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gaps_across_seeds(gap_df: pd.DataFrame, path: Path) -> None:
    attributes = list(GROUP_SPECS)
    fig, axes = plt.subplots(1, len(PLOTTED_GAPS), figsize=(5 * len(PLOTTED_GAPS), 4.5))
    rng = np.random.default_rng(0)
    for ax, name in zip(axes, PLOTTED_GAPS):
        values = [gap_df.loc[gap_df["attribute"] == a, name] for a in attributes]
        ax.boxplot(values, tick_labels=attributes, showmeans=True)
        for i, v in enumerate(values, start=1):
            ax.scatter(rng.normal(i, 0.04, len(v)), v, s=12, color="tab:blue", alpha=0.6, zorder=3)
        ax.set_title(name)
        ax.set_ylabel("max - min across groups")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Fairness gaps across patient-level splits")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


COMPARED_GAPS = ["demographic_parity_gap", "tpr_gap", "fpr_gap", "precision_gap", "equalized_odds_gap"]
COMPARED_PERFORMANCE = ["auc", "precision", "recall", "f1"]


def paired_summary(baseline: pd.DataFrame, other: pd.DataFrame, columns: list[str], keys: list[str]) -> pd.DataFrame:
    # per-seed differences (other - baseline) are paired because both runs use the same patient splits
    merged = baseline.merge(other, on=["seed", *keys], suffixes=("_base", "_new"))
    rows = []
    for key, part in merged.groupby(keys) if keys else [((), merged)]:
        key = key if isinstance(key, tuple) else (key,)
        for col in columns:
            diff = part[f"{col}_new"] - part[f"{col}_base"]
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std() / np.sqrt(len(diff))
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "metric": col,
                    "baseline_mean": part[f"{col}_base"].mean(),
                    "baseline_std": part[f"{col}_base"].std(),
                    "new_mean": part[f"{col}_new"].mean(),
                    "new_std": part[f"{col}_new"].std(),
                    "diff_mean": diff.mean(),
                    "diff_ci_low": diff.mean() - half,
                    "diff_ci_high": diff.mean() + half,
                    "share_seeds_lower": float((diff < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_gap_comparison(base_gaps: pd.DataFrame, new_gaps: pd.DataFrame, labels: tuple[str, str], path: Path) -> None:
    attributes = list(GROUP_SPECS)
    fig, axes = plt.subplots(1, len(PLOTTED_GAPS), figsize=(5 * len(PLOTTED_GAPS), 4.5))
    rng = np.random.default_rng(0)
    colors = ["tab:blue", "tab:orange"]
    for ax, name in zip(axes, PLOTTED_GAPS):
        for offset, gaps_df, color, label in zip((-0.2, 0.2), (base_gaps, new_gaps), colors, labels):
            values = [gaps_df.loc[gaps_df["attribute"] == a, name] for a in attributes]
            positions = np.arange(1, len(attributes) + 1) + offset
            box = ax.boxplot(values, positions=positions, widths=0.3, showmeans=True, patch_artist=True)
            for patch in box["boxes"]:
                patch.set(facecolor=color, alpha=0.3)
            for pos, v in zip(positions, values):
                ax.scatter(rng.normal(pos, 0.03, len(v)), v, s=10, color=color, alpha=0.6, zorder=3)
            ax.plot([], [], color=color, label=label)
        ax.set_xticks(np.arange(1, len(attributes) + 1), attributes)
        ax.set_title(name)
        ax.set_ylabel("max - min across groups")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Fairness gaps across patient-level splits: {labels[0]} vs {labels[1]}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
