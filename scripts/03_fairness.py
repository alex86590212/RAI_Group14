import argparse
import logging

import pandas as pd

from rai.config import N_BOOTSTRAP, RESULTS_DIR, SEEDS
from rai.evaluate import (
    GROUP_SPECS,
    bootstrap_tables,
    per_seed_tables,
    plot_gaps_across_seeds,
    plot_group_metrics,
    summarise_across_seeds,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    in_dir = RESULTS_DIR / args.name
    out_dir = in_dir / "fairness"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(in_dir / "predictions.csv")
    seeds = sorted(predictions["seed"].unique())
    logger.info("%s: %d seeds, %d test predictions", args.name, len(seeds), len(predictions))

    group_df, gap_df = per_seed_tables(predictions)
    group_summary, gap_summary = summarise_across_seeds(group_df, gap_df)
    group_df.to_csv(out_dir / "group_metrics_per_seed.csv", index=False)
    gap_df.to_csv(out_dir / "gaps_per_seed.csv", index=False)
    group_summary.to_csv(out_dir / "group_metrics_summary.csv", index=False)
    gap_summary.to_csv(out_dir / "gaps_summary.csv", index=False)

    boot_seed = SEEDS[0] if SEEDS[0] in seeds else seeds[0]
    group_ci, gap_ci = bootstrap_tables(predictions, boot_seed, N_BOOTSTRAP)
    group_ci.to_csv(out_dir / "group_metrics_bootstrap_ci.csv", index=False)
    gap_ci.to_csv(out_dir / "gaps_bootstrap_ci.csv", index=False)

    for attribute in GROUP_SPECS:
        plot_group_metrics(group_ci, attribute, boot_seed, out_dir / f"group_metrics_{attribute}.png")
    plot_gaps_across_seeds(gap_df, out_dir / "gaps_across_seeds.png")

    gap_cols = ["attribute", "demographic_parity_gap_mean", "demographic_parity_gap_std",
                "tpr_gap_mean", "tpr_gap_std", "fpr_gap_mean", "fpr_gap_std",
                "equalized_odds_gap_mean", "equalized_odds_gap_std"]
    logger.info("Gaps across seeds:\n%s", gap_summary[gap_cols].round(3).to_string(index=False))
    logger.info("Gaps with bootstrap 95%% CI (seed %d):\n%s", boot_seed, gap_ci.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
