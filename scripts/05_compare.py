import argparse
import logging

import pandas as pd

from rai.config import RESULTS_DIR
from rai.evaluate import (
    COMPARED_GAPS,
    COMPARED_PERFORMANCE,
    METRICS,
    paired_summary,
    plot_gap_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--new", default="intervention")
    args = parser.parse_args()
    BASELINE_DIR = RESULTS_DIR / args.reference
    INTERVENTION_DIR = RESULTS_DIR / args.new
    OUT_DIR = RESULTS_DIR / "comparison" / f"{args.new}_vs_{args.reference}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    performance = paired_summary(
        pd.read_csv(BASELINE_DIR / "metrics_per_seed.csv"),
        pd.read_csv(INTERVENTION_DIR / "metrics_per_seed.csv"),
        COMPARED_PERFORMANCE,
        keys=[],
    )
    performance.to_csv(OUT_DIR / "performance.csv", index=False)

    base_gaps = pd.read_csv(BASELINE_DIR / "fairness" / "gaps_per_seed.csv")
    new_gaps = pd.read_csv(INTERVENTION_DIR / "fairness" / "gaps_per_seed.csv")
    gap_comparison = paired_summary(base_gaps, new_gaps, COMPARED_GAPS, keys=["attribute"])
    gap_comparison.to_csv(OUT_DIR / "gaps.csv", index=False)

    base_groups = pd.read_csv(BASELINE_DIR / "fairness" / "group_metrics_per_seed.csv")
    new_groups = pd.read_csv(INTERVENTION_DIR / "fairness" / "group_metrics_per_seed.csv")
    group_comparison = paired_summary(base_groups, new_groups, METRICS, keys=["attribute", "group"])
    group_comparison.to_csv(OUT_DIR / "group_metrics.csv", index=False)

    plot_gap_comparison(base_gaps, new_gaps, (args.reference, args.new), OUT_DIR / "gaps_comparison.png")

    show = ["baseline_mean", "baseline_std", "new_mean", "new_std", "diff_mean", "diff_ci_low", "diff_ci_high", "share_seeds_lower"]
    logger.info("Performance (test, 20 seeds):\n%s", performance[["metric", *show]].round(3).to_string(index=False))
    logger.info("Gaps:\n%s", gap_comparison[["attribute", "metric", *show]].round(3).to_string(index=False))
    race = group_comparison[group_comparison["attribute"] == "race"]
    logger.info("Race groups:\n%s", race[["group", "metric", "baseline_mean", "new_mean", "diff_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
