import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rai.audit import (
    leakage_screen,
    missingness_by_group,
    naive_split_patient_overlap,
    numeric_means_by_group,
    positive_rate_by_group,
    share_by_group,
)
from rai.config import (
    NON_FEATURE_COLS,
    PRIMARY_ATTR,
    RESULTS_DIR,
    SECONDARY_ATTR,
    SEEDS,
    TARGET_COL,
)
from rai.data import load_raw
from rai.preprocess import preprocess_data
from rai.split import patient_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "audit"
UTILISATION_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df = preprocess_data(raw)

    logger.info("Positive rate overall: %.4f (n=%d)", df[TARGET_COL].mean(), len(df))
    for name, cols in {
        PRIMARY_ATTR: PRIMARY_ATTR,
        SECONDARY_ATTR: SECONDARY_ATTR,
        "intersection": [PRIMARY_ATTR, SECONDARY_ATTR],
    }.items():
        table = positive_rate_by_group(df, cols)
        table.to_csv(OUT_DIR / f"positive_rate_{name}.csv", index=False)
        logger.info("\n%s", table.to_string(index=False))

    race_rates = positive_rate_by_group(df, PRIMARY_ATTR)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(race_rates[PRIMARY_ATTR], race_rates["rate"], yerr=race_rates["ci95"], capsize=4)
    ax.axhline(df[TARGET_COL].mean(), color="k", linestyle="--", linewidth=1)
    ax.set_ylabel("30-day readmission rate")
    ax.set_title("Positive rate by race (95% CI, dashed = overall)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "positive_rate_race.png", dpi=150)

    missing = missingness_by_group(raw, PRIMARY_ATTR)
    missing.to_csv(OUT_DIR / "missingness_by_race.csv")
    logger.info("\nMissingness by race (raw data):\n%s", missing.round(3).to_string())

    overlaps = [naive_split_patient_overlap(df, seed) for seed in SEEDS]
    logger.info(
        "Naive row-level split: %.1f%% of test encounters belong to a patient also in train",
        100 * sum(overlaps) / len(overlaps),
    )

    train, val, _ = patient_split(df, SEEDS[0])
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    leakage = leakage_screen(train, val, features)
    leakage.to_csv(OUT_DIR / "leakage_screen.csv", index=False)
    logger.info("\nSingle-feature AUC (top 10):\n%s", leakage.head(10).to_string(index=False))

    numeric_means_by_group(df, PRIMARY_ATTR, UTILISATION_COLS).to_csv(OUT_DIR / "utilisation_by_race.csv")
    for col in ("discharge_disposition_id", "A1Cresult", "admission_source_id"):
        share_by_group(df, PRIMARY_ATTR, col).to_csv(OUT_DIR / f"share_{col}_by_race.csv")
    positive_rate_by_group(df, [PRIMARY_ATTR, "discharge_disposition_id"]).to_csv(
        OUT_DIR / "positive_rate_race_by_discharge.csv", index=False
    )


if __name__ == "__main__":
    main()
