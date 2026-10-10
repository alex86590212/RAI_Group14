import argparse
import logging

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from rai.config import (
    GROUP_COL,
    PRIMARY_ATTR,
    RESULTS_DIR,
    SECONDARY_ATTR,
    SEEDS,
    TARGET_COL,
)
from rai.data import load_raw
from rai.intervention import (
    fit_and_predict_equalized_odds,
    fit_and_predict_global_threshold,
)
from rai.preprocess import preprocess_data
from rai.split import assert_no_patient_overlap, patient_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

METHODS = {
    "equalized_odds": (fit_and_predict_equalized_odds, "intervention"),
    "global_threshold": (fit_and_predict_global_threshold, "control_global_threshold"),
}
METRICS = ["auc", "precision", "recall", "f1"]


def evaluate_seed(seed: int, df: pd.DataFrame, method) -> tuple[dict, pd.DataFrame]:
    train, val, test = patient_split(df, seed)
    assert_no_patient_overlap(train, val, test)
    test_scores, y_pred = method(train, val, test, seed)

    y_true = test[TARGET_COL].to_numpy()
    metrics = {
        "seed": seed,
        "auc": roc_auc_score(y_true, test_scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    predictions = pd.DataFrame(
        {
            "seed": seed,
            "encounter_id": test["encounter_id"].to_numpy(),
            GROUP_COL: test[GROUP_COL].to_numpy(),
            PRIMARY_ATTR: test[PRIMARY_ATTR].to_numpy(),
            SECONDARY_ATTR: test[SECONDARY_ATTR].to_numpy(),
            "y_true": y_true,
            "score": test_scores,
            "y_pred": y_pred,
        }
    )
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="equalized_odds")
    args = parser.parse_args()
    method, name = METHODS[args.method]
    OUT_DIR = RESULTS_DIR / name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = preprocess_data(load_raw())

    all_metrics, all_predictions = [], []
    for seed in SEEDS:
        metrics, predictions = evaluate_seed(seed, df, method)
        logger.info("seed %d: auc=%.3f f1=%.3f recall=%.3f", seed, metrics["auc"], metrics["f1"], metrics["recall"])
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_DIR / "metrics_per_seed.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_DIR / "predictions.csv", index=False)

    for name in METRICS:
        logger.info("%s: %.3f +- %.3f", name, metrics_df[name].mean(), metrics_df[name].std())


if __name__ == "__main__":
    main()
