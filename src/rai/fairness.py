import numpy as np
import pandas as pd

from rai.config import GROUP_COL

METRICS = ["selection_rate", "tpr", "fpr", "precision"]
GAP_NAMES = {
    "selection_rate": "demographic_parity_gap",
    "tpr": "tpr_gap",
    "fpr": "fpr_gap",
    "precision": "precision_gap",
}


def encode_groups(df: pd.DataFrame, group_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    grouped = df.groupby(group_cols, sort=True)
    codes = grouped.ngroup().to_numpy()
    labels = [
        " / ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        for key in grouped.size().index
    ]
    return codes, labels


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )


def group_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    w = np.ones(len(codes)) if weights is None else weights
    t = y_true.astype(float)
    p = y_pred.astype(float)

    def total(x: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=w * x, minlength=n_groups)

    n = total(np.ones(len(codes)))
    positives = total(t)
    selected = total(p)
    true_pos = total(t * p)
    false_pos = total(p * (1 - t))
    return {
        "n": n,
        "positives": positives,
        "selection_rate": _ratio(selected, n),
        "tpr": _ratio(true_pos, positives),
        "fpr": _ratio(false_pos, n - positives),
        "precision": _ratio(true_pos, selected),
    }


def rates_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    codes, labels = encode_groups(df, group_cols)
    rates = group_rates(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), codes, len(labels))
    table = pd.DataFrame(rates, index=pd.Index(labels, name="group"))
    return table.astype({"n": int, "positives": int})


def gaps_from_rates(rates: dict[str, np.ndarray]) -> pd.DataFrame:
    out = {}
    for metric, name in GAP_NAMES.items():
        frame = pd.DataFrame(np.atleast_2d(rates[metric]))
        out[name] = frame.max(axis=1) - frame.min(axis=1)
    gaps = pd.DataFrame(out)
    gaps["equalized_odds_gap"] = gaps[["tpr_gap", "fpr_gap"]].max(axis=1)
    return gaps


def gaps(table: pd.DataFrame) -> pd.Series:
    return gaps_from_rates({m: table[m].to_numpy() for m in METRICS}).iloc[0]


def bootstrap_by_patient(
    df: pd.DataFrame, group_cols: list[str], n_boot: int, seed: int
) -> dict[str, np.ndarray]:
    codes, labels = encode_groups(df, group_cols)
    patient_codes, patients = pd.factorize(df[GROUP_COL])
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    n_patients = len(patients)
    rng = np.random.default_rng(seed)

    draws = {m: np.empty((n_boot, len(labels))) for m in METRICS}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients)
        rates = group_rates(
            y_true, y_pred, codes, len(labels), weights=counts[patient_codes].astype(float)
        )
        for metric in METRICS:
            draws[metric][b] = rates[metric]
    return draws
