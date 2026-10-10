import numpy as np
import pandas as pd

from rai.evaluate import paired_summary
from rai.intervention import (
    fit_and_predict_equalized_odds,
    fit_and_predict_global_threshold,
)


def make_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    race = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1])
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x - 2 + 0.5 * (race == "B"))))).astype(int)
    return pd.DataFrame(
        {
            "x": x,
            "cat": rng.choice(["u", "v"], size=n),
            "race": race,
            "gender": rng.choice(["F", "M"], size=n),
            "patient_nbr": np.arange(n),
            "encounter_id": np.arange(n),
            "readmitted": "NO",
            "is_under_30": y,
        }
    )


def test_methods_return_binary_predictions_for_every_row():
    train, val, test = make_frame(3000, 0), make_frame(1500, 1), make_frame(1500, 2)
    for method in (fit_and_predict_equalized_odds, fit_and_predict_global_threshold):
        scores, y_pred = method(train, val, test, 0)
        assert len(scores) == len(y_pred) == len(test)
        assert set(np.unique(y_pred)) <= {0, 1}


def test_equalized_odds_is_reproducible_for_a_seed():
    train, val, test = make_frame(3000, 0), make_frame(1500, 1), make_frame(1500, 2)
    _, first = fit_and_predict_equalized_odds(train, val, test, 3)
    _, second = fit_and_predict_equalized_odds(train, val, test, 3)
    assert (first == second).all()


def test_equalized_odds_reduces_group_tpr_gap_on_validation_like_data():
    train, val, test = make_frame(6000, 0), make_frame(6000, 1), make_frame(6000, 2)
    _, y_pred = fit_and_predict_equalized_odds(train, val, test, 0)
    y_true = test["is_under_30"].to_numpy()
    tprs = [y_pred[(test["race"] == g) & (y_true == 1)].mean() for g in ("A", "B", "C")]
    assert max(tprs) - min(tprs) < 0.15


def test_paired_summary_reports_difference_and_share_lower():
    base = pd.DataFrame({"seed": range(5), "gap": [0.5, 0.4, 0.6, 0.5, 0.5]})
    new = pd.DataFrame({"seed": range(5), "gap": [0.3, 0.3, 0.7, 0.3, 0.3]})
    out = paired_summary(base, new, ["gap"], keys=[]).iloc[0]
    assert np.isclose(out["diff_mean"], -0.12)
    assert out["share_seeds_lower"] == 0.8
    assert out["diff_ci_low"] < out["diff_mean"] < out["diff_ci_high"]
