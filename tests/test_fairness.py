import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from rai.evaluate import per_seed_tables
from rai.fairness import (
    bootstrap_by_patient,
    encode_groups,
    gaps,
    group_rates,
    rates_table,
)


@pytest.fixture
def hand_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["a"] * 4 + ["b"] * 4,
            "y_true": [1, 1, 0, 0, 1, 0, 0, 0],
            "y_pred": [1, 0, 1, 0, 1, 1, 1, 1],
        }
    )


def random_frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1])
    y_true = rng.binomial(1, np.where(groups == "c", 0.2, 0.1))
    y_pred = rng.binomial(1, np.where(groups == "b", 0.4, 0.25))
    return pd.DataFrame(
        {
            "patient_nbr": rng.integers(0, n // 2, n),
            "group": groups,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def test_group_rates_match_hand_computation(hand_frame):
    table = rates_table(hand_frame, ["group"])
    assert table.loc["a", ["n", "positives"]].tolist() == [4, 2]
    assert table.loc["a", "selection_rate"] == 0.5
    assert table.loc["a", "tpr"] == 0.5
    assert table.loc["a", "fpr"] == 0.5
    assert table.loc["a", "precision"] == 0.5
    assert table.loc["b", "selection_rate"] == 1.0
    assert table.loc["b", "tpr"] == 1.0
    assert table.loc["b", "fpr"] == 1.0
    assert table.loc["b", "precision"] == 0.25


def test_gaps_match_hand_computation(hand_frame):
    result = gaps(rates_table(hand_frame, ["group"]))
    assert result["demographic_parity_gap"] == 0.5
    assert result["tpr_gap"] == 0.5
    assert result["fpr_gap"] == 0.5
    assert result["equalized_odds_gap"] == 0.5
    assert result["precision_gap"] == 0.25


def test_identical_groups_have_zero_gaps():
    df = pd.DataFrame(
        {
            "group": ["a"] * 4 + ["b"] * 4,
            "y_true": [1, 0, 1, 0] * 2,
            "y_pred": [1, 0, 0, 0] * 2,
        }
    )
    result = gaps(rates_table(df, ["group"]))
    assert (result == 0).all()


def test_gaps_agree_with_fairlearn():
    df = random_frame()
    result = gaps(rates_table(df, ["group"]))
    kwargs = {"y_true": df["y_true"], "y_pred": df["y_pred"], "sensitive_features": df["group"]}
    assert result["demographic_parity_gap"] == pytest.approx(demographic_parity_difference(**kwargs))
    assert result["equalized_odds_gap"] == pytest.approx(equalized_odds_difference(**kwargs))


def test_group_without_positives_gives_nan_tpr_and_is_ignored_in_gaps():
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "y_true": [1, 0, 0, 0],
            "y_pred": [1, 0, 1, 0],
        }
    )
    table = rates_table(df, ["group"])
    assert np.isnan(table.loc["b", "tpr"])
    assert gaps(table)["tpr_gap"] == 0


def test_multi_column_labels_are_joined():
    df = pd.DataFrame({"race": ["x", "x", "y"], "gender": ["F", "M", "F"]})
    _, labels = encode_groups(df, ["race", "gender"])
    assert labels == ["x / F", "x / M", "y / F"]


def test_unit_weights_reproduce_unweighted_rates():
    df = random_frame(500)
    codes, labels = encode_groups(df, ["group"])
    args = (df["y_true"].to_numpy(), df["y_pred"].to_numpy(), codes, len(labels))
    plain = group_rates(*args)
    weighted = group_rates(*args, weights=np.ones(len(df)))
    for key in plain:
        np.testing.assert_allclose(plain[key], weighted[key])


def test_bootstrap_interval_brackets_estimate_and_is_reproducible():
    df = random_frame()
    table = rates_table(df, ["group"])
    draws = bootstrap_by_patient(df, ["group"], n_boot=200, seed=1)
    low, high = np.percentile(draws["selection_rate"], [2.5, 97.5], axis=0)
    assert ((low <= table["selection_rate"]) & (table["selection_rate"] <= high)).all()
    again = bootstrap_by_patient(df, ["group"], n_boot=200, seed=1)
    np.testing.assert_array_equal(draws["tpr"], again["tpr"])


def test_smaller_groups_have_wider_bootstrap_intervals():
    df = random_frame(6000)
    draws = bootstrap_by_patient(df, ["group"], n_boot=300, seed=2)
    width = np.ptp(np.percentile(draws["selection_rate"], [2.5, 97.5], axis=0), axis=0)
    assert width[2] > width[0]


def test_per_seed_tables_cover_every_seed_attribute_and_group():
    rng = np.random.default_rng(0)
    n = 600
    frames = []
    for seed in (0, 1):
        frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "patient_nbr": rng.integers(0, 300, n),
                    "race": rng.choice(["A", "B", "C"], n),
                    "gender": rng.choice(["Female", "Male"], n),
                    "y_true": rng.binomial(1, 0.2, n),
                    "y_pred": rng.binomial(1, 0.3, n),
                }
            )
        )
    group_df, gap_df = per_seed_tables(pd.concat(frames, ignore_index=True))
    assert len(group_df) == 2 * (3 + 2 + 6)
    assert len(gap_df) == 2 * 3
    assert set(gap_df["attribute"]) == {"race", "gender", "race_gender"}
