import numpy as np
import pandas as pd
import pytest

from rai.config import GROUP_COL, SEEDS
from rai.split import assert_no_patient_overlap, patient_split


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    patients = np.arange(1000)
    counts = rng.integers(1, 6, size=len(patients))
    return pd.DataFrame({GROUP_COL: np.repeat(patients, counts)})


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_no_patient_overlap(df, seed):
    train, val, test = patient_split(df, seed)
    assert_no_patient_overlap(train, val, test)


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_every_row_in_exactly_one_split(df, seed):
    train, val, test = patient_split(df, seed)
    assert len(train) + len(val) + len(test) == len(df)
    assert len(set(train.index) | set(val.index) | set(test.index)) == len(df)


def test_split_proportions(df):
    train, val, test = patient_split(df, 0)
    assert len(train) / len(df) == pytest.approx(0.6, abs=0.05)
    assert len(val) / len(df) == pytest.approx(0.2, abs=0.05)
    assert len(test) / len(df) == pytest.approx(0.2, abs=0.05)


def test_same_seed_is_reproducible_and_seeds_differ(df):
    a, _, _ = patient_split(df, 0)
    a_again, _, _ = patient_split(df, 0)
    b, _, _ = patient_split(df, 1)
    assert a.index.equals(a_again.index)
    assert not a.index.equals(b.index)


def test_overlap_check_detects_leak(df):
    with pytest.raises(ValueError):
        assert_no_patient_overlap(df.iloc[:10], df.iloc[:10], df.iloc[10:20])
