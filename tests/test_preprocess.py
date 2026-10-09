import numpy as np
import pandas as pd
import pytest

from rai.config import TARGET_COL
from rai.preprocess import EXCLUDED_DISCHARGE_DISPOSITION_IDS, preprocess_data


@pytest.fixture
def raw():
    return pd.DataFrame(
        {
            "discharge_disposition_id": [1, 11, 13, 14, 15, 17, 19, 20, 21, 1, 1],
            "readmitted": ["<30", "<30", "NO", ">30", "<30", "NO", "NO", "NO", "NO", ">30", "NO"],
            "race": ["Caucasian", None, "Asian", "Asian", "Asian", None, "Other", "Other", "Other", "Hispanic", "AfricanAmerican"],
            "gender": ["Female", "Male", "Male", "Male", "Unknown/Invalid", "Male", "Male", "Male", "Male", "Female", "Male"],
            "weight": [np.nan] * 11,
            "max_glu_serum": [None, "Norm", None, None, None, None, None, None, None, ">200", None],
            "A1Cresult": [None] * 10 + [">7"],
        }
    )


def test_death_and_hospice_rows_are_excluded(raw):
    out = preprocess_data(raw)
    assert not out["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_DISPOSITION_IDS).any()


def test_transfers_are_kept(raw):
    out = preprocess_data(raw)
    assert {15, 17}.isdisjoint(EXCLUDED_DISCHARGE_DISPOSITION_IDS)
    assert out["discharge_disposition_id"].isin([17]).any()


def test_target_is_one_only_for_under_30(raw):
    out = preprocess_data(raw)
    assert (out[TARGET_COL] == (out["readmitted"] == "<30").astype(int)).all()
    assert set(out[TARGET_COL]) <= {0, 1}


def test_unknown_gender_rows_dropped(raw):
    out = preprocess_data(raw)
    assert "Unknown/Invalid" not in set(out["gender"])


def test_missing_values_are_labelled_and_weight_dropped(raw):
    out = preprocess_data(raw)
    assert "weight" not in out.columns
    assert out[["race", "max_glu_serum", "A1Cresult"]].notna().all().all()
    assert "Unknown" in set(out["race"])
    assert "Not measured" in set(out["max_glu_serum"])
