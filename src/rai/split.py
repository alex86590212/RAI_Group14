import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from rai.config import GROUP_COL, TEST_SIZE, VAL_SIZE


def patient_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_val_pos, test_pos = next(outer.split(df, groups=df[GROUP_COL]))
    train_val = df.iloc[train_val_pos]
    test = df.iloc[test_pos]

    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=seed)
    train_pos, val_pos = next(inner.split(train_val, groups=train_val[GROUP_COL]))
    return train_val.iloc[train_pos], train_val.iloc[val_pos], test


def assert_no_patient_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_ids, val_ids, test_ids = (set(x[GROUP_COL]) for x in (train, val, test))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("patients appear in more than one split")
