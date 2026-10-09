import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from rai.config import GROUP_COL, TARGET_COL, TEST_SIZE


def positive_rate_by_group(df: pd.DataFrame, group_cols: str | list[str]) -> pd.DataFrame:
    out = df.groupby(group_cols)[TARGET_COL].agg(n="size", positives="sum", rate="mean")
    out["ci95"] = 1.96 * np.sqrt(out["rate"] * (1 - out["rate"]) / out["n"])
    return out.reset_index()


def missingness_by_group(df_raw: pd.DataFrame, group_col: str) -> pd.DataFrame:
    groups = df_raw[group_col].fillna("Unknown")
    cols = df_raw.drop(columns=[group_col])
    out = cols.isna().groupby(groups).mean().T
    out["overall"] = cols.isna().mean()
    return out[out["overall"] > 0].sort_values("overall", ascending=False)


def naive_split_patient_overlap(df: pd.DataFrame, seed: int) -> float:
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=seed)
    return test[GROUP_COL].isin(train[GROUP_COL]).mean()


def leakage_screen(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    base_rate = train[TARGET_COL].mean()
    rows = []
    for col in features:
        if pd.api.types.is_numeric_dtype(train[col]):
            score = val[col].fillna(train[col].median())
        else:
            rate_by_level = train.groupby(col)[TARGET_COL].mean()
            score = val[col].map(rate_by_level).fillna(base_rate)
        auc = roc_auc_score(val[TARGET_COL], score)
        rows.append({"feature": col, "auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def numeric_means_by_group(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_col)[cols].mean()


def share_by_group(df: pd.DataFrame, group_col: str, col: str) -> pd.DataFrame:
    return pd.crosstab(df[group_col], df[col], normalize="index")
