import logging

import pandas as pd

from rai.config import POSITIVE_LABEL, TARGET_COL

logger = logging.getLogger(__name__)

EXCLUDED_DISCHARGE_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    ammount = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_DISPOSITION_IDS)].copy()
    logger.info("Excluded %d death/hospice rows (%d -> %d)", ammount - len(df), ammount, len(df))
    df[TARGET_COL] =(df["readmitted"] == POSITIVE_LABEL).astype(int)
    df["race"] = df["race"].fillna("Unknown")
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Not measured")
    df["A1Cresult"] = df["A1Cresult"].fillna("Not measured")
    n_before_gender = len(df)
    df = df[df["gender"] != "Unknown/Invalid"].copy()
    logger.info("Excluded %d rows with unknown gender", n_before_gender - len(df))
    df = df.drop(columns=["weight"])
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/diabetes_130.csv")
    df = preprocess_data(df)
    print(df.head())