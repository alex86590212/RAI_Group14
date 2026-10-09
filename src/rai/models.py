import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rai.config import NON_FEATURE_COLS, PRIMARY_ATTR, SECONDARY_ATTR, TARGET_COL

EXCLUDED_FROM_MODEL = NON_FEATURE_COLS + [PRIMARY_ATTR, SECONDARY_ATTR]
ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    # columns of df not in EXCLUDED_FROM_MODEL
    return [col for col in df.columns if col not in EXCLUDED_FROM_MODEL]


def build_logreg(feature_df: pd.DataFrame) -> Pipeline:
    features = feature_columns(feature_df)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(feature_df[c]) and c not in ID_CODE_COLS]
    categorical = [c for c in features if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    # pick the threshold that maximises F1 on the validation set
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return float(thresholds[np.argmax(f1)])


def fit_and_predict(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, float]:
    # fit on train, tune threshold on val scores, return (val_scores, test_scores, threshold)
    model = build_logreg(train)
    model.fit(train[feature_columns(train)], train[TARGET_COL])
    val_scores = model.predict_proba(val[feature_columns(val)])[:, 1]
    test_scores = model.predict_proba(test[feature_columns(test)])[:, 1]
    threshold = tune_threshold(val[TARGET_COL], val_scores)
    return val_scores, test_scores, threshold