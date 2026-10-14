import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import roc_curve

from rai.config import PRIMARY_ATTR, TARGET_COL
from rai.models import build_logreg, feature_columns

CONSTRAINT = "equalized_odds"
OBJECTIVE = "balanced_accuracy_score"


def fit_and_predict_equalized_odds(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    # same model as the baseline; group-specific (randomised) thresholds are fitted on val only
    model = build_logreg(train)
    model.fit(train[feature_columns(train)], train[TARGET_COL])
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints=CONSTRAINT,
        objective=OBJECTIVE,
        prefit=True,
        predict_method="predict_proba",
    )
    optimizer.fit(
        val[feature_columns(val)], val[TARGET_COL], sensitive_features=val[PRIMARY_ATTR]
    )
    test_scores = model.predict_proba(test[feature_columns(test)])[:, 1]
    y_pred = optimizer.predict(
        test[feature_columns(test)], sensitive_features=test[PRIMARY_ATTR], random_state=seed
    )
    return test_scores, np.asarray(y_pred).astype(int)


def fit_and_predict_global_threshold(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    # control: one shared threshold maximising balanced accuracy on val (same objective, no group information)
    model = build_logreg(train)
    model.fit(train[feature_columns(train)], train[TARGET_COL])
    val_scores = model.predict_proba(val[feature_columns(val)])[:, 1]
    fpr, tpr, thresholds = roc_curve(val[TARGET_COL], val_scores)
    threshold = thresholds[np.argmax(tpr - fpr)]
    test_scores = model.predict_proba(test[feature_columns(test)])[:, 1]
    return test_scores, (test_scores >= threshold).astype(int)
