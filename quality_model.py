"""Representative reconstruction of the quality model.

Feature names, hyperparameters, and residual formula are illustrative choices,
not the original research specification. No research performance is reproduced.
Caller supplies an explicitly held-out evaluation set and training-only features.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def evaluate_merge_model(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str]):
    """Compare XGBoost with a repository base rate learned only on train.

    merged is binary. Features must be known before the merge decision; exclude
    post-decision fields and target-derived leakage. Split by time or repository
    to match the intended generalization question, and tune without this holdout.
    """
    for frame in (train, evaluation):
        if not frame["merged"].isin([0, 1]).all() or frame["merged"].nunique() != 2:
            raise ValueError("AUC requires both binary outcome classes")
    if {"merged", "repository"}.intersection(features):
        raise ValueError("Exclude target and raw repository ID from features")
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        objective="binary:logistic", eval_metric="logloss", random_state=42,
    )
    model.fit(train[features], train["merged"])
    probability = model.predict_proba(evaluation[features])[:, 1]
    base_rates = train.groupby("repository")["merged"].mean()
    baseline = evaluation["repository"].map(base_rates).fillna(train["merged"].mean())
    metrics = {
        "model_auc": roc_auc_score(evaluation["merged"], probability),
        "repository_baseline_auc": roc_auc_score(evaluation["merged"], baseline),
    }
    return model, metrics, probability


def illustrative_acceptance_residual(observed, predicted_probability):
    """One possible adjustment, not the recovered research quality metric.

    Positive means accepted more than predicted. Use out-of-sample probabilities;
    interpretation requires calibration checks and is not direct code correctness.
    """
    observed, predicted = np.asarray(observed), np.asarray(predicted_probability)
    if observed.shape != predicted.shape or not np.isin(observed, [0, 1]).all():
        raise ValueError("Expected equally shaped binary outcomes and probabilities")
    if not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 1)).any():
        raise ValueError("Invalid probabilities")
    return observed - predicted
