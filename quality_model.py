"""Representative reconstruction of the quality model.

Feature names, hyperparameters, and residual formula are illustrative choices,
not the original research specification. No research performance is reproduced.
Caller supplies an explicitly held-out evaluation set and training-only features.
"""
import pandas as pd
from quality_adjustment import illustrative_acceptance_residual

from quality_baseline import fit_repository_baseline, validate_repository_ids


def evaluate_merge_model(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str]):
    """Compare XGBoost with a repository base rate learned only on train.

    merged is binary. Features must be known before the merge decision; exclude
    post-decision fields and target-derived leakage. Split by time or repository
    to match the intended generalization question, and tune without this holdout.
    """
    if not features or len(features) != len(set(features)):
        raise ValueError("Select a nonempty list of unique feature names")
    for frame in (train, evaluation):
        if not frame.columns.is_unique:
            raise ValueError("Input columns must be unique")
        if not {"merged", "repository", *features}.issubset(frame.columns):
            raise ValueError("Input is missing required columns")
        validate_repository_ids(frame["repository"])
        if not frame["merged"].isin([0, 1]).all() or frame["merged"].nunique() != 2:
            raise ValueError("AUC requires both binary outcome classes")
    if {"merged", "repository"}.intersection(features):
        raise ValueError("Exclude target and raw repository ID from features")
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        objective="binary:logistic", eval_metric="logloss", random_state=42,
    )
    model.fit(train[features], train["merged"])
    probability = model.predict_proba(evaluation[features])[:, 1]
    baseline_model = fit_repository_baseline(train)
    baseline = baseline_model.predict(evaluation["repository"])
    metrics = {
        "model_auc": roc_auc_score(evaluation["merged"], probability),
        "repository_baseline_auc": roc_auc_score(evaluation["merged"], baseline),
    }
    return model, metrics, probability

