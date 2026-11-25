"""Representative reconstruction of the quality model.

Feature names, hyperparameters, and residual formula are illustrative choices,
not the original research specification. No research performance is reproduced.
Caller supplies an explicitly held-out evaluation set and training-only features.
"""
import pandas as pd
from quality_adjustment import illustrative_acceptance_residual

from quality_baseline import fit_repository_baseline
from quality_inputs import validate_model_inputs


def evaluate_merge_model(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str]):
    """Compare XGBoost with a repository base rate learned only on train.

    merged is binary. Features must be known before the merge decision; exclude
    post-decision fields and target-derived leakage. Split by time or repository
    to match the intended generalization question, and tune without this holdout.
    """
    validate_model_inputs(train, evaluation, features)
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

