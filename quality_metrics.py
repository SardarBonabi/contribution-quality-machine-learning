"""Representative holdout metrics; research under review, full code/data proprietary.

No empirical performance is reproduced. Lower Brier scores indicate more
accurate probability forecasts; Brier alone does not establish calibration.
"""
import numpy as np
from quality_adjustment import illustrative_acceptance_residual


def brier_score(observed, probability):
    """Mean squared probability error, with explicit binary/vector validation."""
    residual = illustrative_acceptance_residual(observed, probability)
    return float(np.mean(np.square(residual)))


def compare_probabilities(observed, model_probability, baseline_probability):
    """Compare discrimination and probability error on the same held-out rows."""
    model_brier = brier_score(observed, model_probability)
    baseline_brier = brier_score(observed, baseline_probability)
    from sklearn.metrics import roc_auc_score
    return {
        "model_auc": roc_auc_score(observed, model_probability),
        "repository_baseline_auc": roc_auc_score(observed, baseline_probability),
        "model_brier": model_brier,
        "repository_baseline_brier": baseline_brier,
    }
