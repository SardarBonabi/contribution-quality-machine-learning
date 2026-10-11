"""Representative acceptance adjustment; research under review.

Full research code and data are proprietary. This illustrative residual is not
the recovered research quality metric and reproduces no research performance.
"""
import numpy as np


def illustrative_acceptance_residual(observed, predicted_probability):
    """Return a positional, one-dimensional observed-minus-predicted vector.

    Positive means accepted more than predicted. Use out-of-sample probabilities;
    interpretation requires calibration checks and is not direct code correctness.
    Callers must align observations and predictions before passing them here.
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted_probability)
    if observed.ndim != 1 or predicted.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("Expected equally sized one-dimensional vectors")
    if observed.size == 0:
        raise ValueError("Expected at least one observation")
    # Numeric and Boolean arrays are accepted; strings, objects and complex
    # values fail explicitly instead of being silently coerced or broadcast.
    if observed.dtype.kind not in "biuf" or predicted.dtype.kind not in "biuf":
        raise ValueError("Expected real numeric outcomes and probabilities")
    if not np.isin(observed, [0, 1]).all():
        raise ValueError("Expected binary outcomes")
    if not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 1)).any():
        raise ValueError("Invalid probabilities")
    return observed.astype(float) - predicted.astype(float)
