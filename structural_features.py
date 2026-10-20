"""Representative structural preprocessing; research under review.

Full research code and data remain proprietary. Feature selection and cluster
interpretation are outside this illustrative numerical input contract.
"""
import numpy as np


def structural_counts(features):
    """Validate both training and scoring matrices without imputing missing counts.

    Columns must have the same meaning and order at scoring time. This array
    interface cannot detect swapped columns; callers own feature alignment.
    """
    values = np.asarray(features)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Expected a nonempty matrix with at least one feature")
    if values.dtype.kind not in "iuf":
        raise ValueError("Structural counts must be real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64, copy=True)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Structural counts must be finite and nonnegative")
    return values


def log_structural_counts(features):
    """Apply the same checked log transform during fitting and prediction."""
    return np.log1p(structural_counts(features))
