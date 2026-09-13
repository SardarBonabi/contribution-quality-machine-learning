"""Representative four-cluster structural segmentation.

This compact example does not recover the original feature set or cluster labels.
The fitted preprocessing is retained for consistent assignment of new records.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def fit_archetypes(training_features):
    """Use selected nonnegative, finite structural counts, e.g. files changed."""
    values = np.asarray(training_features, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4:
        raise ValueError("Expected at least four contribution rows")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Structural counts must be finite and nonnegative")
    if len(np.unique(values, axis=0)) < 4:
        raise ValueError("Need at least four distinct feature profiles")
    pipeline = make_pipeline(
        FunctionTransformer(np.log1p), StandardScaler(),
        KMeans(n_clusters=4, n_init=10, random_state=42),
    )
    labels = pipeline.fit_predict(values)
    return pipeline, labels
# Interpret clusters from their feature profiles; do not invent archetype names
# or equate cluster membership with contribution quality.
