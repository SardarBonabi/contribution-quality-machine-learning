"""Representative four-cluster structural segmentation.

This compact example does not recover the original feature set or cluster labels.
The fitted preprocessing is retained for consistent assignment of new records.
Research is under review; full research code and data remain proprietary.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from structural_features import structural_counts, log_structural_counts


def fit_archetypes(training_features):
    """Use selected nonnegative, finite structural counts, e.g. files changed."""
    values = structural_counts(training_features)
    if values.ndim != 2 or values.shape[0] < 4:
        raise ValueError("Expected at least four contribution rows")
    if len(np.unique(values, axis=0)) < 4:
        raise ValueError("Need at least four distinct feature profiles")
    pipeline = make_pipeline(
        FunctionTransformer(log_structural_counts), StandardScaler(),
        KMeans(n_clusters=4, n_init=10, random_state=42),
    )
    labels = pipeline.fit_predict(values)
    return pipeline, labels
# Interpret clusters from their feature profiles; do not invent archetype names
# or equate cluster membership with contribution quality.
