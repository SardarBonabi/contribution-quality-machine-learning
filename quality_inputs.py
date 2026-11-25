"""Representative model input checks; research under review, full code/data proprietary."""
import numpy as np
from pandas.api.types import is_numeric_dtype, is_complex_dtype
from quality_baseline import validate_repository_ids


def validate_model_inputs(train, evaluation, features):
    """Validate the numeric sample before fitting; NaN predictors remain allowed.

    Missing predictors are delegated to XGBoost's missing-value handling. Infinite
    values and implicit text/category conversion are rejected. Feature timing and
    target-derived leakage still require a point-in-time feature design.
    """
    if not features or len(features) != len(set(features)):
        raise ValueError("Select a nonempty list of unique feature names")
    if {"merged", "repository"}.intersection(features):
        raise ValueError("Exclude target and raw repository ID from features")
    for frame in (train, evaluation):
        if not frame.columns.is_unique or not {"merged", "repository", *features}.issubset(frame.columns):
            raise ValueError("Expected unique columns and all required model fields")
        validate_repository_ids(frame["repository"])
        if not frame["merged"].isin([0, 1]).all() or frame["merged"].nunique() != 2:
            raise ValueError("AUC requires both binary outcome classes")
        for name in features:
            values = frame[name]
            if not is_numeric_dtype(values.dtype) or is_complex_dtype(values.dtype):
                raise ValueError("Model features must be real numeric values")
            if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError("Model features must not contain infinite values")
