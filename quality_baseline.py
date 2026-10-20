"""Representative repository baseline; research under review, full code/data proprietary.

This sample separates training statistics from holdout scoring. It does not
reproduce research results or distribute the original model implementation.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from numbers import Real
from math import isfinite

import pandas as pd


def validate_repository_ids(repositories: pd.Series) -> None:
    """Distinguish missing identity from a valid repository unseen in training."""
    if repositories.isna().any():
        raise ValueError("Repository identifiers must not be missing")


@dataclass(frozen=True)
class RepositoryBaseline:
    rates: Mapping[object, float]
    global_rate: float

    def __post_init__(self):
        """Snapshot caller-owned rates and reject invalid probability state."""
        rates = dict(self.rates)
        for rate in [self.global_rate, *rates.values()]:
            if isinstance(rate, bool) or not isinstance(rate, Real) or not isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("Baseline probabilities must be finite and between zero and one")
        if pd.Series(list(rates), dtype=object).isna().any():
            raise ValueError("Repository identifiers must not be missing")
        object.__setattr__(self, "rates", MappingProxyType(rates))
        object.__setattr__(self, "global_rate", float(self.global_rate))

    def predict(self, repositories: pd.Series) -> pd.Series:
        """Keep row order/index; unseen repositories use the training global rate.

        Scoring accepts identifiers only, so evaluation labels cannot change
        the fitted statistics. Missing identifiers are not unseen repositories.
        """
        validate_repository_ids(repositories)
        # Object identifiers prevent categorical mapping from producing a categorical
        # probability series that cannot accept the global fallback value.
        return repositories.astype(object).map(self.rates).fillna(self.global_rate).astype(float)


def fit_repository_baseline(train: pd.DataFrame) -> RepositoryBaseline:
    """Fit repository means and a row-weighted fallback using training data only."""
    if not train.columns.is_unique:
        raise ValueError("Training columns must be unique")
    if not {"repository", "merged"}.issubset(train.columns):
        raise ValueError("Training data requires repository and merged columns")
    if train.empty or not train["merged"].isin([0, 1]).all():
        raise ValueError("Expected nonempty training data with binary outcomes")
    validate_repository_ids(train["repository"])
    rates = train.groupby("repository", observed=True, sort=False)["merged"].mean()
    return RepositoryBaseline(
        rates=MappingProxyType(rates.to_dict()),
        global_rate=float(train["merged"].mean()),
    )
