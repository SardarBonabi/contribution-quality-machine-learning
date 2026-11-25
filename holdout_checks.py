"""Representative holdout integrity checks; research under review, full code/data proprietary."""


def validate_observation_split(train, evaluation, observation_key):
    """Require unique, observed, disjoint record IDs when a key is supplied.

    Repository overlap may be intentional for a temporal holdout; use a stable
    observation ID, not the repository ID or a row position. This check does not
    establish time ordering or rule out leakage through derived features.
    """
    if not isinstance(observation_key, str) or not observation_key.strip():
        raise ValueError("Supply an observation identifier column name")
    for frame in (train, evaluation):
        if observation_key not in frame.columns:
            raise ValueError("Observation identifier column is missing")
        values = frame[observation_key]
        if values.isna().any() or values.duplicated().any():
            raise ValueError("Observation identifiers must be observed and unique")
    if train[observation_key].isin(evaluation[observation_key]).any():
        raise ValueError("Training and evaluation observations overlap")
