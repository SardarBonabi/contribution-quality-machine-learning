"""PPML exposure specification for contribution extent; estimator omitted."""
import numpy as np


def prepare_extent_sample(panel):
    """Keep positive-PR developer-weeks and create a log exposure offset.

    total_lines_changed remains the outcome. Do not replace it with a rate and
    claim an equivalent OLS model. The offset coefficient must be constrained to 1.
    """
    columns = ["total_lines_changed", "pull_requests"]
    if panel[columns].isna().any().any() or panel[columns].lt(0).any().any():
        raise ValueError("Counts must be observed and nonnegative")
    result = panel.loc[panel["pull_requests"].gt(0)].copy()
    result["log_pr_exposure"] = np.log(result["pull_requests"])
    return result


SPECIFICATION = {
    "outcome": "total_lines_changed",
    "offset": "log_pr_exposure",
    "offset_coefficient": 1,
    "treatment": "italy_x_ban",
    "controls": ["working_days"],
    "fixed_effects": ["developer", "week"],
    "cluster": "developer",
}
# E[total_lines_changed | X, PRs] = PRs * exp(developer_FE + week_FE
#                                         + beta * italy_x_ban + controls)
# exp(beta) is the multiplicative effect on expected lines per pull request,
# conditional on contributing. It is not the effect on participation itself.
