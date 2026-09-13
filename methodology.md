# Contribution extent and contextual quality

> **Research status:** The research is under review. Full code and data remain proprietary; public code illustrates the general workflow only.

## Two analytical questions

The causal design asks how changes in AI availability affect public contributions. The predictive model asks how well repository, maintainer, and contribution features discriminate merge outcomes. Strong predictive performance alone cannot answer the causal question.

## Extent model

The manuscript uses Poisson pseudo-maximum likelihood with total lines changed as the outcome and pull-request count as exposure. The log of positive pull-request count enters as an offset with coefficient fixed to one. The fitted mean is PR_count multiplied by an exponential function of developer effects, week effects, treatment, and controls.

This identifies an effect on expected lines per pull request conditional on contributing. It is not equivalent to an OLS regression on a precomputed ratio. Developer-weeks with zero pull requests cannot enter this log-exposure specification. The manuscript reports an incidence-rate ratio of 0.522 for extent, approximately a 48% decrease during the ban.

## Predictive quality measurement

The completed XGBoost analysis achieved 0.82 AUC versus 0.61 for a repository-level base rate. The public example illustrates a baseline calculated on training data and an explicitly supplied holdout. These are representative engineering choices; the original split design, hyperparameters, calibration results, and exact adjustment formula are not supplied in this release.

AUC measures ranking, not the proportion classified correctly. Calibration matters if probabilities are used in a difficulty adjustment. A residual of observed acceptance minus predicted acceptance is shown only as an illustrative adjustment; it is not asserted to be the proprietary research formula. Merge outcomes reflect context as well as quality and should not be labeled direct correctness tests.

### Training context and the causal question

**The XGBoost quality model used a separate sample from the pre-treatment period**. Its repository-, maintainer-, and contribution-level features support measurement of acceptance difficulty before the access interruption.

This timing separates learning acceptance patterns from the subsequent treatment-period response. It does not, by itself, establish calibration or rule out every form of leakage. Exact split boundaries, entity overlap, and tuning procedures are not supplied, so the public example should not be treated as a reconstruction of those details.

| Analytical component | Question answered | Reported evidence |
|---|---|---|
| **XGBoost merge-probability model** | How well do contextual features rank merge outcomes? | **0.82 AUC versus 0.61 repository baseline** |
| **Difficulty-adjusted quality metric** | How can acceptance context be accounted for in quality measurement? | Completed measurement work; proprietary formula withheld |
| Quasi-experimental outcome analysis | How did outcomes respond to the access interruption? | Separate causal estimates, subject to the comparison-group assumptions |

The AUC comparison evaluates predictive discrimination. It is not the treatment effect, and the illustrative residual in the public code is not asserted to produce the manuscript's accuracy estimate.

## Contribution archetypes

The completed research identified four structural archetypes using K-Means. The sample illustrates log transformation, scaling, and four-cluster fitting, retaining preprocessing for new observations. Feature choices and cluster names are not reconstructed as empirical facts.

## Interpretation

The manuscript's accuracy measure changed by approximately 1.1% during the ban, with p < 0.1 in the reported discussion. A small estimate is not proof of no change. The causal estimate and predictive evaluation describe different analytical outputs; this release does not claim the displayed residual formula generated the manuscript's accuracy result.
