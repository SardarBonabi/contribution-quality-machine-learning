# Sample code and provenance

This release presents selected research workflows without distributing the full proprietary implementations.

| File | Relationship to the research |
|---|---|
| quality_model.py | Representative reconstruction of the XGBoost approach. Hyperparameters and residual adjustment are illustrative. |
| contribution_archetypes.py | Reconstructed four-cluster workflow; no empirical cluster names or membership are supplied. |
| extent_model.py | Representative preparation and declarative specification of the manuscript's exposure model. |

## What was refactored

The public samples replace repeated country/partition scripts with parameterized functions; separate I/O from analytical logic; use descriptive variables and explicit model contracts; and remove embedded paths, credentials, debugging output, and confidential records. Illustrative safeguards and settings added in this release are not claims about the historical implementation.

## Reading the examples

The samples are Python source intended to be read on GitHub. There is no installation or replication requirement. External library imports indicate the methods being illustrated, not a locked production environment. No private data, real identifiers, model artifacts, or empirical prediction files are included. The methods notes explain where details are omitted and distinguish study results from illustrative code.

Original code and research data remain proprietary. No open-source license is granted by this showcase.
