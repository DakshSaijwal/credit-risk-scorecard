# Data — Credit Risk Scorecard

## `german_credit.csv`

**Source:** Statlog (German Credit Data), originally from the UCI Machine
Learning Repository, loaded here from a GitHub mirror in CSV form.

- **1000 rows**, one per loan applicant.
- **21 columns**: 20 features (financial history, employment status,
  housing, loan purpose, loan amount/duration, existing credits, etc.) plus
  the target `credit_risk`.
- **Target encoding:** `credit_risk == 1` → good credit, `credit_risk == 0`
  → bad credit (default). `train.py` treats `credit_risk == 0` as the
  positive class (1 = bad/default) since that's the outcome the model is
  trying to catch.
- **Class balance:** 700 good (70%) / 300 bad (30%) — matches the canonical
  Statlog distribution, confirmed against the dataset documentation before
  use.
- **Feature types:** a mix of categorical (e.g. checking account status,
  purpose, housing) and numeric (e.g. loan duration, amount, age) columns,
  split automatically in `train.py` via dtype.

## Canonical cost matrix

The Statlog documentation for this dataset specifies a standard 5:1 cost
matrix: classifying a bad applicant as good (false negative / approving a
future defaulter) costs 5 units; classifying a good applicant as bad (false
positive / rejecting a creditworthy applicant) costs 1 unit. `train.py`
uses this matrix to choose a cost-optimal decision threshold instead of the
default 0.5 cutoff — see the main `README.md` for why that matters.

## If you swap in a different dataset

`train.py`'s column split (categorical vs. numeric via `select_dtypes`) and
pipeline structure are dataset-agnostic. You'd need to update:
- the target column name and positive-class encoding,
- the cost matrix constants (`COST_FN`, `COST_FP`) to match your actual
  business costs, if known.
