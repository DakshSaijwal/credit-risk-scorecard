# Credit Risk Scorecard: Probability of Default on the German Credit Dataset

Two probability-of-default models — a logistic regression scorecard (the
industry-standard interpretable baseline) and a gradient boosting model (the
performance benchmark) — evaluated the way credit risk teams actually
evaluate models: honest out-of-fold predictions, ranking metrics (AUC, Gini,
KS), calibration (Brier score), and a cost-optimal decision threshold instead
of the default 0.5 cutoff.

## Dataset

**Statlog (German Credit Data)**, 1000 loan applicants, 21 features
(financial history, employment, housing, loan purpose, etc.), target
`credit_risk` (1 = good credit, 0 = bad credit / default). Class split:
700 good / 300 bad. See `data/README.md` for source and feature notes.

## Method

- **5-fold stratified cross-validation** with `cross_val_predict` — every
  applicant is scored exactly once, by a model that never saw them during
  training. No test-set leakage.
- **Logistic regression**: one-hot encoded categoricals + standardized
  numerics, `class_weight="balanced"` to counter the 70/30 imbalance.
- **HistGradientBoostingClassifier**: native categorical support (ordinal-
  encoded, flagged as categorical to the model — no one-hot explosion).
- **Cost-optimal threshold**, not accuracy-optimal or 0.5. The dataset's
  canonical cost matrix says approving a bad loan (false negative) costs
  **5x** as much as rejecting a good applicant (false positive) — so the
  decision threshold is chosen to minimize expected cost per applicant, not
  to maximize accuracy.

## Results

| Model | ROC-AUC | Gini | Cost/applicant (optimal threshold) |
|---|---|---|---|
| **Logistic regression (scorecard)** | **0.787** | 0.574 | **0.513** |
| Gradient boosting (HistGBM) | 0.776 | 0.552 | 0.578 |

*(KS and Brier scores print when you run `train.py` — add them here from
your own run if you want the table complete.)*

**Key finding:** the simpler, fully interpretable logistic scorecard
outperforms gradient boosting on both AUC and cost per applicant, despite
GBM's greater modeling flexibility. This is a legitimate and fairly common
result on small, tabular datasets with strong categorical signal and only
1000 rows — there isn't enough data for GBM's extra capacity to pay off,
and the scorecard's linear structure happens to match the data well. It's
also the more useful outcome for a lending business: an interpretable model
with equal-or-better performance is what regulators and credit committees
actually want.

## Running it

```bash
pip install -r requirements.txt
python3 train.py
```

## Top risk drivers (scorecard odds ratios)

`train.py` prints the six largest logistic regression coefficients by
magnitude, with sign indicating direction of risk (positive = increases
default probability). This is the interpretability payoff of using a linear
scorecard: every coefficient is directly explainable to a credit committee,
which a gradient-boosted tree ensemble is not.

## Limitations / next steps

- 1000 rows is small for a production credit model — variance across CV
  folds should be reported (e.g. AUC ± std across the 5 folds), not just
  the point estimate.
- No temporal validation — real credit models must be checked for
  performance drift over time (population shift), which a single
  cross-validated snapshot can't reveal.
- Cost matrix (5:1) is the dataset's canonical convention, not a real
  bank's actual loss given default — worth stating explicitly if asked.
