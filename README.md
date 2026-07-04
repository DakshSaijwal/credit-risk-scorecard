# Credit Risk Scorecard (German Credit dataset)

I wanted to build a probability-of-default model the way a credit risk team
would actually score one, not the way a Kaggle notebook does. So this repo
trains two models on the German Credit data and compares them on the metrics
lenders care about: ranking (AUC, Gini, KS), calibration (Brier), and cost
at a sensible operating point instead of a blind 0.5 cutoff.

The two models are:

1. A logistic regression scorecard. This is the boring, interpretable
   baseline that most banks still run in production.
2. A histogram gradient boosting model, as the "can something fancier beat
   it" benchmark.

## The data

Statlog (German Credit Data), 1000 loan applicants, 21 columns. The features
are the usual credit stuff: checking account status, credit history,
employment, housing, loan purpose, amount, duration, age, and so on. Target
is `credit_risk` (1 = good, 0 = bad/default). The split is 700 good to 300
bad, so it's imbalanced but not brutally so.

More detail on the columns and the source is in `data/README.md`.

## How the models are evaluated

Everything is scored with 5-fold stratified CV using `cross_val_predict`, so
every applicant gets one prediction from a model that never trained on them.
No peeking, no test-set leakage.

A couple of choices worth calling out:

- The logistic model gets one-hot encoded categoricals, scaled numerics, and
  `class_weight="balanced"` to deal with the 70/30 split.
- The GBM uses scikit-learn's native categorical support, so ordinal-encoded
  columns get flagged as categorical instead of blowing up into one-hot.
- Thresholds are picked on cost, not accuracy. The dataset ships with a
  standard 5:1 cost matrix (approving a defaulter is 5x worse than turning
  away a good customer), and I use it to find the threshold that minimizes
  expected cost per applicant. Accuracy-optimal or 0.5 would be the wrong
  call here.

## Results

| Model | ROC-AUC | Gini | Cost/applicant |
|---|---|---|---|
| Logistic regression (scorecard) | 0.787 | 0.574 | 0.513 |
| Gradient boosting (HistGBM) | 0.776 | 0.552 | 0.578 |

KS and Brier get printed when you run `train.py`; I left them out of the
table so I'm not copying numbers by hand, but they're right there in the
output.

The interesting part: the plain logistic scorecard beats the gradient
boosting model on both AUC and cost. That surprises people, but it's a pretty
normal outcome on a small tabular dataset like this. With only 1000 rows and
strong categorical signal, there just isn't enough data for the GBM's extra
flexibility to earn its keep, and the linear structure fits this problem
well. It's also the answer a lending business would rather hear: the model
you can fully explain to a credit committee is the one that performs at least
as well.

## Running it

```bash
pip install -r requirements.txt
python3 train.py
```

## Reading the scorecard

At the end, `train.py` prints the six logistic coefficients with the largest
magnitude, with the sign telling you which direction pushes risk (positive =
more likely to default). That's the whole reason to keep a linear model
around: you can point at any coefficient and explain it. A tree ensemble
doesn't give you that.

## Caveats

A few things I'd want to fix before trusting this anywhere real:

- 1000 rows is small. I'm reporting point estimates, but the AUC probably
  swings a fair bit across folds, so a mean ± std would be more honest.
- There's no time dimension here. Real credit models drift as the applicant
  population shifts, and a single CV snapshot tells you nothing about that.
- The 5:1 cost ratio is the dataset's convention, not a real bank's loss
  given default. Swap in actual numbers if you have them.
