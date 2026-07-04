"""
Probability-of-default modeling on the German Credit dataset (Statlog, n=1000).

Target: bad credit (credit_risk == 0), prevalence 30%.

Two models, both evaluated with out-of-fold predictions from 5-fold
stratified CV (no test-set leakage, every applicant scored exactly once
by a model that never saw them):
  1. Logistic regression on one-hot features  — the interpretable scorecard
  2. Histogram gradient boosting              — the performance benchmark

Metrics are the ones credit risk teams actually use:
  ROC-AUC, Gini (= 2*AUC - 1), KS statistic (max separation between
  cumulative good/bad score distributions), Brier score (calibration).

Threshold selection uses the dataset's canonical asymmetric cost matrix:
approving a bad loan costs 5, rejecting a good one costs 1 — so the
operating point is NOT 0.5 and NOT accuracy-optimal, it is cost-optimal.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RNG = 7
COST_FN, COST_FP = 5.0, 1.0   # missing a defaulter costs 5x a lost good customer

df = pd.read_csv("data/german_credit.csv")
y = (df["credit_risk"] == 0).astype(int)          # 1 = bad credit (default)
X = df.drop(columns=["credit_risk"])
cat = X.select_dtypes(exclude="number").columns.tolist()
num = X.select_dtypes("number").columns.tolist()
print(f"n = {len(df)}, default rate = {y.mean():.1%}, "
      f"{len(cat)} categorical + {len(num)} numeric features")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

logit = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
])

gbm = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OrdinalEncoder(), cat),
        ("num", "passthrough", num),
    ])),
    ("clf", HistGradientBoostingClassifier(
        categorical_features=list(range(len(cat))),
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=RNG)),
])


def ks_stat(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def cost_optimal(y_true, p):
    """Expected cost per applicant, minimized over thresholds."""
    ts = np.linspace(0.01, 0.99, 197)
    costs = []
    for t in ts:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        costs.append((COST_FN * fn + COST_FP * fp) / len(y_true))
    i = int(np.argmin(costs))
    return ts[i], costs[i]


def report(name, model):
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    t, c = cost_optimal(y, p)
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"\n{name}")
    print(f"  ROC-AUC = {auc:.3f}   Gini = {2*auc-1:.3f}   "
          f"KS = {ks_stat(y, p):.3f}   Brier = {brier_score_loss(y, p):.3f}")
    print(f"  cost-optimal threshold = {t:.2f}  ->  {c:.3f} cost/applicant  "
          f"(vs {COST_FN*y.mean():.3f} for approve-everyone)")
    print(f"  at that threshold: catches {tp}/{tp+fn} defaulters "
          f"({tp/(tp+fn):.0%} recall), rejects {fp} good applicants")
    return p


p_logit = report("Logistic regression (scorecard)", logit)
p_gbm = report("Gradient boosting (HistGBM)", gbm)

# interpretability: top risk drivers from the scorecard
logit.fit(X, y)
names = logit["prep"].get_feature_names_out()
coefs = logit["clf"].coef_.ravel()
top = np.argsort(np.abs(coefs))[::-1][:6]
print("\ntop scorecard risk drivers (|coefficient|, sign = direction of risk):")
for i in top:
    print(f"  {coefs[i]:+.2f}  {names[i]}")
