# Client disengagement risk scoring using an operational proxy outcome

## Intended use

This model is an analytical prioritization signal for customer-success and
product teams. It ranks clients for review based on observed usage behavior in
the supplied snapshot. It must not be used as a claim that a customer has
churned, as an automated customer decision, or as a substitute for contacting
the client.

## Proxy target definition

The positive proxy outcome (`ChurnSignal = 1`) is assigned when both conditions
are true:

- `DaysSinceLastRequest > 45`
- `QuotaUtilizationPct < 20`

This is **client disengagement risk scoring using an operational proxy
outcome**, not confirmed customer churn. The label is derived from the same
usage snapshot used for scoring and describes inactivity plus low quota
utilization only.

## Features

The logistic regression uses:

- `DaysSinceLastRequest`
- `AvgMonthlyCostTrend`
- `SatisfactionTrend`
- `SuccessRateLast30Days`
- `SuccessRateOverall`
- `QuotaUtilizationPct`
- `PremiumFeatureUsagePct`

Missing feature values are filled with zero for this client-level model.
The existing `ChurnRiskScore` output column remains a normalized 0-100 model
probability.

## Validation strategy

The evaluation writes artifacts under `outputs/evaluation/`. When both classes
have at least two clients, it uses a fixed, stratified 25% holdout
(`random_state=42`). The scaler and logistic model are fitted on the training
partition only before holdout metrics are calculated. The explicit proxy rule
is reported as a baseline. If the snapshot cannot support a two-class holdout,
the evaluation reports that limitation rather than adding synthetic labels.

The evaluation includes class counts and balance, ROC AUC and PR AUC when both
classes are present, precision, recall, F1, a confusion matrix, threshold
analysis, and logistic coefficients.

## Metrics

`outputs/evaluation/churn_evaluation.csv` contains the exact holdout metrics
from the latest pipeline run. `churn_class_balance.csv` records the positive
and negative counts. `churn_confusion_matrix.png` visualizes model errors, and
`churn_threshold_analysis.csv` shows precision/recall/F1 across thresholds.

## Known limitations

- The target is an operational proxy and is not a measured customer outcome.
- Because the label is defined from recency and quota utilization, strong
  performance is partly a consequence of that rule and should not be read as
  evidence of real churn prediction.
- The data is a single historical snapshot at client grain; it does not
  establish whether a client later renewed, cancelled, expanded, or returned.
- Class imbalance can make accuracy misleading and can make holdout metrics
  unstable when the positive class is small. Precision, recall, F1, ROC AUC,
  and PR AUC should be interpreted with the reported class counts.
- The score is not calibrated for a production intervention threshold.
- Satisfaction and usage fields may reflect data capture and measurement bias.

## Why real churn labels would be preferable

Confirmed outcomes such as cancellation, non-renewal, or a defined period of
post-contract inactivity would align the target with the business question.
They would support temporal backtesting, calibration, intervention-cost
analysis, and monitoring for false positives and false negatives. Those labels
are not present in this project, so this model should remain an exploratory
operational signal.

## Satisfaction prediction

The satisfaction model is a random-forest regression model trained only on
requests with an observed `SatisfactionScore`. The target is not imputed for
training. The model uses request attributes such as model, feature, region,
subscription tier, tokens, latency, cost, success, time, and request type.
Preprocessing is fitted inside the training pipeline, and the fixed seed is
`42`.

The pipeline compares the random forest with a training-set mean baseline and
reports MAE, RMSE, and R2 on a reproducible 20% holdout. It also reports
performance for model, client, region, feature, and request-type groups when
the holdout group has at least 10 rated requests. Predictions are constrained
to the valid satisfaction range of 1–5; this is a validity check and not a
claim that the model is calibrated.

`outputs/evaluation/satisfaction_coverage.csv` distinguishes labeled
requests from previously unrated requests and reports coverage overall and by
model, client, region, feature, and request type. The observed score remains
in `SatisfactionScore`. Model output is in `SatisfactionPredicted`, and
`SatisfactionPredictionPopulation` identifies whether an observed score was
available or the row was previously unrated. The model's prediction for an
unrated request is an estimated satisfaction score, not actual user feedback.

`outputs/evaluation/satisfaction_evaluation.csv` contains baseline/model
holdout comparison, sufficiently sized group-level performance, and separate
population summaries for observed-score rows and previously unrated rows.
Metrics cannot be computed for unrated rows because no observed target exists.

### Selection bias and uncertainty

Satisfaction labels are incomplete (and may be volunteered by users who are
more or less engaged), so the rated subset may not represent all requests.
Coverage differences across models, clients, regions, features, and request
types can therefore create selection bias. Holdout and cross-validation
metrics quantify agreement with observed scores in rated requests only; they
do not establish accuracy for previously unrated requests or future feedback.
Small group samples produce unstable estimates, which is why group metrics
are emitted only at the documented minimum. Predictions should be treated as
uncertain prioritization signals and should not be presented as measured user
sentiment.
