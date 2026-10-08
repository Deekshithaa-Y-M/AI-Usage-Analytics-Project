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
