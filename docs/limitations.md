# Limitations and Interpretation Boundaries

This project is an analytical prototype built from a fixed portfolio-data
snapshot. The outputs are useful for demonstrating lineage, feature
engineering, model evaluation, and dashboard design, but they should not be
read as validated production business decisions.

## Synthetic or portfolio-data limitations

The repository does not establish that the raw records represent a complete
production population, a random sample, or real customer behavior. Values may
be synthetic, simulated, anonymized, or assembled for portfolio demonstration.
Counts, costs, model mix, latency, and relationships should therefore be
interpreted as properties of this included snapshot. They should not be
generalized to an actual customer base or used as an external benchmark
without provenance and independent validation.

## Proxy-based client disengagement label

The churn model is trained on a rule-derived `ChurnSignal`: the last request is
more than 45 days old and maximum quota utilization is below 20%. This is a
proxy for disengagement, not a historical cancellation, renewal, or
independently observed churn outcome. A high `ChurnRiskScore` means similarity
to that rule within this dataset; it does not establish that a client will
cancel. Thresholds, calibration, temporal validation, and observed outcomes
would be needed before operational use.

## Incomplete satisfaction labels

Only requests with a recorded `SatisfactionScore` train the satisfaction
regressor. Unrated requests receive `SatisfactionPredicted`, which is a model
estimate and must not be confused with collected feedback. Group averages can
also be based on different numbers of rated requests. Improving response
capture and reporting coverage is necessary before comparing satisfaction
across all clients, models, or features.

## Potential selection bias

Feedback is unlikely to be missing completely at random: users who respond may
differ from users who do not, and clients with more engagement may generate
more observations. The rated subset can therefore over- or under-represent
experience. The same concern applies to request logs, which may omit failed
ingestion, offline activity, or uninstrumented channels. Random train/test
splits do not remove this sampling risk.

## Isolation Forest interpretation

Isolation Forest is an unsupervised relative outlier detector over cost, total
tokens, and latency, configured with 2% contamination. `IsAnomaly` identifies
observations that are unusual in the fitted feature space; it does not identify
fraud, defects, or root causes. `AnomalyScore` is a model score whose higher
values indicate greater relative abnormality after the project’s sign
inversion. Results depend on feature scaling, the contamination setting, and
the included snapshot, so flagged rows require domain review.

## Forecast uncertainty

The forecasting stage extrapolates historical daily patterns for a 30-day
horizon. Prophet provides lower and upper interval columns for cost and token
forecasts, while the ARIMA request forecast is evaluated with a historical
holdout but does not make the future certain. Intervals depend on model
assumptions and may not capture product launches, outages, pricing changes,
seasonality shifts, customer churn, or structural breaks. A holdout metric is
evidence about one historical slice, not a guarantee of future accuracy.

## Analytical prototype versus production system

The current workflow is a reproducible local pipeline of Python scripts, CSV
files, serialized models, Power BI assets, and a local presentation layer. It
does not by itself provide production ingestion, orchestration, authentication,
authorization, secrets management, schema evolution, data retention,
observability, alerting, model registry governance, disaster recovery, or
service-level guarantees. CSV handoffs can also create concurrency and
freshness issues at larger scale. Production use would require an agreed data
contract, warehouse/lakehouse or service architecture, controlled deployment,
time-aware validation, drift monitoring, retraining policy, and human review
for consequential decisions.
