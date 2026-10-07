# Data Dictionary

This dictionary describes the CSV contract used by the project. The raw tables are
stored in [`data/raw/`](../data/raw/). The pipeline preserves the raw request grain
in `fact_master.csv` and `fact_with_ml.csv`; aggregate files intentionally use
coarser grains.

## Conventions

- Types describe the intended pandas/CSV representation: `integer`, `decimal`,
  `boolean`, `date`, `datetime-like`, and `string`.
- A blank `SatisfactionScore` means that no satisfaction response was recorded for
  that request. It is not a zero or a negative rating.
- Blank descriptive attributes or keys indicate missing source data or a failed
  dimension match. The ETL validation checks critical keys and does not silently
  repair invalid relationships.
- Percent fields in aggregate outputs are expressed as percentages (for example,
  `98.5` means 98.5%), except where a field's name or source definition indicates
  a ratio.

## Raw tables

### `Fact_AI_Usage.csv`

**Grain:** one AI usage request/event.  
**Primary key:** `UsageID`.  
**Foreign keys:** `DateKey` -> `Dim_Date.DateKey`; `ClientKey` ->
`Dim_Client.ClientKey`; `ModelKey` -> `Dim_Model.ModelKey`; `FeatureKey` ->
`Dim_Feature.FeatureKey`; `RegionKey` -> `Dim_Region.RegionKey`;
`SubscriptionTierKey` -> `Dim_Subscription.SubscriptionTierKey`.

| Column | Type | Business meaning |
| --- | --- | --- |
| `UsageID` | integer | Unique request/event identifier. |
| `DateKey` | integer | Calendar key for the request date. |
| `TimeKey` | integer | HHMM-style request time key. |
| `ClientKey` | integer | Client dimension key. |
| `ModelKey` | integer | Model dimension key. |
| `FeatureKey` | integer | Feature dimension key. |
| `RegionKey` | integer | Region dimension key. |
| `SubscriptionTierKey` | integer | Subscription-tier key at the event. |
| `TokensInput`, `TokensOutput`, `TotalTokens` | integer | Input, output, and total token counts. |
| `LatencyMs` | integer/decimal | Request latency in milliseconds. |
| `CostUSD` | decimal | Recorded API cost in US dollars. |
| `SuccessFlag` | integer/boolean-like | 1 for a successful request and 0 for an unsuccessful request. |
| `SatisfactionScore` | decimal, nullable | Recorded user satisfaction rating when feedback exists. |
| `RequestType` | string | Request category, such as `Chat`. |

Critical fact keys and measures are expected to be non-null and non-negative.
The current quality report flags 348 `CostUSD` values above mean plus three
standard deviations as outliers; this is a warning, not an automatic deletion.

### `Dim_Date.csv`

**Grain:** one calendar date.  
**Primary key:** `DateKey`.  
**Foreign keys:** none; referenced by `Fact_AI_Usage.DateKey`.

| Column | Type | Business meaning |
| --- | --- | --- |
| `DateKey` | integer | YYYYMMDD-style calendar key. |
| `Date` | date | Calendar date. |
| `Year` | integer | Calendar year. |
| `Quarter` | string | Calendar quarter, such as `Q1`. |
| `Month` | integer | Month number within the year. |
| `MonthName` | string | Month label. |
| `WeekOfYear` | integer | Week number. |
| `DayOfWeek` | string | Day name. |
| `IsWeekend` | boolean | Whether the date falls on Saturday or Sunday. |

### `Dim_Client.csv`

**Grain:** one client/account.  
**Primary key:** `ClientKey`.  
**Foreign keys:** none; referenced by `Fact_AI_Usage.ClientKey`.

| Column | Type | Business meaning |
| --- | --- | --- |
| `ClientKey` | integer | Internal client dimension key. |
| `ClientID` | string | Business-facing client identifier. |
| `ClientName` | string | Display name for the client. |
| `Industry` | string | Client industry classification. |
| `CompanySize` | string | Company-size segment. |
| `OnboardingDate` | date | Date the client was onboarded. |
| `AccountManager` | string | Account-manager identifier. |
| `Status` | string | Client lifecycle status, such as `Active`. |

### `Dim_Model.csv`

**Grain:** one AI model/version available to usage events.  
**Primary key:** `ModelKey`.  
**Foreign keys:** none; referenced by `Fact_AI_Usage.ModelKey`.

| Column | Type | Business meaning |
| --- | --- | --- |
| `ModelKey` | integer | Internal model dimension key. |
| `ModelName` | string | Model display name. |
| `Provider` | string | Model provider. |
| `ModelFamily` | string | Broader model family. |
| `ContextWindow` | integer | Context-window capacity in tokens. |
| `CostPer1kInput`, `CostPer1kOutput` | decimal | Reference price per 1,000 input/output tokens. |
| `IsLatest` | boolean | Whether the model is marked as the latest version. |

### `Dim_Feature.csv`

**Grain:** one product/API feature.  
**Primary key:** `FeatureKey`.  
**Foreign keys:** none; referenced by `Fact_AI_Usage.FeatureKey`.

| Column | Type | Business meaning |
| --- | --- | --- |
| `FeatureKey` | integer | Internal feature dimension key. |
| `FeatureName` | string | Feature display name. |
| `FeatureCategory` | string | Functional feature category. |
| `IsPremium` | boolean | Whether the feature is designated premium. |

### `Dim_Region.csv`

**Grain:** one reporting region.  
**Primary key:** `RegionKey`.  
**Foreign keys:** none; referenced by `Fact_AI_Usage.RegionKey`.

| Column | Type | Business meaning |
| --- | --- | --- |
| `RegionKey` | integer | Internal region dimension key. |
| `RegionName` | string | Reporting region name. |
| `Country` | string | Country associated with the region. |
| `TimeZoneOffset` | decimal | Offset from UTC used for regional interpretation. |

### `Dim_Subscription.csv`

**Grain:** one subscription tier.  
**Primary key:** `SubscriptionTierKey`.  
**Foreign keys:** none; referenced by `Fact_AI_Usage.SubscriptionTierKey`.

| Column | Type | Business meaning |
| --- | --- | --- |
| `SubscriptionTierKey` | integer | Internal subscription-tier key. |
| `TierName` | string | Tier display name. |
| `MonthlyQuotaTokens` | integer | Monthly token quota used for utilization calculations. |
| `OverageRateMultiplier` | decimal | Cost multiplier applied after quota is exceeded. |
| `SupportLevel` | string | Support offering for the tier. |

## ETL-derived columns in `fact_master.csv`

The ETL left-joins the dimensions to the fact table and derives these request-grain
columns:

| Column(s) | Meaning |
| --- | --- |
| `Date`, `Year`, `Quarter`, `Month`, `MonthName`, `WeekOfYear`, `DayOfWeek`, `IsWeekend` | Calendar attributes from `Dim_Date`; `Month` is also normalized to a `YYYY-MM` string for grouping. |
| Client, model, feature, region, and subscription descriptive columns | Attributes brought in from their corresponding dimensions. |
| `TokenEfficiencyRatio` | `TokensOutput / TokensInput`; zero when input tokens are zero. |
| `CostPerSuccessfulRequest` | `CostUSD` for successful requests, otherwise zero. |
| `TheoreticalCostUSD`, `CostVariance`, `IsOverBudget` | Reference token-rate cost, actual-minus-reference variance, and whether variance exceeds 0.005 USD. |
| `HourOfDay`, `PeakHour` | Hour extracted from `TimeKey`; peak is 09:00 through 17:59. |
| `ResponseSpeedTier` | `Fast` below 400 ms, `Medium` below 800 ms, otherwise `Slow`. |
| `SatisfactionBucket` | Low, mid, high, or `Not Rated`; missing satisfaction is explicitly mapped to `Not Rated`. |
| `DaysSinceOnboarding`, `IsFirstWeekClient`, `ClientAgeCohort` | Client age at request time and lifecycle bands. |
| `MonthlyTokensUsed`, `QuotaUtilizationPct`, `IsOverQuota` | Running client-month token use, percentage of tier quota, and a greater-than-100% flag. |
| `EffectiveCostWithOverage` | Recorded cost multiplied by the tier overage multiplier only when over quota. |

## ML-derived and forecast fields

`fact_with_ml.csv` retains the master grain and adds:

- `SatisfactionPredicted`: random-forest prediction trained only on requests with
  a recorded satisfaction score; it fills a modeled estimate, not observed feedback.
- `ChurnRiskScore`: client-level 0-100 score merged onto each request.
- `IsAnomaly` and `AnomalyScore`: Isolation Forest flag and inverted decision score
  based on cost, total tokens, and latency; higher anomaly scores mean more unusual
  observations within the fitted data.

`client_churn_scores.csv` is at **one row per `ClientKey`** with
`ChurnRiskScore`. Forecast files are at **one row per future day** and contain
`ds`, `yhat`, `Model`, and `Metric`; Prophet cost/token files also contain
`yhat_lower` and `yhat_upper` for its 95% interval. Forecast missing values are
not interpreted as zero; zeros in the daily modeling series represent dates with
no recorded requests after the calendar is completed.

## Aggregate output grains

- `agg_client_monthly.csv`: one client per month.
- `agg_model_performance.csv`: one model.
- `agg_region_monthly.csv`: one region per month.
- `agg_feature_adoption.csv`: one feature.

These tables contain summarized measures such as cost, tokens, request count,
latency, success rate, satisfaction, adoption, and ranking fields. A missing
average satisfaction means no rated request contributed to that group; it does
not mean the group received a zero rating.
