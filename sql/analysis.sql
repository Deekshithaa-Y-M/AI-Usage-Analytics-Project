-- SQL Analysis Scripts
--
-- Import the seven raw CSVs as tables named Fact_AI_Usage, Dim_Date,
-- Dim_Client, Dim_Model, Dim_Feature, Dim_Region, and Dim_Subscription.
-- DuckDB example:
--   CREATE TABLE Fact_AI_Usage AS SELECT * FROM read_csv_auto('data/raw/Fact_AI_Usage.csv');
--   CREATE TABLE Dim_Date AS SELECT * FROM read_csv_auto('data/raw/Dim_Date.csv');
-- Repeat for the remaining six files. In SQLite, use .import in the same
-- table names. This view reconstructs the analysis grain from those raw tables.
DROP VIEW IF EXISTS fact_master;

CREATE TEMP VIEW fact_master AS
WITH raw_fact AS (
    SELECT
        f.*,
        d.Date AS UsageDate,
        c.ClientName,
        c.OnboardingDate,
        m.ModelName,
        m.Provider,
        m.CostPer1kInput,
        m.CostPer1kOutput,
        feat.FeatureName,
        r.RegionName,
        s.TierName,
        s.MonthlyQuotaTokens
    FROM Fact_AI_Usage f
    JOIN Dim_Date d ON d.DateKey = f.DateKey
    JOIN Dim_Client c ON c.ClientKey = f.ClientKey
    JOIN Dim_Model m ON m.ModelKey = f.ModelKey
    JOIN Dim_Feature feat ON feat.FeatureKey = f.FeatureKey
    JOIN Dim_Region r ON r.RegionKey = f.RegionKey
    JOIN Dim_Subscription s ON s.SubscriptionTierKey = f.SubscriptionTierKey
),
monthly_tokens AS (
    SELECT
        ClientKey,
        substr(UsageDate, 1, 7) AS UsageMonth,
        SUM(TotalTokens) AS MonthlyTokensUsed
    FROM raw_fact
    GROUP BY ClientKey, substr(UsageDate, 1, 7)
)
SELECT
    rf.*,
    rf.UsageDate AS Date,
    substr(rf.UsageDate, 1, 7) AS Month,
    mt.MonthlyTokensUsed,
    mt.MonthlyTokensUsed * 100.0 / NULLIF(rf.MonthlyQuotaTokens, 0)
        AS QuotaUtilizationPct
FROM raw_fact rf
JOIN monthly_tokens mt
  ON mt.ClientKey = rf.ClientKey
 AND mt.UsageMonth = substr(rf.UsageDate, 1, 7);

-- 1. Running seven-day cost average per model.
-- Aggregate to one row per model/day before applying the seven-row window.
WITH daily_model_cost AS (
    SELECT
        ModelKey,
        ModelName,
        Date,
        SUM(CostUSD) AS DailyCostUSD
    FROM fact_master
    GROUP BY ModelKey, ModelName, Date
),
with_previous_day AS (
    SELECT
        *,
        LAG(DailyCostUSD) OVER (
            PARTITION BY ModelKey ORDER BY Date
        ) AS PriorDayCostUSD
    FROM daily_model_cost
)
SELECT
    ModelKey,
    ModelName,
    Date,
    DailyCostUSD,
    PriorDayCostUSD,
    AVG(DailyCostUSD) OVER (
        PARTITION BY ModelKey
        ORDER BY Date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS Rolling7DayAvgCostUSD
FROM with_previous_day
ORDER BY ModelName, Date;

-- 2. Cost-efficiency ranking within each provider.
WITH model_cost_efficiency AS (
    SELECT
        ModelKey,
        ModelName,
        Provider,
        SUM(CostUSD) * 1000.0 / NULLIF(SUM(TotalTokens), 0)
            AS CostPer1000Tokens
    FROM fact_master
    GROUP BY ModelKey, ModelName, Provider
)
SELECT
    *,
    RANK() OVER (
        PARTITION BY Provider ORDER BY CostPer1000Tokens
    ) AS ProviderCostRank
FROM model_cost_efficiency
ORDER BY Provider, ProviderCostRank, ModelName;

-- 3. Client RFM analysis. Higher frequency/monetary values are better;
-- lower recency (days since last request) is better.
WITH analysis_end AS (
    SELECT MAX(Date) AS MaxDate FROM fact_master
),
client_rfm AS (
    SELECT
        ClientKey,
        ClientName,
        CAST(
            (
                CAST(strftime('%s', (SELECT MaxDate FROM analysis_end)) AS BIGINT)
                - CAST(strftime('%s', MAX(Date)) AS BIGINT)
            ) / 86400
            AS INTEGER
        ) AS RecencyDays,
        COUNT(*) AS Frequency,
        SUM(CostUSD) AS MonetaryUSD
    FROM fact_master
    GROUP BY ClientKey, ClientName
),
rfm_scores AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY RecencyDays DESC) AS RecencyScore,
        NTILE(4) OVER (ORDER BY Frequency) AS FrequencyScore,
        NTILE(4) OVER (ORDER BY MonetaryUSD) AS MonetaryScore
    FROM client_rfm
)
SELECT
    *,
    CAST(RecencyScore AS VARCHAR) || CAST(FrequencyScore AS VARCHAR)
        || CAST(MonetaryScore AS VARCHAR) AS RFMSegment
FROM rfm_scores
ORDER BY MonetaryUSD DESC;

-- 4. Feature-adoption funnel: percentage of all clients using each feature.
WITH total_clients AS (
    SELECT COUNT(DISTINCT ClientKey) AS ClientCount FROM fact_master
),
feature_users AS (
    SELECT
        FeatureKey,
        FeatureName,
        COUNT(DISTINCT ClientKey) AS AdoptingClients
    FROM fact_master
    GROUP BY FeatureKey, FeatureName
)
SELECT
    FeatureKey,
    FeatureName,
    AdoptingClients,
    AdoptingClients * 100.0 / NULLIF((SELECT ClientCount FROM total_clients), 0)
        AS AdoptionPct
FROM feature_users
ORDER BY AdoptionPct DESC, FeatureName;

-- 5. Month-over-month cost and request growth by region.
WITH region_monthly AS (
    SELECT
        RegionKey,
        RegionName,
        Month,
        SUM(CostUSD) AS TotalCostUSD,
        COUNT(*) AS RequestCount
    FROM fact_master
    GROUP BY RegionKey, RegionName, Month
),
with_prior_month AS (
    SELECT
        *,
        LAG(TotalCostUSD) OVER (
            PARTITION BY RegionKey ORDER BY Month
        ) AS PriorMonthCostUSD,
        LAG(RequestCount) OVER (
            PARTITION BY RegionKey ORDER BY Month
        ) AS PriorMonthRequestCount
    FROM region_monthly
)
SELECT
    *,
    TotalCostUSD - PriorMonthCostUSD AS CostChangeUSD,
    (TotalCostUSD - PriorMonthCostUSD) * 100.0
        / NULLIF(PriorMonthCostUSD, 0) AS CostGrowthPct,
    RequestCount - PriorMonthRequestCount AS RequestChange,
    (RequestCount - PriorMonthRequestCount) * 100.0
        / NULLIF(PriorMonthRequestCount, 0) AS RequestGrowthPct
FROM with_prior_month
ORDER BY RegionName, Month;

-- 6. Upsell opportunities: high quota utilization outside Enterprise.
SELECT
    ClientKey,
    ClientName,
        TierName,
        QuotaUtilizationPct,
        MonthlyTokensUsed,
    'Upsell opportunity' AS Opportunity
FROM fact_master
WHERE QuotaUtilizationPct > 80
    AND TierName <> 'Enterprise'
ORDER BY QuotaUtilizationPct DESC;

-- 7. Model substitution: compare each client's primary model with the
-- cheapest model observed for each feature used by that client.
WITH client_model_usage AS (
    SELECT
        ClientKey,
        ModelKey,
        ModelName,
        COUNT(*) AS RequestCount,
        ROW_NUMBER() OVER (
            PARTITION BY ClientKey ORDER BY COUNT(*) DESC, ModelKey
        ) AS ModelUsageRank
    FROM fact_master
    GROUP BY ClientKey, ModelKey, ModelName
),
primary_models AS (
    SELECT ClientKey, ModelKey AS PrimaryModelKey, ModelName AS PrimaryModelName
    FROM client_model_usage
    WHERE ModelUsageRank = 1
),
primary_model_requests AS (
    SELECT f.*
    FROM fact_master f
    JOIN primary_models p
      ON p.ClientKey = f.ClientKey AND p.PrimaryModelKey = f.ModelKey
),
feature_model_rates AS (
    SELECT
        FeatureKey,
        ModelKey,
        ModelName,
        CostPer1kInput,
        CostPer1kOutput,
        ROW_NUMBER() OVER (
            PARTITION BY FeatureKey
            ORDER BY CostPer1kInput + CostPer1kOutput, ModelKey
        ) AS CheapestRank
    FROM fact_master
    GROUP BY FeatureKey, ModelKey, ModelName, CostPer1kInput, CostPer1kOutput
),
substitution AS (
    SELECT
        r.ClientKey,
        p.PrimaryModelName,
        r.FeatureKey,
        c.ModelName AS CheapestModelName,
        SUM(r.CostUSD) AS CurrentCostUSD,
        SUM(
            r.TokensInput * c.CostPer1kInput / 1000.0
            + r.TokensOutput * c.CostPer1kOutput / 1000.0
        ) AS CheapestEquivalentCostUSD
    FROM primary_model_requests r
    JOIN primary_models p ON p.ClientKey = r.ClientKey
    JOIN feature_model_rates c
      ON c.FeatureKey = r.FeatureKey AND c.CheapestRank = 1
    GROUP BY r.ClientKey, p.PrimaryModelName, r.FeatureKey, c.ModelName
)
SELECT
    ClientKey,
    PrimaryModelName,
    CheapestModelName,
    SUM(CurrentCostUSD) AS CurrentCostUSD,
    SUM(CheapestEquivalentCostUSD) AS CheapestEquivalentCostUSD,
    SUM(CurrentCostUSD - CheapestEquivalentCostUSD) AS EstimatedSavingsUSD
FROM substitution
GROUP BY ClientKey, PrimaryModelName, CheapestModelName
HAVING EstimatedSavingsUSD > 0
ORDER BY EstimatedSavingsUSD DESC;

-- 8. Correlation between request cost and satisfaction, written without CORR.
WITH rated AS (
    SELECT CostUSD AS x, SatisfactionScore AS y
    FROM fact_master
    WHERE SatisfactionScore IS NOT NULL
),
summary AS (
    SELECT
        COUNT(*) AS n,
        AVG(x) AS mean_x,
        AVG(y) AS mean_y,
        SUM(x * y) AS sum_xy,
        SUM(x * x) AS sum_x2,
        SUM(y * y) AS sum_y2
    FROM rated
)
SELECT
    (sum_xy - n * mean_x * mean_y)
    / NULLIF(
        SQRT(
            (sum_x2 - n * mean_x * mean_x)
            * (sum_y2 - n * mean_y * mean_y)
        ),
        0
    ) AS CostSatisfactionCorrelation
FROM summary;

-- 9. Cost anomalies: population standard deviation estimated from windowed
-- moments, avoiding the non-portable STDEV aggregate name.
WITH cost_stats AS (
    SELECT
        f.*,
        AVG(CostUSD) OVER (PARTITION BY ModelKey) AS ModelAvgCostUSD,
        AVG(CostUSD * CostUSD) OVER (PARTITION BY ModelKey) AS ModelAvgSquaredCost
    FROM fact_master f
),
scored AS (
    SELECT
        *,
        SQRT(
            CASE
                WHEN ModelAvgSquaredCost - ModelAvgCostUSD * ModelAvgCostUSD > 0
                THEN ModelAvgSquaredCost - ModelAvgCostUSD * ModelAvgCostUSD
                ELSE 0
            END
        ) AS ModelStdDevCostUSD
    FROM cost_stats
)
SELECT
    UsageID,
    Date,
    ClientKey,
    ClientName,
    ModelName,
    CostUSD,
    ModelAvgCostUSD,
    ModelStdDevCostUSD,
    ModelAvgCostUSD + 2 * ModelStdDevCostUSD AS AnomalyThresholdUSD,
    CostUSD - (ModelAvgCostUSD + 2 * ModelStdDevCostUSD) AS ExcessCostUSD
FROM scored
WHERE CostUSD > ModelAvgCostUSD + 2 * ModelStdDevCostUSD
ORDER BY ExcessCostUSD DESC
LIMIT 20;

-- 10. Cohort retention by onboarding month and observed activity month.
WITH clients AS (
    SELECT
        ClientKey,
        substr(OnboardingDate, 1, 7) AS CohortMonth
    FROM fact_master
    GROUP BY ClientKey, OnboardingDate
),
activity AS (
    SELECT DISTINCT
        ClientKey,
        substr(Date, 1, 7) AS ActivityMonth
    FROM fact_master
),
cohort_sizes AS (
    SELECT CohortMonth, COUNT(*) AS CohortSize
    FROM clients
    GROUP BY CohortMonth
),
cohort_activity AS (
    SELECT
        c.CohortMonth,
        a.ActivityMonth,
        COUNT(DISTINCT a.ClientKey) AS ActiveClients
    FROM clients c
    JOIN activity a ON a.ClientKey = c.ClientKey
    WHERE a.ActivityMonth >= c.CohortMonth
    GROUP BY c.CohortMonth, a.ActivityMonth
)
SELECT
    ca.CohortMonth,
    ca.ActivityMonth,
    cs.CohortSize,
    ca.ActiveClients,
    ca.ActiveClients * 100.0 / NULLIF(cs.CohortSize, 0) AS RetentionPct
FROM cohort_activity ca
JOIN cohort_sizes cs ON cs.CohortMonth = ca.CohortMonth
ORDER BY ca.CohortMonth, ca.ActivityMonth;