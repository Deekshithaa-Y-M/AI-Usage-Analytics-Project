## 📊 DAX Measures

The following DAX measures were created to evaluate AI platform usage, request performance, latency, and cost efficiency using the `Fact_AI_Usage` table.

I only used raw data for the dashboard pages. However, for the python visuals, I imported the processed data, and all the relvant CSV files from outputs folder. The python script (py_visuals.py) was used for their generation.

### 1. Request Performance Measures

**Total Requests**

```dax
Total Requests =
COUNTROWS(Fact_AI_Usage)
```

Counts the total number of AI requests analysed.

**Success Rate**

```dax
Success Rate =
DIVIDE(
    CALCULATE(
        COUNTROWS(Fact_AI_Usage),
        Fact_AI_Usage[SuccessFlag] = 1
    ),
    [Total Requests]
)
```

Calculates the percentage of successful requests relative to total requests.

**Failed Requests**

```dax
Failed Requests =
CALCULATE(
    COUNTROWS(Fact_AI_Usage),
    Fact_AI_Usage[SuccessFlag] = 0
)
```

Counts the total number of failed requests.

### 2. Latency Measures

**Median Latency**

```dax
Median Latency =
MEDIAN(Fact_AI_Usage[LatencyMs])
```

Returns the median response time in milliseconds, reducing the influence of extreme values.

**% High Latency**

```dax
% High Latency =
DIVIDE(
    CALCULATE(
        COUNTROWS(Fact_AI_Usage),
        Fact_AI_Usage[HighLatencyFlag] = "High Latency"
    ),
    [Total Requests]
)
```

Calculates the percentage of requests flagged as high latency.

### 3. Cost Analysis Measures

**Total Cost**

```dax
Total Cost =
SUM(Fact_AI_Usage[CostUSD])
```

Calculates the total platform expenditure across all requests.

**Cost per Request**

```dax
Cost per Request =
DIVIDE(
    [Total Cost],
    [Total Requests]
)
```

Calculates the average cost incurred per AI request.

**Average Cost per 1K Tokens**

```dax
Avg Cost per 1K Tokens =
DIVIDE(
    [Total Cost],
    SUM(Fact_AI_Usage[TotalTokens]) / 1000
)
```

Calculates the average expenditure for every 1,000 tokens processed.

**Cost per 1K Tokens**

```dax
Cost per 1K Tokens =
DIVIDE(
    SUM(Fact_AI_Usage[CostUSD]),
    SUM(Fact_AI_Usage[TotalTokens]) / 1000
)
```

Calculates the cost per 1,000 tokens, useful for comparing the cost efficiency of different AI models and providers.

---

**Note:** Format percentage measures as percentages and cost measures as USD currency in Power BI.