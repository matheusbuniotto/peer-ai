-- DuckDB adapter. In BigQuery, point `cases` at the loaded parquet table and
-- run everything below this view. The queries themselves use only standard SQL.
CREATE OR REPLACE VIEW cases AS
SELECT * FROM read_parquet('results/*/cases.parquet', union_by_name=True);

-- Accuracy by model and flaw.
SELECT model, flaw, AVG(correct) AS accuracy, COUNT(*) AS n
FROM cases
GROUP BY model, flaw
ORDER BY model, flaw;

-- Cost against accuracy. cost_usd is null on a replay, so the average is null.
SELECT model, variant, AVG(correct) AS accuracy, AVG(cost_usd) AS cost_per_case
FROM cases
GROUP BY model, variant
ORDER BY model, variant;

-- Cases that are right on some trials and wrong on others.
SELECT id, variant, SUM(correct) AS correct_trials, COUNT(*) AS trials
FROM cases
GROUP BY id, variant
HAVING SUM(correct) > 0 AND SUM(correct) < COUNT(*)
ORDER BY id, variant;

-- Latency at the 95th percentile via PERCENT_RANK, which BigQuery also has.
WITH ranked AS (
    SELECT
        variant,
        latency_s,
        PERCENT_RANK() OVER (PARTITION BY variant ORDER BY latency_s) AS pct_rank
    FROM cases
    WHERE latency_s IS NOT NULL
)
SELECT variant, MIN(latency_s) AS latency_p95
FROM ranked
WHERE pct_rank >= 0.95
GROUP BY variant
ORDER BY variant;
