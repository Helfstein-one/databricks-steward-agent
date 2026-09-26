-- Gold Layer Aggregation: workspace.default.medallion_gold_customer_kpis
CREATE OR REPLACE TABLE workspace.default.medallion_gold_customer_kpis
USING DELTA
AS
SELECT
  `id`,
  COUNT(*) AS `record_count`
FROM silver_workspace.default.medallion_gold_customer_kpis
GROUP BY `id`;

OPTIMIZE workspace.default.medallion_gold_customer_kpis ZORDER BY (id);
