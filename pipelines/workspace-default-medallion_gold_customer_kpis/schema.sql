-- Gold Layer Aggregation: workspace.default.medallion_gold_customer_kpis
CREATE OR REPLACE TABLE workspace.default.medallion_gold_customer_kpis
USING DELTA
AS
SELECT
  `user_id`,
  `lifetime_spend`,
  `lifetime_orders`,
  `avg_spend_per_order`,
  `is_vip`,
  COUNT(DISTINCT CASE WHEN is_vip = true THEN user_id END) AS `total_vip_customers`,
  COUNT(DISTINCT user_id) AS `total_customer_base`,
  AVG(lifetime_spend) AS `average_customer_ltv`,
  SUM(lifetime_spend) AS `total_portfolio_revenue`
FROM silver_medallion_gold_customer_kpis
GROUP BY `user_id`, `lifetime_spend`, `lifetime_orders`, `avg_spend_per_order`, `is_vip`;

OPTIMIZE workspace.default.medallion_gold_customer_kpis ZORDER BY (user_id, lifetime_spend);