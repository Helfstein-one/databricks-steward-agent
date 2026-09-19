-- Gold Layer Aggregation: workspace.default.medallion_silver_transactions
CREATE OR REPLACE TABLE workspace.default.medallion_silver_transactions
USING DELTA
AS
SELECT
  `transaction_id`,
  `user_id`,
  `amount`,
  `status`,
  `category`,
  `date`,
  `timestamp`,
  `created_at`,
  `_ingested_at`,
  `_silver_processed_at`,
  COUNT(DISTINCT transaction_id) AS `total_transactions`,
  SUM(amount) AS `total_revenue`,
  AVG(amount) AS `avg_transaction_value`,
  SUM(CASE WHEN status = 'COMPLETED' THEN amount ELSE 0 END) AS `completed_revenue`
FROM silver_medallion_silver_transactions
GROUP BY `transaction_id`, `user_id`, `amount`, `status`, `category`, `date`, `timestamp`, `created_at`, `_ingested_at`, `_silver_processed_at`;

OPTIMIZE workspace.default.medallion_silver_transactions ZORDER BY (transaction_id, user_id);