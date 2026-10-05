-- Silver Layer Cleansing & Merge: workspace.default.medallion_silver_transactions
CREATE TABLE IF NOT EXISTS workspace.default.medallion_silver_transactions (
  `transaction_id` STRING,
  `user_id` STRING,
  `amount` DOUBLE,
  `status` STRING,
  `category` STRING,
  `date` DATE,
  `timestamp` TIMESTAMP,
  `created_at` STRING,
  `_ingested_at` TIMESTAMP,
  `_silver_processed_at` TIMESTAMP
) USING DELTA
COMMENT 'Transações higienizadas, deduplicadas e tipadas com campos temporais enriquecidos (timestamp e date).';

MERGE INTO workspace.default.medallion_silver_transactions AS target
USING bronze_medallion_silver_transactions AS source
ON target.`transaction_id` = source.`transaction_id`
WHEN MATCHED THEN UPDATE SET `user_id` = source.`user_id`, `amount` = source.`amount`, `status` = source.`status`, `category` = source.`category`, `date` = source.`date`, `timestamp` = source.`timestamp`, `created_at` = source.`created_at`, `_ingested_at` = source.`_ingested_at`, `_silver_processed_at` = source.`_silver_processed_at`
WHEN NOT MATCHED THEN INSERT (`transaction_id`, `user_id`, `amount`, `status`, `category`, `date`, `timestamp`, `created_at`, `_ingested_at`, `_silver_processed_at`) VALUES (source.`transaction_id`, source.`user_id`, source.`amount`, source.`status`, source.`category`, source.`date`, source.`timestamp`, source.`created_at`, source.`_ingested_at`, source.`_silver_processed_at`);

OPTIMIZE workspace.default.medallion_silver_transactions ZORDER BY (transaction_id);
