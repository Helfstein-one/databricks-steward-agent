"""Gold Layer Aggregated Business KPIs: workspace.default.medallion_silver_transactions"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_gold_pipeline(spark: SparkSession) -> None:
    silver_df = spark.table("silver_medallion_silver_transactions")

    gold_df = (
        silver_df
        .groupBy("transaction_id", "user_id", "amount", "status", "category", "date", "timestamp", "created_at", "_ingested_at", "_silver_processed_at")
        .agg(
        F.expr('COUNT(DISTINCT transaction_id)').alias('total_transactions'),
        F.expr('SUM(amount)').alias('total_revenue'),
        F.expr('AVG(amount)').alias('avg_transaction_value'),
        F.expr('SUM(CASE WHEN status = \'COMPLETED\' THEN amount ELSE 0 END)').alias('completed_revenue')
        )
    )

    (
        gold_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable("workspace.default.medallion_silver_transactions")
    )

    spark.sql("OPTIMIZE workspace.default.medallion_silver_transactions ZORDER BY (transaction_id, user_id);")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Gold_medallion_silver_transactions").getOrCreate()
    run_gold_pipeline(spark)