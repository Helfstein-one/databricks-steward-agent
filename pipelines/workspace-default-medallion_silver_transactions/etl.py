"""Silver Layer Cleansing & Deduplication: workspace.default.medallion_silver_transactions"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_silver_pipeline(spark: SparkSession) -> None:
    source_df = spark.table("bronze_medallion_silver_transactions")

    # Deduplication and explicit casting
    cleaned_df = (
        source_df
        .withColumn("transaction_id", F.col("transaction_id").cast("string"))
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("amount", F.col("amount").cast("double"))
        .withColumn("status", F.col("status").cast("string"))
        .withColumn("category", F.col("category").cast("string"))
        .withColumn("date", F.col("date").cast("date"))
        .withColumn("timestamp", F.col("timestamp").cast("timestamp"))
        .withColumn("created_at", F.col("created_at").cast("string"))
        .withColumn("_silver_processed_at", F.col("_silver_processed_at").cast("timestamp"))
        .dropDuplicates(["transaction_id"])
        .filter(F.col("transaction_id").isNotNull())
    )

    (
        cleaned_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("workspace.default.medallion_silver_transactions")
    )

    spark.sql("OPTIMIZE workspace.default.medallion_silver_transactions ZORDER BY (transaction_id);")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Silver_medallion_silver_transactions").getOrCreate()
    run_silver_pipeline(spark)
