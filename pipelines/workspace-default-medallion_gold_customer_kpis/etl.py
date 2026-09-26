"""Gold Layer Aggregated Business KPIs: workspace.default.medallion_gold_customer_kpis"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_gold_pipeline(spark: SparkSession) -> None:
    silver_df = spark.table("silver_workspace.default.medallion_gold_customer_kpis")

    gold_df = (
        silver_df
        .groupBy("id")
        .agg(
        F.count('*').alias('record_count')
        )
    )

    (
        gold_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable("workspace.default.medallion_gold_customer_kpis")
    )

    spark.sql("OPTIMIZE workspace.default.medallion_gold_customer_kpis ZORDER BY (id);")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Gold_workspace.default.medallion_gold_customer_kpis").getOrCreate()
    run_gold_pipeline(spark)
