"""Gold Layer Aggregated Business KPIs: workspace.default.medallion_gold_customer_kpis"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_gold_pipeline(spark: SparkSession) -> None:
    silver_df = spark.table("silver_medallion_gold_customer_kpis")

    gold_df = (
        silver_df
        .groupBy("user_id", "lifetime_spend", "lifetime_orders", "avg_spend_per_order", "is_vip")
        .agg(
        F.expr('COUNT(DISTINCT CASE WHEN is_vip = true THEN user_id END)').alias('total_vip_customers'),
        F.expr('COUNT(DISTINCT user_id)').alias('total_customer_base'),
        F.expr('AVG(lifetime_spend)').alias('average_customer_ltv'),
        F.expr('SUM(lifetime_spend)').alias('total_portfolio_revenue')
        )
    )

    (
        gold_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable("workspace.default.medallion_gold_customer_kpis")
    )

    spark.sql("OPTIMIZE workspace.default.medallion_gold_customer_kpis ZORDER BY (user_id, lifetime_spend);")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Gold_medallion_gold_customer_kpis").getOrCreate()
    run_gold_pipeline(spark)