"""Unit and component integration tests for Medallion ETL code generation (Tier 2)."""

import pytest

from src.etl.generator import GeneratedPipeline, generate_medallion_pipeline
from src.etl.templates import (
    build_delta_create_table,
    build_delta_merge_query,
    build_optimize_query,
    build_vacuum_query,
)
from src.semantic.models import (
    ColumnModel,
    DimensionModel,
    EntityModel,
    MetricModel,
)

# ==============================================================================
# Delta Lake Templates Unit Tests
# ==============================================================================


def test_build_optimize_query():
    """Verify OPTIMIZE statement generation with and without ZORDER BY."""
    opt_simple = build_optimize_query("main.default.transactions")
    assert opt_simple == "OPTIMIZE main.default.transactions;"

    opt_zorder = build_optimize_query(
        "main.default.transactions",
        zorder_columns=["transaction_date", "user_id"],
        where_clause="dt_partition >= '2026-01-01'",
    )
    assert "OPTIMIZE main.default.transactions" in opt_zorder
    assert "WHERE dt_partition >= '2026-01-01'" in opt_zorder
    assert "ZORDER BY (transaction_date, user_id)" in opt_zorder


def test_build_vacuum_query():
    """Verify VACUUM statement generation with default and custom retention hours."""
    vac_default = build_vacuum_query("main.default.events")
    assert vac_default == "VACUUM main.default.events RETAIN 168 HOURS;"

    vac_custom = build_vacuum_query("main.default.events", retention_hours=72)
    assert vac_custom == "VACUUM main.default.events RETAIN 72 HOURS;"


def test_build_delta_create_table():
    """Verify CREATE TABLE IF NOT EXISTS statement for Delta Lake."""
    cols = [
        {"name": "id", "type": "STRING"},
        {"name": "val", "type": "DOUBLE"},
        {"name": "dt", "type": "DATE"},
    ]
    ddl = build_delta_create_table(
        "main.default.sample_table",
        cols,
        partition_by=["dt"],
        comment="Sample partitioned table",
    )
    assert "CREATE TABLE IF NOT EXISTS main.default.sample_table (" in ddl
    assert "`id` STRING" in ddl
    assert "USING DELTA" in ddl
    assert "PARTITIONED BY (`dt`)" in ddl
    assert "COMMENT 'Sample partitioned table'" in ddl


def test_build_delta_merge_query():
    """Verify MERGE INTO statement generation with matched and not matched clauses."""
    merge_sql = build_delta_merge_query(
        target_table="main.silver.customers",
        source_table="bronze_customers",
        join_keys=["customer_id"],
        update_columns=["name", "city"],
        insert_columns=["customer_id", "name", "city"],
    )
    assert "MERGE INTO main.silver.customers AS target" in merge_sql
    assert "USING bronze_customers AS source" in merge_sql
    assert "ON target.`customer_id` = source.`customer_id`" in merge_sql
    assert (
        "WHEN MATCHED THEN UPDATE SET `name` = source.`name`, `city` = source.`city`" in merge_sql
    )
    assert "WHEN NOT MATCHED THEN INSERT (`customer_id`, `name`, `city`)" in merge_sql


# ==============================================================================
# Medallion Pipeline Generator Tests
# ==============================================================================


@pytest.fixture
def sample_test_entity():
    return EntityModel(
        name="transactions",
        table_name="main.lakehouse.transactions",
        primary_key="transaction_id",
        description="Core financial transactions",
        columns=[
            ColumnModel(name="transaction_id", type="string", primary_key=True),
            ColumnModel(name="user_id", type="string"),
            ColumnModel(name="amount", type="double"),
            ColumnModel(name="category", type="string"),
            ColumnModel(name="transaction_date", type="date"),
        ],
        dimensions=[
            DimensionModel(name="category", column="category"),
            DimensionModel(name="transaction_date", column="transaction_date"),
        ],
        metrics=[
            MetricModel(
                name="total_amount",
                entity="transactions",
                type="sum",
                sql="SUM(amount)",
            ),
            MetricModel(
                name="avg_ticket",
                entity="transactions",
                type="ratio",
                sql="SUM(amount) / NULLIF(COUNT(*), 0)",
            ),
        ],
    )


def test_generate_bronze_pipeline(sample_test_entity):
    """Verify Bronze layer generation produces raw ingestion and audit tracking."""
    pipeline = generate_medallion_pipeline(sample_test_entity, layer="bronze")

    assert isinstance(pipeline, GeneratedPipeline)
    assert pipeline.layer == "bronze"
    assert pipeline.table_name == "main.lakehouse.transactions"

    # Verify PySpark code
    pyspark = pipeline.pyspark_code
    assert "def run_bronze_pipeline(spark: SparkSession)" in pyspark
    assert "StructType([" in pyspark
    assert "StructField('transaction_id'" in pyspark
    assert "_ingested_at" in pyspark
    assert "_source_file" in pyspark
    assert '.format("delta")' in pyspark
    assert '.mode("append")' in pyspark

    # Verify SparkSQL code
    sql = pipeline.sparksql_code
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert "COPY INTO" in sql
    assert "VACUUM" in sql


def test_generate_silver_pipeline(sample_test_entity):
    """Verify Silver layer generation produces deduplication, typing, and merge."""
    pipeline = generate_medallion_pipeline(sample_test_entity, layer="silver")

    assert pipeline.layer == "silver"

    # Verify PySpark code
    pyspark = pipeline.pyspark_code
    assert "def run_silver_pipeline(spark: SparkSession)" in pyspark
    assert '.dropDuplicates(["transaction_id"])' in pyspark
    assert '.mode("overwrite")' in pyspark
    assert "OPTIMIZE" in pyspark

    # Verify SparkSQL code
    sql = pipeline.sparksql_code
    assert "MERGE INTO" in sql
    assert "WHEN MATCHED THEN UPDATE" in sql
    assert "OPTIMIZE" in sql


def test_generate_gold_pipeline(sample_test_entity):
    """Verify Gold layer generation produces aggregations on business dimensions."""
    pipeline = generate_medallion_pipeline(sample_test_entity, layer="gold")

    assert pipeline.layer == "gold"

    # Verify PySpark code
    pyspark = pipeline.pyspark_code
    assert "def run_gold_pipeline(spark: SparkSession)" in pyspark
    assert ".groupBy(" in pyspark
    assert '"category"' in pyspark
    assert "F.expr('SUM(amount)').alias('total_amount')" in pyspark
    assert "NULLIF(" in pyspark

    # Verify SparkSQL code
    sql = pipeline.sparksql_code
    assert "CREATE OR REPLACE TABLE" in sql
    assert "GROUP BY" in sql
    assert "`category`" in sql
    assert "SUM(amount) AS `total_amount`" in sql
    assert "NULLIF(" in sql


def test_generate_invalid_layer(sample_test_entity):
    """Verify ValueError on unsupported medallion layer."""
    with pytest.raises(ValueError) as exc_info:
        generate_medallion_pipeline(sample_test_entity, layer="platinum")
    assert "Unsupported layer 'platinum'" in str(exc_info.value)
