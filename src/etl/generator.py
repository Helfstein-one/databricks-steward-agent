"""Modular ETL code generator producing PySpark and SparkSQL for Medallion architecture."""

from __future__ import annotations

from pydantic import BaseModel

from src.etl.templates import (
    build_delta_create_table,
    build_delta_merge_query,
    build_optimize_query,
    build_vacuum_query,
)
from src.semantic.compiler import ensure_nullif_division_safety
from src.semantic.models import EntityModel


class GeneratedPipeline(BaseModel):
    """Container holding generated PySpark and SparkSQL pipeline code."""

    pyspark_code: str
    sparksql_code: str
    layer: str
    table_name: str


def _map_spark_type(sql_type: str) -> str:
    """Map generic or SQL types to PySpark DataType class names."""
    t = sql_type.lower().strip()
    if t in ("string", "text", "varchar"):
        return "StringType()"
    if t in ("int", "integer"):
        return "IntegerType()"
    if t in ("bigint", "long"):
        return "LongType()"
    if t in ("short", "smallint"):
        return "ShortType()"
    if t in ("byte", "tinyint"):
        return "ByteType()"
    if t in ("binary", "varbinary"):
        return "BinaryType()"
    if t in ("double", "float", "decimal"):
        return "DoubleType()"
    if t in ("bool", "boolean"):
        return "BooleanType()"
    if t in ("date",):
        return "DateType()"
    if t in ("timestamp", "datetime"):
        return "TimestampType()"
    return "StringType()"


def generate_medallion_pipeline(
    entity: EntityModel,
    layer: str,
    source_table_or_path: str | None = None,
) -> GeneratedPipeline:
    """Generate production-ready PySpark and SparkSQL pipelines for a Medallion layer.

    Args:
        entity: EntityModel specifying schema, dimensions, metrics, and primary keys.
        layer: Target medallion layer ('bronze', 'silver', or 'gold').
        source_table_or_path: Optional source table or file path for input data.

    Returns:
        GeneratedPipeline containing PySpark script and SparkSQL script.
    """
    layer_norm = layer.lower().strip()
    table_name = entity.table_name or entity.name
    cols = entity.columns or []
    dims = entity.dimensions or []
    metrics = entity.metrics or []
    pk = entity.primary_key

    if layer_norm == "bronze":
        source_path = source_table_or_path or f"/mnt/raw/{entity.name}"
        
        # Build StructType schema for raw ingestion
        schema_fields = []
        for col in cols:
            stype = _map_spark_type(col.type)
            nullable = "True" if col.nullable else "False"
            schema_fields.append(f"    StructField('{col.name}', {stype}, {nullable})")

        schema_code = "StructType([\n" + ",\n".join(schema_fields) + "\n])" if schema_fields else "None"

        pyspark_code = f'''"""Bronze Layer Raw Ingestion Pipeline: {table_name}"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

def run_bronze_pipeline(spark: SparkSession) -> None:
    schema = {schema_code}

    raw_df = (
        spark.read
        .format("json")
        .schema(schema)
        .load("{source_path}")
    )

    # Ingestion audit columns
    bronze_df = (
        raw_df
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )

    (
        bronze_df.write
        .format("delta")
        .mode("append")
        .saveAsTable("{table_name}")
    )

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Bronze_{entity.name}").getOrCreate()
    run_bronze_pipeline(spark)
'''

        col_dicts = [{"name": c.name, "type": c.type} for c in cols]
        if not any(c["name"] == "_ingested_at" for c in col_dicts):
            col_dicts.append({"name": "_ingested_at", "type": "TIMESTAMP"})
        if not any(c["name"] == "_source_file" for c in col_dicts):
            col_dicts.append({"name": "_source_file", "type": "STRING"})

        sparksql_code = f"""-- Bronze Layer DDL and Ingestion: {table_name}
{build_delta_create_table(table_name, col_dicts, comment=entity.description)}

COPY INTO {table_name}
FROM '{source_path}'
FILEFORMAT = JSON
COPY_OPTIONS ('mergeSchema' = 'true');

{build_vacuum_query(table_name, retention_hours=168)}
"""

    elif layer_norm == "silver":
        source_bronze = source_table_or_path or f"bronze_{entity.name}"
        dedup_clause = f'        .dropDuplicates(["{pk}"])\n' if pk else ""
        filter_clause = f'        .filter(F.col("{pk}").isNotNull())\n' if pk else ""

        cast_expressions = []
        for col in cols:
            if col.name not in ("_ingested_at", "_source_file"):
                cast_expressions.append(f'        .withColumn("{col.name}", F.col("{col.name}").cast("{col.type}"))')
        casts_code = "\n".join(cast_expressions)
        if casts_code:
            casts_code += "\n"

        pyspark_code = f'''"""Silver Layer Cleansing & Deduplication: {table_name}"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_silver_pipeline(spark: SparkSession) -> None:
    source_df = spark.table("{source_bronze}")

    # Deduplication and explicit casting
    cleaned_df = (
        source_df
{casts_code}{dedup_clause}{filter_clause}    )

    (
        cleaned_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("{table_name}")
    )

    spark.sql("{build_optimize_query(table_name, zorder_columns=[pk] if pk else None)}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Silver_{entity.name}").getOrCreate()
    run_silver_pipeline(spark)
'''

        col_dicts = [{"name": c.name, "type": c.type} for c in cols]
        join_keys = [pk] if pk else [cols[0].name] if cols else ["id"]
        update_cols = [c.name for c in cols if c.name != pk]

        sparksql_code = f"""-- Silver Layer Cleansing & Merge: {table_name}
{build_delta_create_table(table_name, col_dicts, comment=entity.description)}

{build_delta_merge_query(table_name, source_bronze, join_keys, update_cols, [c.name for c in cols])}

{build_optimize_query(table_name, zorder_columns=join_keys)}
"""

    elif layer_norm == "gold":
        source_silver = source_table_or_path or f"silver_{entity.name}"
        dim_cols = [d.column or d.name for d in dims] or ([c.name for c in cols[:2]] if cols else ["id"])
        dim_strings = ", ".join([f'"{d}"' for d in dim_cols])

        metric_selects = []
        for m in metrics:
            safe_sql = ensure_nullif_division_safety(m.sql).replace("'", "\\'")
            metric_selects.append(f"        F.expr('{safe_sql}').alias('{m.name}')")
        metrics_code = ",\n".join(metric_selects) if metric_selects else "        F.count('*').alias('record_count')"

        pyspark_code = f'''"""Gold Layer Aggregated Business KPIs: {table_name}"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_gold_pipeline(spark: SparkSession) -> None:
    silver_df = spark.table("{source_silver}")

    gold_df = (
        silver_df
        .groupBy({dim_strings})
        .agg(
{metrics_code}
        )
    )

    (
        gold_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable("{table_name}")
    )

    spark.sql("{build_optimize_query(table_name, zorder_columns=dim_cols[:2])}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Gold_{entity.name}").getOrCreate()
    run_gold_pipeline(spark)
'''

        sql_projections = [f"  `{d}`" for d in dim_cols]
        for m in metrics:
            safe_sql = ensure_nullif_division_safety(m.sql)
            sql_projections.append(f"  {safe_sql} AS `{m.name}`")
        if not metrics:
            sql_projections.append("  COUNT(*) AS `record_count`")

        proj_str = ",\n".join(sql_projections)
        group_str = ", ".join([f"`{d}`" for d in dim_cols])

        sparksql_code = f"""-- Gold Layer Aggregation: {table_name}
CREATE OR REPLACE TABLE {table_name}
USING DELTA
AS
SELECT
{proj_str}
FROM {source_silver}
GROUP BY {group_str};

{build_optimize_query(table_name, zorder_columns=dim_cols[:2])}
"""

    else:
        raise ValueError(f"Unsupported layer '{layer}'. Must be 'bronze', 'silver', or 'gold'.")

    return GeneratedPipeline(
        pyspark_code=pyspark_code,
        sparksql_code=sparksql_code,
        layer=layer_norm,
        table_name=table_name,
    )
