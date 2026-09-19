"""Databricks Unity Catalog introspection and offline demo entity builder."""

from __future__ import annotations

import logging

from src.databricks.client import DatabricksCEClient
from src.semantic.models import (
    ColumnModel,
    DimensionModel,
    EntityModel,
    MetricModel,
    RelationshipModel,
)

logger = logging.getLogger(__name__)


def _build_mock_entities(catalog: str = "main", schema: str = "default") -> list[EntityModel]:
    """Build production-like mock lakehouse entities for offline and demo mode."""
    # Bronze layer: raw transactions
    bronze_entity = EntityModel(
        name="bronze_raw_transactions",
        table_name=f"{catalog}.{schema}.bronze_raw_transactions",
        catalog=catalog,
        schema_name=schema,
        primary_key="transaction_id",
        layer="bronze",
        description="Bronze layer raw ingestion transactions table with payload audit metadata",
        columns=[
            ColumnModel(name="transaction_id", type="string", primary_key=True, nullable=False),
            ColumnModel(name="user_id", type="string", nullable=False),
            ColumnModel(name="amount", type="double", nullable=False),
            ColumnModel(name="status", type="string", nullable=False),
            ColumnModel(name="raw_payload", type="string", nullable=True),
            ColumnModel(name="_ingested_at", type="timestamp", nullable=False),
        ],
        dimensions=[
            DimensionModel(name="transaction_id", type="string", column="transaction_id"),
            DimensionModel(name="user_id", type="string", column="user_id"),
            DimensionModel(name="status", type="string", column="status"),
        ],
        metrics=[
            MetricModel(
                name="raw_record_count",
                entity="bronze_raw_transactions",
                type="count",
                sql="COUNT(*)",
                description="Total raw records ingested",
            )
        ],
        synonyms=["raw_transactions", "bronze_events"],
    )

    # Silver layer: cleansed and deduplicated transactions
    silver_entity = EntityModel(
        name="silver_transactions",
        table_name=f"{catalog}.{schema}.silver_transactions",
        catalog=catalog,
        schema_name=schema,
        primary_key="transaction_id",
        layer="silver",
        description="Silver layer cleansed and deduplicated transactions with validated typing",
        columns=[
            ColumnModel(name="transaction_id", type="string", primary_key=True, nullable=False),
            ColumnModel(name="user_id", type="string", nullable=False),
            ColumnModel(name="amount", type="double", nullable=False),
            ColumnModel(name="status", type="string", nullable=False),
            ColumnModel(name="transaction_date", type="date", nullable=False),
            ColumnModel(name="is_valid", type="boolean", nullable=False),
        ],
        dimensions=[
            DimensionModel(name="transaction_id", type="string", column="transaction_id"),
            DimensionModel(name="user_id", type="string", column="user_id"),
            DimensionModel(name="status", type="string", column="status"),
            DimensionModel(name="transaction_date", type="date", column="transaction_date"),
        ],
        metrics=[
            MetricModel(
                name="transaction_count",
                entity="silver_transactions",
                type="count",
                sql="COUNT(transaction_id)",
                description="Total validated transactions",
            ),
            MetricModel(
                name="total_transaction_amount",
                entity="silver_transactions",
                type="sum",
                sql="SUM(amount)",
                description="Total monetary volume of transactions",
            ),
            MetricModel(
                name="average_transaction_amount",
                entity="silver_transactions",
                type="avg",
                sql="AVG(amount)",
                description="Average transaction ticket",
            ),
        ],
        relationships=[
            RelationshipModel(
                from_entity="silver_transactions",
                to_entity="bronze_raw_transactions",
                from_column="transaction_id",
                to_column="transaction_id",
                type="one_to_one",
            )
        ],
        synonyms=["clean_transactions", "cleansed_events"],
    )

    # Gold layer: aggregated KPIs
    gold_entity = EntityModel(
        name="gold_sales_kpis",
        table_name=f"{catalog}.{schema}.gold_sales_kpis",
        catalog=catalog,
        schema_name=schema,
        primary_key="metric_date",
        layer="gold",
        description="Gold layer analytical sales KPIs aggregated daily",
        columns=[
            ColumnModel(name="metric_date", type="date", primary_key=True, nullable=False),
            ColumnModel(name="status", type="string", nullable=False),
            ColumnModel(name="total_amount", type="double", nullable=False),
            ColumnModel(name="transaction_count", type="bigint", nullable=False),
            ColumnModel(name="avg_amount", type="double", nullable=False),
        ],
        dimensions=[
            DimensionModel(name="metric_date", type="date", column="metric_date"),
            DimensionModel(name="status", type="string", column="status"),
        ],
        metrics=[
            MetricModel(
                name="total_sales",
                entity="gold_sales_kpis",
                type="sum",
                sql="SUM(total_amount)",
                description="Total sales volume in gold layer",
            ),
            MetricModel(
                name="average_ticket",
                entity="gold_sales_kpis",
                type="ratio",
                sql="SUM(total_amount) / NULLIF(SUM(transaction_count), 0)",
                description="Average ticket size with division safety",
            ),
        ],
        synonyms=["daily_sales_kpis", "sales_summary"],
    )

    return [bronze_entity, silver_entity, gold_entity]


def introspect_catalog(
    catalog: str = "main",
    schema: str = "default",
    client: DatabricksCEClient | None = None,
) -> list[EntityModel]:
    """Discover Unity Catalog tables and constraints, falling back to mock entities if offline.

    Args:
        catalog: Databricks Unity Catalog name.
        schema: Databricks schema/database name.
        client: Optional DatabricksCEClient instance.

    Returns:
        List of discovered or fallback EntityModel objects.
    """
    if client is None:
        client = DatabricksCEClient()

    if not client.is_configured():
        logger.info("Databricks client not configured. Using mock entities.")
        return _build_mock_entities(catalog=catalog, schema=schema)

    try:
        assert client.client is not None
        effective_catalog = catalog
        effective_schema = schema

        # Check if requested catalog exists; if not, discover available user catalogs
        try:
            list(client.client.schemas.list(catalog_name=effective_catalog))
        except Exception:  # noqa: BLE001
            available_cats = [c.name for c in client.client.catalogs.list() if c.name != "system"]
            if available_cats:
                effective_catalog = (
                    "workspace" if "workspace" in available_cats else available_cats[0]
                )
                logger.info(
                    "Catalog '%s' not found. Auto-selected available catalog '%s'.",
                    catalog,
                    effective_catalog,
                )
                available_schemas = [
                    s.name
                    for s in client.client.schemas.list(catalog_name=effective_catalog)
                    if s.name != "information_schema"
                ]
                if available_schemas:
                    effective_schema = (
                        "default" if "default" in available_schemas else available_schemas[0]
                    )

        tables_iter = client.client.tables.list(
            catalog_name=effective_catalog, schema_name=effective_schema
        )
        entities: list[EntityModel] = []

        for table in tables_iter:
            table_name = getattr(table, "name", "")
            full_table_name = f"{effective_catalog}.{effective_schema}.{table_name}"
            raw_columns = getattr(table, "columns", []) or []

            cols: list[ColumnModel] = []
            dims: list[DimensionModel] = []

            for col in raw_columns:
                col_name = getattr(col, "name", "")
                type_name = (
                    str(getattr(col, "type_name", "string")).lower().replace("columntypename.", "")
                )
                nullable = bool(getattr(col, "nullable", True))
                cols.append(ColumnModel(name=col_name, type=type_name, nullable=nullable))
                dims.append(DimensionModel(name=col_name, type=type_name, column=col_name))

            # Infer medallion layer from table name naming convention
            inferred_layer = None
            tl = table_name.lower()
            if "bronze" in tl or "raw" in tl:
                inferred_layer = "bronze"
            elif "silver" in tl or "clean" in tl or "curated" in tl:
                inferred_layer = "silver"
            elif "gold" in tl or "kpi" in tl or "metric" in tl or "dim_" in tl or "fact_" in tl:
                inferred_layer = "gold"

            # Infer primary key from column names if present
            pk_col = None
            for c in cols:
                if c.name.lower() in (
                    f"{table_name.lower()}_id",
                    "id",
                    "transaction_id",
                    "customer_id",
                    "facility_id",
                    "order_id",
                ):
                    pk_col = c.name
                    break

            entity = EntityModel(
                name=table_name,
                table_name=full_table_name,
                catalog=effective_catalog,
                schema_name=effective_schema,
                primary_key=pk_col,
                layer=inferred_layer,
                columns=cols,
                dimensions=dims,
                description=getattr(table, "comment", None),
            )
            entities.append(entity)

        if not entities:
            logger.warning(
                f"No tables discovered in {effective_catalog}.{effective_schema}. Using mock entities."
            )
            return _build_mock_entities(catalog=effective_catalog, schema=effective_schema)

        return entities

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unity Catalog introspection failed ({e}). Falling back to mock entities.")
        return _build_mock_entities(catalog=catalog, schema=schema)
