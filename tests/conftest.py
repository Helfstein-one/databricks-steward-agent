"""Global PyTest fixtures and mocks for Databricks Steward Agent test suite."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is in sys.path for test discovery
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_env(monkeypatch):
    """Provides a controlled environment variables setup."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://test-workspace.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test-token-12345")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "warehouse-sql-test-123")
    monkeypatch.setenv("DATABRICKS_DEFAULT_CATALOG", "main")
    monkeypatch.setenv("DATABRICKS_DEFAULT_SCHEMA", "default")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "ollama")
    monkeypatch.setenv("LOCAL_LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_mocktoken1234567890abcdef")
    monkeypatch.setenv("GITHUB_REPOSITORY", "test-org/test-data-lakehouse")
    monkeypatch.setenv("GITHUB_BASE_BRANCH", "main")
    monkeypatch.setenv("DEFAULT_QUERY_LIMIT", "50")
    monkeypatch.setenv("MAX_QUERY_LIMIT", "200")
    return monkeypatch


@pytest.fixture
def mock_workspace_client():
    """Mock Databricks SDK WorkspaceClient for Unity Catalog introspection & workspace ops."""
    mock_client = MagicMock()

    # Mock catalog
    mock_catalog = MagicMock()
    mock_catalog.name = "main"
    mock_catalog.comment = "Main business lakehouse catalog"
    mock_client.catalogs.list.return_value = [mock_catalog]

    # Mock schema
    mock_schema = MagicMock()
    mock_schema.name = "default"
    mock_schema.catalog_name = "main"
    mock_schema.comment = "Default schema"
    mock_client.schemas.list.return_value = [mock_schema]

    # Mock columns
    col_id = MagicMock(name="id", type_name="STRING", comment="Primary key")
    col_id.name = "transaction_id"
    col_id.type_name = "STRING"
    col_id.comment = "Primary key"

    col_amount = MagicMock(name="amount", type_name="DOUBLE", comment="Transaction amount")
    col_amount.name = "amount"
    col_amount.type_name = "DOUBLE"
    col_amount.comment = "Transaction amount"

    col_category = MagicMock(name="category", type_name="STRING", comment="Category")
    col_category.name = "category"
    col_category.type_name = "STRING"
    col_category.comment = "Category"

    # Mock tables
    table_bronze = MagicMock()
    table_bronze.name = "bronze_raw_transactions"
    table_bronze.catalog_name = "main"
    table_bronze.schema_name = "default"
    table_bronze.table_type = "MANAGED"
    table_bronze.columns = [col_id, col_amount, col_category]
    table_bronze.comment = "Raw incoming transactions"

    table_silver = MagicMock()
    table_silver.name = "silver_transactions"
    table_silver.catalog_name = "main"
    table_silver.schema_name = "default"
    table_silver.table_type = "MANAGED"
    table_silver.columns = [col_id, col_amount, col_category]
    table_silver.comment = "Cleaned and deduplicated transactions"

    table_gold = MagicMock()
    table_gold.name = "gold_sales_kpis"
    table_gold.catalog_name = "main"
    table_gold.schema_name = "default"
    table_gold.table_type = "MANAGED"
    table_gold.columns = [col_category, col_amount]
    table_gold.comment = "Aggregated sales KPIs"

    mock_client.tables.list.return_value = [table_bronze, table_silver, table_gold]
    mock_client.workspace.import_.return_value = {"status": "imported"}

    return mock_client


@pytest.fixture
def mock_github():
    """Mock PyGithub client for PR creation and branch management."""
    mock_gh = MagicMock()
    mock_repo = MagicMock()
    mock_pr = MagicMock()

    mock_pr.number = 42
    mock_pr.html_url = "https://github.com/test-org/test-data-lakehouse/pull/42"
    mock_pr.title = "feat(data-product): add credit risk medallion pipeline"
    mock_pr.state = "open"

    mock_repo.create_pull.return_value = mock_pr
    mock_gh.get_repo.return_value = mock_repo

    return mock_gh, mock_repo, mock_pr


@pytest.fixture
def sample_credit_yaml_content():
    """Sample corporate credit domain YAML ontology content."""
    return """
domain: corporate_credit
catalog: main
schema_name: credit_risk
entities:
  - name: counterpart
    table: main.credit_risk.counterparts
    primary_key: counterpart_id
    description: "Empresas e grupos tomadores de credito corporativo"
    synonyms:
      - empresa
      - tomador
      - cliente
    dimensions:
      - name: economic_group
        type: string
        column: nm_economic_group
        description: "Nome do grupo economico"
        synonyms:
          - grupo
          - conglomerado
      - name: sector
        type: string
        column: ds_cnae_sector
        description: "Setor economico CNAE"
      - name: rating
        type: string
        column: cd_rating_agency
        description: "Rating de credito"
    relationships: []

  - name: credit_facility
    table: main.credit_risk.facilities
    primary_key: facility_id
    description: "Operacoes e contratos de credito corporativo"
    synonyms:
      - operacao
      - contrato
      - facilidade
    dimensions:
      - name: operation_type
        type: string
        column: tp_operation
        description: "Tipo da operacao de credito"
      - name: status
        type: string
        column: st_operation
        description: "Status do contrato"
    relationships:
      - from_entity: credit_facility
        from_column: counterpart_id
        to_entity: counterpart
        to_column: counterpart_id
        type: many_to_one
        description: "Pertence a contraparte"

relationships:
  - from_entity: credit_facility
    from_column: counterpart_id
    to_entity: counterpart
    to_column: counterpart_id
    type: many_to_one
    description: "Pertence a contraparte"

metrics:
  - name: total_exposure
    entity: credit_facility
    type: sum
    sql: "SUM(vl_outstanding_balance)"
    description: "Saldo devedor total em aberto (EAD)"
    synonyms:
      - exposicao_total
      - saldo_devedor

  - name: overdue_ratio_90d
    entity: credit_facility
    type: derived
    sql: "SUM(CASE WHEN nr_days_overdue > 90 THEN vl_outstanding_balance ELSE 0 END) / NULLIF(SUM(vl_outstanding_balance), 0)"
    description: "Indice de inadimplencia superior a 90 dias (NPL Ratio)"
    synonyms:
      - inadimplencia_90d
      - npl_ratio
"""


@pytest.fixture
def sample_sales_yaml_content():
    """Sample sales lakehouse domain YAML ontology content."""
    return """
domain: sales_lakehouse
catalog: main
schema_name: retail
entities:
  - name: customer
    table: main.retail.dim_customers
    primary_key: customer_id
    description: "Cadastro de clientes e segmentacao"
    synonyms:
      - comprador
      - cliente
    dimensions:
      - name: customer_city
        type: string
        column: city
      - name: customer_segment
        type: string
        column: segment
    relationships: []

  - name: order
    table: main.retail.fct_orders
    primary_key: order_id
    description: "Fato de pedidos de vendas"
    synonyms:
      - venda
      - pedido
    dimensions:
      - name: order_status
        type: string
        column: status
      - name: order_date
        type: date
        column: dt_order
    relationships:
      - from_entity: order
        from_column: customer_id
        to_entity: customer
        to_column: customer_id
        type: many_to_one

relationships:
  - from_entity: order
    from_column: customer_id
    to_entity: customer
    to_column: customer_id
    type: many_to_one

metrics:
  - name: total_revenue
    entity: order
    type: sum
    sql: "SUM(order_amount)"
    description: "Receita bruta total de vendas"
    synonyms:
      - receita_total
      - faturamento

  - name: avg_ticket
    entity: order
    type: avg
    sql: "AVG(order_amount / NULLIF(item_quantity, 0))"
    description: "Ticket medio por item"
"""


@pytest.fixture
def sample_yaml_dir(tmp_path, sample_credit_yaml_content, sample_sales_yaml_content):
    """Creates a temporary directory with valid and invalid YAML ontologies."""
    models_dir = tmp_path / "semantic_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    credit_file = models_dir / "corporate_credit.yaml"
    credit_file.write_text(sample_credit_yaml_content, encoding="utf-8")

    sales_file = models_dir / "sales_lakehouse.yaml"
    sales_file.write_text(sample_sales_yaml_content, encoding="utf-8")

    return models_dir


# PySpark & SparkSQL test code fixtures
@pytest.fixture
def clean_pyspark_code():
    return '''"""Clean Medallion PySpark pipeline."""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def process_silver_transactions(spark: SparkSession, source_path: str, target_path: str) -> None:
    """Ingest and deduplicate transactions for Silver layer."""
    df = (
        spark.read.format("delta")
        .load(source_path)
        .filter(F.col("dt_partition") >= "2026-01-01")
        .dropDuplicates(["transaction_id"])
        .withColumn("amount_clean", F.coalesce(F.col("amount"), F.lit(0.0)))
    )

    (
        df.write.format("delta")
        .mode("overwrite")
        .partitionBy("dt_partition")
        .save(target_path)
    )
'''


@pytest.fixture
def flawed_pyspark_collect_code():
    return '''"""PySpark pipeline with dangerous unbounded collect()."""
from pyspark.sql import SparkSession


def bad_pipeline(spark: SparkSession) -> None:
    df = spark.read.table("main.default.huge_transactions")
    # Anti-pattern: Unbounded collect on driver memory
    all_rows = df.collect()
    print(f"Loaded {len(all_rows)} rows")
'''


@pytest.fixture
def flawed_pyspark_cross_join_code():
    return '''"""PySpark pipeline with accidental crossJoin."""
from pyspark.sql import SparkSession


def bad_join_pipeline(spark: SparkSession) -> None:
    df1 = spark.read.table("main.default.customers")
    df2 = spark.read.table("main.default.orders")
    # Anti-pattern: Cartesian crossJoin
    result = df1.crossJoin(df2)
    result.write.format("delta").saveAsTable("main.default.huge_cartesian")
'''


@pytest.fixture
def flawed_pyspark_syntax_error_code():
    return '''"""PySpark pipeline with syntax error."""
def broken_function(:
    return "syntax error"
'''


@pytest.fixture
def clean_sparksql_code():
    return """-- Clean SparkSQL analytical KPI model
SELECT
    c.nm_economic_group AS economic_group,
    f.tp_operation AS operation_type,
    SUM(f.vl_outstanding_balance) AS total_exposure,
    SUM(CASE WHEN f.nr_days_overdue > 90 THEN f.vl_outstanding_balance ELSE 0 END)
        / NULLIF(SUM(f.vl_outstanding_balance), 0) AS npl_ratio
FROM main.credit_risk.facilities AS f
INNER JOIN main.credit_risk.counterparts AS c
    ON f.counterpart_id = c.counterpart_id
WHERE f.dt_partition >= '2026-01-01'
GROUP BY c.nm_economic_group, f.tp_operation
HAVING SUM(f.vl_outstanding_balance) > 0
"""


@pytest.fixture
def flawed_sparksql_cross_join_code():
    return """-- Flawed SparkSQL with Cartesian cross join
SELECT * FROM main.credit_risk.facilities, main.credit_risk.counterparts
WHERE facilities.status = 'ACTIVE'
"""


@pytest.fixture
def flawed_sparksql_syntax_error_code():
    return """-- Flawed SparkSQL with invalid syntax
SELEC col_1, col_2 FORM my_table GROUP
"""
