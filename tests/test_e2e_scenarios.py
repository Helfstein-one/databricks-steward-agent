"""Tier 4 Real-World End-to-End Application Scenarios for Databricks Steward Agent."""

from open_webui_pipe import Pipe
from src.ci.runner import run_ci_pipeline
from src.databricks.client import DatabricksCEClient
from src.databricks.introspector import introspect_catalog
from src.etl.generator import generate_medallion_pipeline
from src.gitops.github_pr import create_data_product_pr
from src.semantic.compiler import SemanticQueryCompiler
from src.semantic.registry import SemanticRegistry
from src.visualizer.mermaid import generate_er_diagram, generate_lineage_diagram

# ==============================================================================
# Scenario 1: Complete Conversational Lakehouse Data Product Lifecycle
# ==============================================================================


def test_e2e_scenario_full_lifecycle(sample_yaml_dir, mock_github):
    """Scenario 1: End-to-end flow from conversational prompt to GitHub PR.

    Flow:
    1. Open WebUI Pipe processes user request.
    2. Catalog Introspector discovers lakehouse entities.
    3. Semantic Registry resolves domain ontology and compiles business queries.
    4. Mermaid Visualizer renders Crow's foot ERD and Medallion lineage flowchart.
    5. ETL Generator produces idempotent Bronze, Silver, and Gold PySpark and SparkSQL.
    6. CI Pipeline validates generated code through Ruff, SQLFluff, and Anti-pattern rules.
    7. GitOps Manager opens a GitHub PR embedding diagrams and CI report.
    """
    _mock_gh, mock_repo, _mock_pr = mock_github

    # 1. User prompts Open WebUI
    pipe = Pipe()
    pipe_response = pipe.pipe(
        {
            "messages": [{"role": "user", "content": "Desenhar modelo de diagramas e arquitetura"}],
            "stream": False,
        }
    )
    assert "```mermaid" in pipe_response
    assert "erDiagram" in pipe_response

    # 2. Databricks Catalog Introspection
    client = DatabricksCEClient(host="", token="")  # offline demo mode
    entities = introspect_catalog(catalog="main", schema="default", client=client)
    assert len(entities) == 3
    entity_names = [e.name for e in entities]
    assert "bronze_raw_transactions" in entity_names
    assert "silver_transactions" in entity_names
    assert "gold_sales_kpis" in entity_names

    # 3. Semantic Modeling & Business Query Compilation
    registry = SemanticRegistry(models_dir=sample_yaml_dir)
    assert "corporate_credit" in registry.domains
    credit_entity = registry.get_entity("credit_facility")
    assert credit_entity is not None

    compiler = SemanticQueryCompiler(registry)
    sql = compiler.compile_query(
        entity_name="credit_facility",
        metric_names=["total_exposure"],
        group_by_dims=["operation_type"],
    )
    assert "SELECT" in sql
    assert "FROM" in sql
    assert "GROUP BY" in sql

    # 4. Mermaid Visualizer: ERD and Lineage
    erd_markdown = generate_er_diagram(registry.list_entities())
    assert "erDiagram" in erd_markdown
    assert "counterpart" in erd_markdown
    assert "credit_facility" in erd_markdown

    lineage_markdown = generate_lineage_diagram(entities)
    assert "graph LR" in lineage_markdown
    assert "subgraph Bronze" in lineage_markdown
    assert "subgraph Silver" in lineage_markdown
    assert "subgraph Gold" in lineage_markdown

    # 5. Modular ETL Generation for Silver Layer
    pipeline = generate_medallion_pipeline(credit_entity, layer="silver")
    assert pipeline.layer == "silver"
    assert "def run_silver_pipeline" in pipeline.pyspark_code
    assert "MERGE INTO" in pipeline.sparksql_code

    # 6. CI Quality Gate
    ci_report = run_ci_pipeline(
        pyspark_code=pipeline.pyspark_code,
        sparksql_code=sql,
    )
    assert ci_report.is_approved is True
    assert ci_report.ruff_status == "PASSED"
    assert ci_report.sqlfluff_status == "PASSED"
    assert len(ci_report.anti_patterns) == 0

    # 7. GitOps Automation & GitHub Pull Request
    files_to_commit = {
        "pipelines/credit_facility/silver_etl.py": pipeline.pyspark_code,
        "pipelines/credit_facility/silver_schema.sql": pipeline.sparksql_code,
    }
    gitops_result = create_data_product_pr(
        product_name="corporate-credit-risk",
        files=files_to_commit,
        ci_report=ci_report,
        diagram_md=erd_markdown,
        repo=mock_repo,
    )

    assert gitops_result.status == "success"
    assert gitops_result.branch_name == "feature/data-product-corporate-credit-risk"
    assert gitops_result.pr_number == 42
    assert "```mermaid" in gitops_result.body
    assert "CI Quality Gate" in gitops_result.body
    mock_repo.create_pull.assert_called_once()


# ==============================================================================
# Scenario 2: Anti-Pattern Detection & Strict Safety Gating
# ==============================================================================


def test_e2e_scenario_anti_pattern_rejection(flawed_pyspark_collect_code, mock_github):
    """Scenario 2: Code with dangerous anti-patterns is rejected and blocked from GitOps.

    Flow:
    1. Generated or user-submitted code contains an unbounded .collect() anti-pattern.
    2. CI Quality Gate executes static AST analysis and Ruff.
    3. CI Runner issues an explicit REJECTED report with diagnostic line numbers.
    4. GitOps Manager refuses to create a branch or open a Pull Request.
    """
    _mock_gh, mock_repo, _mock_pr = mock_github

    # 1 & 2. Execute CI quality gate on anti-pattern code
    ci_report = run_ci_pipeline(pyspark_code=flawed_pyspark_collect_code)

    # 3. Assert rejection
    assert ci_report.is_approved is False
    assert len(ci_report.anti_patterns) >= 1
    assert ci_report.anti_patterns[0].rule == "SPARK-ANTI-001"
    assert "collect()" in ci_report.anti_patterns[0].message
    assert "REJECTED" in ci_report.summary_markdown

    # 4. Attempt GitOps PR creation
    gitops_result = create_data_product_pr(
        product_name="risky-pipeline",
        files={"etl.py": flawed_pyspark_collect_code},
        ci_report=ci_report,
        repo=mock_repo,
    )

    assert gitops_result.status == "rejected"
    assert gitops_result.pr_number is None
    # Verify GitHub API was never called
    mock_repo.create_pull.assert_not_called()


# ==============================================================================
# Scenario 3: Offline Demo Mode Resilience
# ==============================================================================


def test_e2e_scenario_offline_demo_mode_resilience():
    """Scenario 3: Verifies complete resilience when running offline without live credentials.

    Flow:
    1. Databricks client detects missing credentials.
    2. Gracefully falls back to demo mock entities without throwing uncaught exceptions.
    3. Generates diagrams and Bronze/Silver/Gold ETL cleanly.
    """
    client = DatabricksCEClient(host="", token="")
    assert not client.is_configured()

    entities = introspect_catalog(catalog="main", schema="default", client=client)
    assert len(entities) == 3

    # Generate Medallion lineage diagram offline
    lineage = generate_lineage_diagram(entities)
    assert "graph LR" in lineage
    assert "Bronze" in lineage
    assert "Silver" in lineage
    assert "Gold" in lineage

    # Generate all three medallion layers offline
    for ent in entities:
        bronze_pipe = generate_medallion_pipeline(ent, layer="bronze")
        silver_pipe = generate_medallion_pipeline(ent, layer="silver")
        gold_pipe = generate_medallion_pipeline(ent, layer="gold")

        assert "run_bronze_pipeline" in bronze_pipe.pyspark_code
        assert "run_silver_pipeline" in silver_pipe.pyspark_code
        assert "run_gold_pipeline" in gold_pipe.pyspark_code


# ==============================================================================
# Scenario 4: Multi-Hop Semantic Query Compilation & Join Resolution
# ==============================================================================


def test_e2e_scenario_sales_lakehouse_multi_hop_query(sample_yaml_dir):
    """Scenario 4: Multi-hop query compilation across sales fact and customer dimension.

    Flow:
    1. Load sales lakehouse domain ontology.
    2. Request total_revenue metric from order entity grouped by customer_segment.
    3. Verify compiler automatically detects customer_segment on customer entity,
       finds shortest path join via customer_id, and emits valid SparkSQL with LEFT JOIN.
    """
    registry = SemanticRegistry(models_dir=sample_yaml_dir)
    compiler = SemanticQueryCompiler(registry)

    sql = compiler.compile_query(
        entity_name="order",
        metric_names=["total_revenue"],
        group_by_dims=["customer_segment"],
    )

    assert "SELECT" in sql
    assert "customer.segment AS customer_segment" in sql
    assert "SUM(order_amount) AS total_revenue" in sql
    assert "FROM main.retail.fct_orders AS order" in sql
    assert "LEFT JOIN main.retail.dim_customers AS customer" in sql
    assert "ON order.customer_id = customer.customer_id" in sql
    assert "GROUP BY customer.segment" in sql
