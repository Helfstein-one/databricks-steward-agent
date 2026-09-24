"""Tests for the autonomous closed-loop Data Product lifecycle.

Covers:
1. Table data preview from Databricks SQL Warehouse.
2. Direct GitOps commit and push to main.
3. Dynamic Semantic Layer registration (dimensions, metrics, relationships).
4. Full deploy_and_materialize_data_product orchestrator tool.
5. LangGraph routing for preview and confirmation.
"""

from unittest.mock import MagicMock, patch

from src.agent.graph import (
    _is_confirmation,
    _is_data_preview_query,
    steward_node,
)
from src.agent.state import AgentState
from src.agent.tools import deploy_and_materialize_data_product, preview_table_data
from src.databricks.client import DatabricksCEClient
from src.gitops.git_client import GitClient
from src.semantic.registry import SemanticRegistry

# ==============================================================================
# 1. Preview Table Data Tests
# ==============================================================================


def test_preview_table_data_formatting():
    """Verify table preview executes query and formats result as Markdown table."""
    mock_db = MagicMock(spec=DatabricksCEClient)
    mock_db.preview_table_data.return_value = {
        "table_name": "workspace.default.medallion_silver_transactions",
        "markdown_table": (
            "| transaction_id | customer_id | amount |\n"
            "| --- | --- | --- |\n"
            "| TXN001 | CUST10 | 150.50 |\n"
            "| TXN002 | CUST20 | 89.90 |"
        ),
        "row_count": 2,
    }

    with patch("src.adapters.databricks_adapter.DatabricksCEClient", return_value=mock_db):
        res = preview_table_data(
            table_name="workspace.default.medallion_silver_transactions",
            limit=2,
        )

    assert isinstance(res, str)
    assert "| transaction_id |" in res
    assert "TXN001" in res
    mock_db.preview_table_data.assert_called_once_with(
        table_name="workspace.default.medallion_silver_transactions",
        limit=2,
    )


def test_client_preview_table_data_query_execution():
    """Verify DatabricksCEClient.preview_table_data constructs query and parses columns."""
    with patch("src.databricks.client.WorkspaceClient"):
        client = DatabricksCEClient(host="https://fake.cloud.databricks.com", token="fake-token")

    mock_col1 = MagicMock()
    mock_col1.name = "id"
    mock_col2 = MagicMock()
    mock_col2.name = "total"

    mock_schema = MagicMock()
    mock_schema.columns = [mock_col1, mock_col2]
    mock_manifest = MagicMock()
    mock_manifest.schema = mock_schema
    mock_result = MagicMock()
    mock_result.data_array = [["1", "100.0"], ["2", "250.5"]]

    mock_resp = {
        "manifest": mock_manifest,
        "result": mock_result,
    }

    with (
        patch.object(client, "get_default_warehouse_id", return_value="wh-999"),
        patch.object(client, "execute_query", return_value=mock_resp),
    ):
        res_dict = client.preview_table_data("catalog.schema.tbl", limit=5)

    table_md = res_dict.get("markdown_table", "")
    assert "| id | total |" in table_md
    assert "| 1 | 100.0 |" in table_md
    assert "| 2 | 250.5 |" in table_md


# ==============================================================================
# 2. Direct GitOps Commit and Push to Main
# ==============================================================================


def test_gitops_commit_and_push_to_main(tmp_path):
    """Verify commit_and_push_to_main stages files, commits and pushes to origin/main."""
    git_client = GitClient(repo_path=str(tmp_path))
    mock_repo = MagicMock()
    mock_repo.is_dirty.return_value = True
    mock_origin = MagicMock()
    mock_repo.remote.return_value = mock_origin

    with patch.object(git_client, "_run_git") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "a1b2c3d\n"
        mock_run.return_value = mock_proc

        res = git_client.commit_and_push_to_main(
            files={"src/etl/gold.py": "# Code", "configs/semantic_models/m.yaml": "yaml"},
            message="feat(gold): add daily metrics",
        )

    assert res["push_success"] is True
    assert res["branch"] == "main"
    assert "src/etl/gold.py" in res["files"]


# ==============================================================================
# 3. Dynamic Semantic Layer Registration
# ==============================================================================


def test_register_data_product_entity(tmp_path):
    """Verify dynamic registration appends new entity, dimensions, metrics, relationships."""
    yaml_content = """domain: test_domain
description: Test domain
entities:
  - name: customers
    table_name: workspace.default.customers
    primary_key: customer_id
    dimensions:
      - name: customer_id
        type: string
        description: Customer ID
relationships: []
"""
    model_file = tmp_path / "test_model.yaml"
    model_file.write_text(yaml_content)

    registry = SemanticRegistry(models_dir=str(tmp_path))
    assert len(registry.entities) == 1

    columns = [
        {"name": "customer_id", "type": "string"},
        {"name": "segment", "type": "string"},
        {"name": "total_revenue", "type": "double"},
        {"name": "order_count", "type": "bigint"},
    ]

    updated = registry.register_data_product_entity(
        entity_name="customer_summary",
        table_name="workspace.default.gold_customer_summary",
        columns=columns,
        source_entity="customers",
        domain_name="test_domain",
        models_dir=str(tmp_path),
    )

    assert updated is not None
    assert "customer_summary" in registry.entities

    summary_entity = registry.get_entity("customer_summary")
    assert summary_entity is not None
    assert summary_entity.table_name == "workspace.default.gold_customer_summary"
    assert summary_entity.primary_key == "customer_id"

    dim_names = [d.name for d in summary_entity.dimensions]
    assert "customer_id" in dim_names
    assert "segment" in dim_names

    metric_names = [m.name for m in summary_entity.metrics]
    assert any("sum" in m or "total" in m for m in metric_names)
    assert any("count" in m for m in metric_names)


# ==============================================================================
# 4. Deploy and Materialize Data Product Orchestrator
# ==============================================================================


def test_deploy_and_materialize_data_product_success(tmp_path):
    """Verify deploy_and_materialize_data_product runs CI, GitOps, Job and Semantic registration."""
    mock_ci = MagicMock()
    mock_ci.is_approved = True

    mock_git = MagicMock()
    mock_git.commit_and_push_to_main.return_value = {
        "status": "success",
        "commit_hash": "a1b2c3d",
        "branch": "main",
        "files": ["test.py"],
    }

    mock_db = MagicMock()
    mock_db.create_or_update_pipeline_job.return_value = {
        "status": "created",
        "job_id": "job-888",
    }
    mock_db.preview_table_data.return_value = {"columns": ["id", "metric_val"]}

    mock_registry = MagicMock()
    mock_registry.register_data_product_entity.return_value = True

    with (
        patch("src.ci.runner.run_ci_pipeline", return_value=mock_ci),
        patch("src.gitops.git_client.GitClient", return_value=mock_git),
        patch("src.adapters.databricks_adapter.DatabricksCEClient", return_value=mock_db),
        patch("src.semantic.registry.SemanticRegistry", return_value=mock_registry),
        patch("src.visualizer.mermaid.generate_er_diagram", return_value="erDiagram"),
    ):
        result = deploy_and_materialize_data_product(
            product_name="gold_summary",
            sparksql_code="SELECT id, SUM(val) as metric_val FROM base GROUP BY id",
            pyspark_code="# pyspark pipeline",
            source_entity="base_entity",
        )

    assert result["status"] == "success"
    assert result["job_result"]["job_id"] == "job-888"


# ==============================================================================
# 5. Graph Intent & Routing
# ==============================================================================


def test_intent_detection():
    """Verify regex intent matching for confirmation and data preview queries."""
    assert _is_confirmation("sim") is True
    assert _is_confirmation("confirmar") is True
    assert _is_confirmation("prosseguir") is True
    assert _is_confirmation("executar") is True
    assert _is_confirmation("qual a modelagem?") is False

    assert (
        _is_data_preview_query("consultar dados da tabela medallion_silver_transactions limit 3")
        is True
    )
    assert _is_data_preview_query("preview da tabela customers") is True
    assert _is_data_preview_query("ver dados de sales.orders") is True
    assert _is_data_preview_query("gerar pipeline etl gold") is False


def test_steward_node_routing_preview():
    """Verify data preview routing in steward_node invokes preview_table_data."""
    mock_preview = "### 📊 Amostra de Dados da Tabela"
    state: AgentState = {
        "messages": [
            {
                "role": "user",
                "content": "consultar dados da tabela medallion_silver_transactions limit 5",
            }
        ],
        "user_query": "consultar dados da tabela medallion_silver_transactions limit 5",
        "catalog_context": None,
        "semantic_entities": [],
        "active_diagram": None,
        "generated_code": None,
        "ci_report": None,
        "gitops_result": None,
        "response": None,
    }

    with patch("src.agent.tools.preview_table_data", return_value=mock_preview):
        result_state = steward_node(state)

    assert "Amostra de Dados" in result_state.get("response", "")
