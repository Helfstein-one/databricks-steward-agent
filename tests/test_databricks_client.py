"""Unit and component integration tests for Databricks client and Unity Catalog (Tier 2)."""

from unittest.mock import MagicMock, patch

import pytest

from src.databricks.client import (
    DatabricksCEClient,
    DatabricksConnectionError,
)
from src.databricks.introspector import introspect_catalog


def test_client_unconfigured():
    """Verify unconfigured client reports is_configured as False."""
    client = DatabricksCEClient(host="", token="")
    assert client.is_configured() is False

    with pytest.raises(DatabricksConnectionError):
        client.verify_connection()

    with pytest.raises(DatabricksConnectionError):
        client.mkdirs("/Workspace/test")


@patch("src.databricks.client.WorkspaceClient")
def test_client_configured_and_verify_connection(mock_wc_class):
    """Verify configured client initializes WorkspaceClient and verifies connection."""
    mock_instance = MagicMock()
    mock_wc_class.return_value = mock_instance

    mock_user = MagicMock()
    mock_user.user_name = "data.engineer@lakehouse.com"
    mock_instance.current_user.me.return_value = mock_user

    client = DatabricksCEClient(
        host="https://adb-123456.azuredatabricks.net",
        token="dapi-secure-token",
    )

    assert client.is_configured() is True
    res = client.verify_connection()
    assert res["status"] == "connected"
    assert res["user"] == "data.engineer@lakehouse.com"
    assert res["host"] == "https://adb-123456.azuredatabricks.net"


@patch("src.databricks.client.WorkspaceClient")
def test_client_upload_file(mock_wc_class, tmp_path):
    """Verify base64 encoding and workspace upload of local file."""
    mock_instance = MagicMock()
    mock_wc_class.return_value = mock_instance

    test_file = tmp_path / "sample_pipeline.py"
    test_file.write_text("print('hello lakehouse')", encoding="utf-8")

    client = DatabricksCEClient(host="https://test.databricks.com", token="token")
    result = client.upload_file(
        local_path=test_file,
        remote_workspace_path="/Workspace/pipelines/sample_pipeline.py",
        language="PYTHON",
    )

    assert result["status"] == "success"
    assert mock_instance.workspace.import_.called
    _args, kwargs = mock_instance.workspace.import_.call_args
    assert kwargs["path"] == "/Workspace/pipelines/sample_pipeline.py"
    assert kwargs["language"] == "PYTHON"
    assert kwargs["content"] != ""


@patch("src.databricks.client.WorkspaceClient")
def test_client_upload_nonexistent_file(mock_wc_class):
    """Verify FileNotFoundError is raised when uploading missing local file."""
    client = DatabricksCEClient(host="https://test.databricks.com", token="token")
    client.client = MagicMock()

    with pytest.raises(FileNotFoundError):
        client.upload_file(
            local_path="/non/existent/file.py",
            remote_workspace_path="/Workspace/file.py",
        )


@patch("src.databricks.client.WorkspaceClient")
def test_client_execute_query(mock_wc_class):
    """Verify execution of SQL statement on Databricks SQL Warehouse."""
    mock_instance = MagicMock()
    mock_wc_class.return_value = mock_instance

    mock_resp = MagicMock()
    mock_resp.statement_id = "stmt-12345"
    mock_resp.status.state = "SUCCEEDED"
    mock_instance.statement_execution.execute_statement.return_value = mock_resp

    client = DatabricksCEClient(
        host="https://test.databricks.com",
        token="token",
        warehouse_id="warehouse-abc",
    )

    result = client.execute_query("SELECT 1")
    assert result["statement_id"] == "stmt-12345"
    assert result["status"] == "SUCCEEDED"


def test_introspect_catalog_offline_mock_fallback():
    """Verify introspect_catalog falls back to demo mock entities when disconnected."""
    client = DatabricksCEClient(host="", token="")
    entities = introspect_catalog(catalog="main", schema="default", client=client)

    assert len(entities) == 3
    entity_names = [e.name for e in entities]
    assert "bronze_raw_transactions" in entity_names
    assert "silver_transactions" in entity_names
    assert "gold_sales_kpis" in entity_names

    # Verify Bronze entity properties
    bronze = next(e for e in entities if e.name == "bronze_raw_transactions")
    assert bronze.layer == "bronze"
    assert bronze.primary_key == "transaction_id"

    # Verify Silver entity properties
    silver = next(e for e in entities if e.name == "silver_transactions")
    assert silver.layer == "silver"
    assert len(silver.relationships) == 1

    # Verify Gold entity properties
    gold = next(e for e in entities if e.name == "gold_sales_kpis")
    assert gold.layer == "gold"
    assert len(gold.metrics) >= 1


@patch("src.databricks.client.WorkspaceClient")
def test_introspect_catalog_with_mock_workspace_client(mock_wc_class, mock_workspace_client):
    """Verify introspect_catalog maps Unity Catalog tables into EntityModels."""
    client = DatabricksCEClient(host="https://test.databricks.com", token="token")
    client.client = mock_workspace_client

    entities = introspect_catalog(catalog="main", schema="default", client=client)
    assert len(entities) == 3
    entity_names = [e.name for e in entities]
    assert "bronze_raw_transactions" in entity_names
    assert "silver_transactions" in entity_names
    assert "gold_sales_kpis" in entity_names

    for ent in entities:
        assert ent.catalog == "main"
        assert ent.schema_name == "default"
        assert len(ent.columns) > 0


@patch("src.databricks.client.WorkspaceClient")
def test_introspect_catalog_error_fallback(mock_wc_class, mock_workspace_client):
    """Verify introspect_catalog falls back to mock entities if API raises an error."""
    mock_workspace_client.tables.list.side_effect = RuntimeError("Databricks API Timeout")

    client = DatabricksCEClient(host="https://test.databricks.com", token="token")
    client.client = mock_workspace_client

    entities = introspect_catalog(catalog="main", schema="default", client=client)
    assert len(entities) == 3
    entity_names = [e.name for e in entities]
    assert "bronze_raw_transactions" in entity_names
