"""Empirical stress testing and challenger verification for DatabricksCEClient."""

from unittest.mock import MagicMock, patch
import pytest

from databricks.sdk.service.jobs import Source, SqlTask, SqlTaskFile, Task
from databricks.sdk.service.workspace import ImportFormat, Language
from src.databricks.client import DatabricksCEClient, DatabricksClientError


def test_sdk_enums_existence_and_types() -> None:
    """Verify that required SDK enums are available and of expected types."""
    assert hasattr(ImportFormat, "AUTO")
    assert hasattr(Language, "SQL")
    assert hasattr(Language, "PYTHON")
    assert hasattr(Source, "WORKSPACE")
    assert isinstance(ImportFormat.AUTO, ImportFormat)
    assert isinstance(Language.SQL, Language)
    assert isinstance(Language.PYTHON, Language)
    assert isinstance(Source.WORKSPACE, Source)


def test_warehouse_resolution_explicit() -> None:
    """Verify explicit warehouse_id is preserved."""
    client = DatabricksCEClient(warehouse_id="explicit-wh-123")
    assert client.warehouse_id == "explicit-wh-123"
    assert client.get_default_warehouse_id() == "explicit-wh-123"


def test_warehouse_resolution_running_priority() -> None:
    """Verify running warehouse is prioritized over stopped ones."""
    client = DatabricksCEClient(host="https://fake.databricks.com", token="fake", warehouse_id="")
    client.warehouse_id = None
    client.client = MagicMock()

    wh_stopped = MagicMock(id="wh-stopped", state="State.STOPPED")
    wh_running = MagicMock(id="wh-running", state="State.RUNNING")
    client.client.warehouses.list.return_value = [wh_stopped, wh_running]

    resolved = client.get_default_warehouse_id()
    assert resolved == "wh-running"
    assert client.warehouse_id == "wh-running"


def test_warehouse_resolution_none_running_fallback() -> None:
    """Verify first warehouse is selected when none is running."""
    client = DatabricksCEClient(host="https://fake.databricks.com", token="fake", warehouse_id="")
    client.warehouse_id = None
    client.client = MagicMock()

    wh_stopped1 = MagicMock(id="wh-stopped-1", state="State.STOPPED")
    wh_stopped2 = MagicMock(id="wh-stopped-2", state="State.STOPPED")
    client.client.warehouses.list.return_value = [wh_stopped1, wh_stopped2]

    resolved = client.get_default_warehouse_id()
    assert resolved == "wh-stopped-1"


def test_warehouse_resolution_empty_list() -> None:
    """Verify empty string returned if no warehouses exist."""
    client = DatabricksCEClient(host="https://fake.databricks.com", token="fake", warehouse_id="")
    client.warehouse_id = None
    client.client = MagicMock()
    client.client.warehouses.list.return_value = []

    resolved = client.get_default_warehouse_id()
    assert resolved == ""


def test_create_or_update_pipeline_job_task_structure() -> None:
    """Verify create_or_update_pipeline_job creates Task with correct SDK structure."""
    client = DatabricksCEClient(host="https://fake.databricks.com", token="fake", warehouse_id="wh-123")
    client.client = MagicMock()

    mock_job = MagicMock(job_id="998877")
    client.client.jobs.create.return_value = mock_job

    client.execute_query = MagicMock(return_value={"status": "SUCCEEDED"})
    client.mkdirs = MagicMock(return_value={"status": "success"})
    client.run_job_now = MagicMock(return_value={"status": "triggered", "run_id": "run-101"})

    res = client.create_or_update_pipeline_job(
        job_name="TestJob",
        product_slug="test-slug",
        sql_statement="SELECT 1;",
        pyspark_code="print(1)",
    )

    assert res["status"] == "created_and_dispatched"
    assert res["job_id"] == "998877"

    # Verify workspace import arguments
    assert client.client.workspace.import_.call_count == 2
    sql_call = client.client.workspace.import_.call_args_list[0]
    assert sql_call.kwargs["format"] == ImportFormat.AUTO
    assert sql_call.kwargs["language"] == Language.SQL

    py_call = client.client.workspace.import_.call_args_list[1]
    assert py_call.kwargs["format"] == ImportFormat.AUTO
    assert py_call.kwargs["language"] == Language.PYTHON

    # Verify job creation task structure
    assert client.client.jobs.create.called
    job_kwargs = client.client.jobs.create.call_args.kwargs
    assert job_kwargs["name"] == "TestJob"
    tasks = job_kwargs["tasks"]
    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, Task)
    assert isinstance(task.sql_task, SqlTask)
    assert task.sql_task.warehouse_id == "wh-123"
    assert isinstance(task.sql_task.file, SqlTaskFile)
    assert task.sql_task.file.path == "/Shared/pipelines/test-slug/schema.sql"
    assert task.sql_task.file.source == Source.WORKSPACE


def test_preview_table_data_graceful_error_on_query_failure() -> None:
    """Verify preview_table_data returns graceful markdown when execute_query returns error dict."""
    client = DatabricksCEClient(host="https://fake.databricks.com", token="fake", warehouse_id="wh-123")
    client.client = MagicMock()

    client.execute_query = MagicMock(return_value={
        "status": "StatementState.FAILED",
        "error": "[TABLE_OR_VIEW_NOT_FOUND] Table foo does not exist",
    })

    res = client.preview_table_data("workspace.default.foo", limit=5)
    assert res["row_count"] == 0
    assert res["columns"] == []
    assert res["rows"] == []
    assert "❌ **Erro ao consultar Databricks:**" in res["markdown_table"]
    assert "TABLE_OR_VIEW_NOT_FOUND" in res["markdown_table"]


def test_preview_table_data_with_none_data_array_empty_table() -> None:
    """Empirical challenger test: when Databricks returns ResultData with data_array=None (0 rows).

    In Databricks SDK Statement Execution API, empty tables or queries returning 0 rows
    have result.data_array == None (not []).
    This reproduces the TypeError: object of type 'NoneType' has no len().
    """
    client = DatabricksCEClient(host="https://fake.databricks.com", token="fake", warehouse_id="wh-123")
    client.client = MagicMock()

    mock_result = MagicMock()
    mock_result.data_array = None  # Exact Databricks SDK behavior on 0 rows
    mock_manifest = MagicMock()
    col = MagicMock()
    col.name = "id"
    mock_manifest.schema.columns = [col]

    client.execute_query = MagicMock(return_value={
        "status": "StatementState.SUCCEEDED",
        "error": None,
        "result": mock_result,
        "manifest": mock_manifest,
    })

    # Verify that data_array=None does not raise TypeError and returns graceful empty table
    res = client.preview_table_data("workspace.default.reviewer_test_tab", limit=10)
    assert res["row_count"] == 0
    assert res["rows"] == []
    assert res["markdown_table"] == "*Tabela vazia ou nenhum dado retornado.*"


def test_preview_table_data_unhandled_client_error_on_warehouse_failure() -> None:
    """Empirical challenger test: execute_query raises DatabricksClientError on warehouse/connection failure.

    preview_table_data catches DatabricksClientError and returns a graceful markdown error dict.
    """
    client = DatabricksCEClient(host="https://fake.databricks.com", token="fake", warehouse_id="wh-123")
    client.client = MagicMock()
    client.execute_query = MagicMock(side_effect=DatabricksClientError("Warehouse endpoint is terminated"))

    res = client.preview_table_data("workspace.default.customers", limit=5)
    assert res["row_count"] == 0
    assert res["columns"] == []
    assert res["rows"] == []
    assert "❌ **Erro ao consultar Databricks:**" in res["markdown_table"]
    assert "Warehouse endpoint is terminated" in res["markdown_table"]

