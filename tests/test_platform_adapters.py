from unittest.mock import MagicMock, patch

from src.adapters.ci_adapter import CiAdapter
from src.adapters.databricks_adapter import DatabricksAdapter, execute_query, preview_table_data
from src.adapters.gitops_adapter import GitOpsAdapter, open_pull_request


def test_databricks_adapter_execute_query():
    adapter = DatabricksAdapter()
    expected = [{"col": "val"}]
    with patch("src.adapters.databricks_adapter.execute_query", return_value=expected) as mock_exec:
        res = adapter.execute_query("SELECT 1")
        mock_exec.assert_called_once_with("SELECT 1")
        assert res == expected


def test_databricks_adapter_preview_table():
    adapter = DatabricksAdapter()
    expected = "table preview data"
    with patch(
        "src.adapters.databricks_adapter.preview_table_data", return_value=expected
    ) as mock_preview:
        res = adapter.preview_table("catalog.schema.table", limit=20)
        mock_preview.assert_called_once_with("catalog.schema.table", 20)
        assert res == expected


def test_databricks_adapter_inspect_schema():
    adapter = DatabricksAdapter()
    expected = "unity catalog schema"
    with patch(
        "src.adapters.databricks_adapter.inspect_unity_catalog", return_value=expected
    ) as mock_inspect:
        res = adapter.inspect_schema()
        mock_inspect.assert_called_once()
        assert res == expected


def test_databricks_adapter_deploy_job():
    adapter = DatabricksAdapter()
    expected = {"status": "deployed"}
    with patch(
        "src.adapters.databricks_adapter.deploy_and_materialize_data_product",
        return_value=expected,
    ) as mock_deploy:
        res = adapter.deploy_job("prod_a", "print('spark')", "SELECT 1")
        mock_deploy.assert_called_once_with("prod_a", "print('spark')", "SELECT 1")
        assert res == expected


def test_databricks_adapter_helper_functions():
    with patch("src.adapters.databricks_adapter._client_instance") as mock_client:
        mock_client.execute_query.return_value = {
            "result": MagicMock(data_array=[{"id": 1}])
        }
        res_exec = execute_query("SELECT 1")
        assert res_exec == [{"id": 1}]

        mock_client.preview_table_data.return_value = {"markdown_table": "| col |\n|---|"}
        res_prev = preview_table_data("tbl", 5)
        assert res_prev == "| col |\n|---|"


def test_gitops_adapter_commit_and_push():
    adapter = GitOpsAdapter()
    expected = {"pr_url": "https://github.com/pr/1"}
    with patch("src.adapters.gitops_adapter.open_pull_request", return_value=expected) as mock_pr:
        res = adapter.commit_and_push("prod_a", "code_py", "code_sql", "ci_rep", "diagram_str")
        mock_pr.assert_called_once_with(
            "prod_a", "code_py", "code_sql", "ci_rep", "diagram_str"
        )
        assert res == expected


def test_gitops_adapter_open_pull_request_helper():
    mock_res = MagicMock()
    mock_res.model_dump.return_value = {"pr_url": "http://pr.com"}
    with patch("src.adapters.gitops_adapter.create_data_product_pr", return_value=mock_res):
        res = open_pull_request("prod_b", "py_code", "sql_code", None, "diagram")
        assert res == {"pr_url": "http://pr.com"}


def test_ci_adapter_run_pipeline():
    adapter = CiAdapter()
    expected = {"passed": True}
    with patch("src.adapters.ci_adapter.run_ci_pipeline", return_value=expected) as mock_ci:
        res = adapter.run_pipeline("code_py", "code_sql")
        mock_ci.assert_called_once_with("code_py", "code_sql")
        assert res == expected


def test_databricks_adapter_error_handling():
    adapter = DatabricksAdapter()

    # execute_query exception
    with patch("src.adapters.databricks_adapter._client_instance") as mock_client:
        mock_client.execute_query.side_effect = Exception("API Down")
        res = adapter.execute_query("SELECT 1")
        assert len(res) == 1
        assert "❌ Erro ao consultar Databricks: API Down" in res[0].get("error", "")

    # preview_table exception
    with patch("src.adapters.databricks_adapter._client_instance") as mock_client:
        mock_client.preview_table_data.side_effect = Exception("Table Not Found")
        res = adapter.preview_table("catalog.schema.missing_table")
        assert "❌ Erro ao consultar Databricks: Table Not Found" in res

    # inspect_schema exception
    with patch(
        "src.adapters.databricks_adapter.inspect_unity_catalog",
        side_effect=Exception("Unity Catalog Unreachable"),
    ):
        res = adapter.inspect_schema()
        assert "❌ Erro ao consultar Databricks: Unity Catalog Unreachable" in res

    # deploy_job exception
    with patch(
        "src.adapters.databricks_adapter.deploy_and_materialize_data_product",
        side_effect=Exception("Deployment Failed"),
    ):
        res = adapter.deploy_job("prod_err", "py", "sql")
        assert res.get("status") == "error"
        assert "❌ Erro ao consultar Databricks: Deployment Failed" in res.get("error", "")


def test_gitops_adapter_error_handling():
    adapter = GitOpsAdapter()
    import subprocess
    import requests

    # RequestException
    with patch(
        "src.adapters.gitops_adapter.create_data_product_pr",
        side_effect=requests.exceptions.RequestException("GitHub API timeout"),
    ):
        res = adapter.commit_and_push("prod", "py", "sql", None, "diagram")
        assert res.get("status") == "error"
        assert "❌ Erro GitOps: GitHub API timeout" in res.get("error", "")

    # SubprocessError
    with patch(
        "src.adapters.gitops_adapter.create_data_product_pr",
        side_effect=subprocess.SubprocessError("Git command failed"),
    ):
        res = adapter.commit_and_push("prod", "py", "sql", None, "diagram")
        assert res.get("status") == "error"
        assert "❌ Erro GitOps: Git command failed" in res.get("error", "")

    # General Exception
    with patch(
        "src.adapters.gitops_adapter.create_data_product_pr",
        side_effect=Exception("API Down"),
    ):
        res = adapter.commit_and_push("prod", "py", "sql", None, "diagram")
        assert res.get("status") == "error"
        assert "❌ Erro GitOps: API Down" in res.get("error", "")


def test_ci_adapter_error_handling():
    adapter = CiAdapter()
    with patch(
        "src.adapters.ci_adapter.run_ci_pipeline", side_effect=Exception("CI Execution Failed")
    ):
        res = adapter.run_pipeline("py_code", "sql_code")
        assert res.get("success") is False
        assert "❌ Erro CI: CI Execution Failed" in res.get("error", "")
