from typing import Any

from src.agent.tools import deploy_and_materialize_data_product, inspect_unity_catalog
from src.databricks.client import DatabricksCEClient
from src.domain.ports.databricks_port import IDatabricksAdapter

_client_instance = DatabricksCEClient()


def execute_query(query: str) -> list[dict[str, Any]]:
    try:
        res = _client_instance.execute_query(query)
        if isinstance(res, dict) and "error" in res:
            return [res]
        raw_res = res.get("result") if isinstance(res, dict) else None
        return getattr(raw_res, "data_array", []) if raw_res else []
    except Exception as e:
        return [{"error": f"❌ Erro ao consultar Databricks: {e}"}]


def preview_table_data(table_name: str, limit: int = 10) -> str:
    try:
        res = _client_instance.preview_table_data(table_name, limit)
        if isinstance(res, dict) and "error" in res:
            return str(res["error"])
        return str(res.get("markdown_table", "")) if isinstance(res, dict) else str(res)
    except Exception as e:
        return f"❌ Erro ao consultar Databricks: {e}"


class DatabricksAdapter(IDatabricksAdapter):
    def execute_query(self, query: str) -> list[dict[str, Any]]:
        return execute_query(query)

    def preview_table(self, table_name: str, limit: int = 10) -> str:
        return preview_table_data(table_name, limit)

    def inspect_schema(self) -> str:
        try:
            return inspect_unity_catalog()
        except Exception as e:
            return f"❌ Erro ao consultar Databricks: {e}"

    def deploy_job(
        self, product_name: str, pyspark_code: str, sparksql_code: str
    ) -> dict[str, Any]:
        try:
            return deploy_and_materialize_data_product(product_name, pyspark_code, sparksql_code)
        except Exception as e:
            return {"status": "error", "error": f"❌ Erro ao consultar Databricks: {e}"}
