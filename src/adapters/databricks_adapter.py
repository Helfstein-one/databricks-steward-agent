from typing import Any

from src.agent.tools import deploy_and_materialize_data_product, inspect_unity_catalog
from src.databricks.client import DatabricksCEClient
from src.domain.ports.databricks_port import IDatabricksAdapter

_client_instance = DatabricksCEClient()


def execute_query(query: str) -> list[dict[str, Any]]:
    res = _client_instance.execute_query(query)
    raw_res = res.get("result")
    return getattr(raw_res, "data_array", []) if raw_res else []


def preview_table_data(table_name: str, limit: int = 10) -> str:
    res = _client_instance.preview_table_data(table_name, limit)
    return str(res.get("markdown_table", ""))


class DatabricksAdapter(IDatabricksAdapter):
    def execute_query(self, query: str) -> list[dict[str, Any]]:
        return execute_query(query)

    def preview_table(self, table_name: str, limit: int = 10) -> str:
        return preview_table_data(table_name, limit)

    def inspect_schema(self) -> str:
        return inspect_unity_catalog()

    def deploy_job(
        self, product_name: str, pyspark_code: str, sparksql_code: str
    ) -> dict[str, Any]:
        return deploy_and_materialize_data_product(product_name, pyspark_code, sparksql_code)
