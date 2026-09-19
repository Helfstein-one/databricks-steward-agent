from typing import Any

from src.agent.tools import deploy_and_materialize_data_product
from src.databricks.client import execute_query, preview_table_data
from src.databricks.introspector import inspect_unity_catalog
from src.domain.ports.databricks_port import IDatabricksAdapter


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
