from abc import ABC, abstractmethod
from typing import Any


class IDatabricksAdapter(ABC):
    @abstractmethod
    def execute_query(self, query: str) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def preview_table(self, table_name: str, limit: int = 10) -> str:
        pass

    @abstractmethod
    def inspect_schema(self, catalog: str | None = None, schema: str | None = None) -> str:
        pass

    @abstractmethod
    def deploy_job(
        self,
        product_name: str,
        pyspark_code: str,
        sparksql_code: str,
        source_entity: str | None = None,
    ) -> dict[str, Any]:
        pass
