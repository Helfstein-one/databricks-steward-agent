from abc import ABC, abstractmethod
from typing import Any


class IGitOpsAdapter(ABC):
    @abstractmethod
    def commit_and_push(
        self,
        product_name: str,
        pyspark_code: str,
        sparksql_code: str,
        ci_report: Any,
        active_diagram: str,
    ) -> dict[str, Any]:
        pass
