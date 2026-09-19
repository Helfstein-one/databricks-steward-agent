from abc import ABC, abstractmethod
from typing import Any


class ICiAdapter(ABC):
    @abstractmethod
    def run_pipeline(self, pyspark_code: str, sparksql_code: str) -> Any:
        pass
