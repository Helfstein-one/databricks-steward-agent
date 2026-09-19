from typing import Any

from src.ci.runner import run_ci_pipeline
from src.domain.ports.ci_port import ICiAdapter


class CiAdapter(ICiAdapter):
    def run_pipeline(self, pyspark_code: str, sparksql_code: str) -> Any:
        return run_ci_pipeline(pyspark_code, sparksql_code)
