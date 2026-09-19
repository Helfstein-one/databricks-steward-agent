from typing import Any

from src.domain.ports.gitops_port import IGitOpsAdapter
from src.gitops.github_pr import open_pull_request


class GitOpsAdapter(IGitOpsAdapter):
    def commit_and_push(
        self,
        product_name: str,
        pyspark_code: str,
        sparksql_code: str,
        ci_report: Any,
        active_diagram: str,
    ) -> dict[str, Any]:
        return open_pull_request(
            product_name, pyspark_code, sparksql_code, ci_report, active_diagram
        )
