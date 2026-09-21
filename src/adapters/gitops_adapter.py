from typing import Any

from src.domain.ports.gitops_port import IGitOpsAdapter
from src.gitops.github_pr import create_data_product_pr


def open_pull_request(
    product_name: str,
    pyspark_code: str,
    sparksql_code: str,
    ci_report: Any,
    active_diagram: str,
) -> dict[str, Any]:
    files = {
        f"pipelines/{product_name}/etl.py": pyspark_code,
        f"pipelines/{product_name}/schema.sql": sparksql_code,
    }
    res = create_data_product_pr(
        product_name=product_name,
        files=files,
        ci_report=ci_report,
        diagram_md=active_diagram,
        dry_run=True,
    )
    return res.model_dump() if hasattr(res, "model_dump") else res.__dict__


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
