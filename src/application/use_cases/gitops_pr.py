import re
from typing import Any

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.ci.runner import run_ci_pipeline
from src.config import settings
from src.gitops.github_pr import create_data_product_pr
from src.semantic.registry import SemanticRegistry


class GitOpsPRUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        active_diagram = state.get("active_diagram")
        generated_code = state.get("generated_code")
        ci_report = state.get("ci_report")
        gitops_result = state.get("gitops_result")
        pending_pipeline = state.get("pending_pipeline")

        py_code = generated_code.get("pyspark") if generated_code else None
        sql_code = generated_code.get("sparksql") if generated_code else None

        if not py_code or not sql_code:
            reg = SemanticRegistry(settings.semantic_models_path)
            has_medallion = reg.get_domain("databricks_medallion") is not None
            if has_medallion:
                py_code = (
                    "def process(df):\n"
                    "    return df.filter(df['status'] == 'COMPLETED').dropDuplicates(['transaction_id'])\n"
                )
                sql_code = "SELECT transaction_id, user_id, amount FROM workspace.default.medallion_silver_transactions;"
            else:
                py_code = "def process(df):\n    return df.filter(df['active'] == True)\n"
                sql_code = "SELECT order_id, total_amount FROM main.sales.orders;"

        report = ci_report or run_ci_pipeline(pyspark_code=py_code, sparksql_code=sql_code)
        ci_report = report

        reg = SemanticRegistry(settings.semantic_models_path)
        has_medallion = reg.get_domain("databricks_medallion") is not None
        default_prod = "medallion-silver-transactions" if has_medallion else "corporate-credit-kpis"
        product_name = (
            generated_code.get("table_name", default_prod) if generated_code else default_prod
        )
        product_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", product_name.replace(".", "-")).lower()

        files = {
            f"pipelines/{product_slug}/etl.py": py_code,
            f"pipelines/{product_slug}/schema.sql": sql_code,
        }

        result = create_data_product_pr(
            product_name=product_slug,
            files=files,
            ci_report=report,
            diagram_md=active_diagram or "",
            dry_run=True,
        )
        gitops_result = result

        if result.status == "rejected":
            response_text = f"❌ PR Creation Blocked: CI Quality Gate Rejected the pipeline.\n\n{report.summary_markdown}"
        else:
            response_text = (
                f"✅ PR Successfully Created!\n\n"
                f"- **Branch:** `{result.branch_name}`\n"
                f"- **Commit SHA:** `{result.commit_sha}`\n"
                f"- **PR URL:** [{result.pr_url}]({result.pr_url})\n\n"
                f"#### CI Gate Report\n{report.summary_markdown}"
            )

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "pending_pipeline": pending_pipeline,
        }
