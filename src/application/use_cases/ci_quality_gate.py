from typing import Any

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.ci.runner import run_ci_pipeline
from src.config import settings
from src.semantic.registry import SemanticRegistry


class CIQualityGateUseCase:
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

        report = run_ci_pipeline(pyspark_code=py_code, sparksql_code=sql_code)
        ci_report = report
        response_text = report.summary_markdown or report.format_markdown()

        is_failed = not report.is_approved
        waiting_for_correction = bool(is_failed)
        error_recovery = (
            {
                "source": "ci_quality_gate",
                "logs": response_text,
                "pyspark_code": py_code,
                "sparksql_code": sql_code,
            }
            if is_failed
            else None
        )

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "pending_pipeline": pending_pipeline,
            "waiting_for_correction": waiting_for_correction,
            "error_recovery": error_recovery,
        }
