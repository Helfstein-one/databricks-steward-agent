import re
from typing import Any

from langchain_core.messages import AIMessage

from src.application.query_parser import (
    extract_pending_from_history,
    extract_table_or_entity,
    resolve_anaphoric_entity,
)


class StewardAppService:
    def __init__(self, db_adapter, ci_adapter, git_adapter):
        self.db = db_adapter
        self.ci = ci_adapter
        self.git = git_adapter

    def execute(self, state: dict[str, Any], intent: str, user_query: str) -> dict[str, Any]:
        """Main Application Use Case Orchestrator."""
        messages = state.get("messages", [])
        q_lower = (user_query or "").strip().lower()

        active_diagram = state.get("active_diagram")
        generated_code = state.get("generated_code")
        ci_report = state.get("ci_report")
        gitops_result = state.get("gitops_result")
        pending_pipeline = state.get("pending_pipeline")
        response_text = "Opção Inválida."

        if intent == "PREVIEW":
            target_table = extract_table_or_entity(user_query)
            lim_match = re.search(r"\blimit\s+(\d+)\b", q_lower)
            limit_val = int(lim_match.group(1)) if lim_match else 10

            if not target_table:
                target_table = resolve_anaphoric_entity(user_query, messages, state)
            if not target_table:
                response_text = "❌ Nenhuma tabela identificada para consulta."
            else:
                response_text = self.db.preview_table(target_table, limit_val)

            return {
                "messages": [AIMessage(content=response_text)],
                "response": response_text,
                "active_diagram": active_diagram,
                "generated_code": generated_code,
                "ci_report": ci_report,
                "gitops_result": gitops_result,
                "pending_pipeline": pending_pipeline,
            }

        # Confirmation of pending ETL pipeline lifecycle
        if intent == "CONFIRM":
            pending = pending_pipeline or extract_pending_from_history(messages)
            prod_name = pending.get("product_name", "medallion_gold_sales_kpis")
            p_py = pending.get("pyspark", "")
            p_sql = pending.get("sparksql", "")
            pending.get("source_entity", "medallion_silver_transactions")

            res = self.db.deploy_job(product_name=prod_name, pyspark_code=p_py, sparksql_code=p_sql)
            # Add semantic models etc (Mocked logic omitted for brevity in adapter, needs to be handled)
            return res

        return {"response": "Not Implemented Yet"}
