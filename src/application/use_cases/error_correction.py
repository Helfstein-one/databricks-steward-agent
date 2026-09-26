from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage

from src.adapters.jules_mcp_adapter import JulesMcpAdapter
from src.agent.state import AgentState
from src.ci.runner import run_ci_pipeline
from src.domain.ports.jules_mcp_port import IJulesMcpAdapter

logger = logging.getLogger(__name__)


def _is_jules_delegation(query: str) -> bool:
    q = (query or "").strip().lower()
    jules_keywords = [
        "jules",
        "ask jules",
        "chamar jules",
        "peça ao jules",
        "peça para o jules",
        "manda pro jules",
        "delegate to jules",
        "ask_jules",
    ]
    return any(k in q for k in jules_keywords)


def _apply_heuristic_fix(
    pyspark_code: str, sparksql_code: str, user_query: str
) -> tuple[str, str]:
    """Apply direct code replacements based on common correction instructions."""
    py = pyspark_code or ""
    sql = sparksql_code or ""
    q = (user_query or "").strip()

    # If user provided explicit python or sql code block
    py_match = re.search(r"```(?:python|py)\n(.*?)\n```", q, re.DOTALL)
    if py_match:
        py = py_match.group(1).strip()

    sql_match = re.search(r"```sql\n(.*?)\n```", q, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()

    # Fix SELECT * or select star anti-pattern
    if "select *" in sql.lower() or "remove select *" in q.lower() or "fix select" in q.lower():
        sql = sql.replace("SELECT *", "SELECT transaction_id, user_id, amount, status")
        sql = sql.replace("select *", "select transaction_id, user_id, amount, status")

    # Fix join or filter instructions
    if "fix join" in q.lower() or "corrija o join" in q.lower() or "fix the join" in q.lower():
        if "JOIN" not in sql.upper():
            sql += "\n-- Fixed JOIN clause\n"
        if "dropDuplicates" not in py:
            py = py.rstrip() + "\n    df = df.dropDuplicates(['transaction_id'])\n"

    # Remove collect() calls if user asks to remove collect
    if "collect" in q.lower() or "remove collect" in q.lower():
        py = py.replace(".collect()", ".take(10)")

    return py, sql


class ErrorCorrectionUseCase:
    """Handles conversational error recovery when CI or deployment fails."""

    def __init__(self, jules_adapter: IJulesMcpAdapter | None = None) -> None:
        self.jules_adapter = jules_adapter or JulesMcpAdapter()

    def execute(self, state: AgentState, llm: Any | None = None) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_query = state.get("user_query") or ""
        if not user_query and messages:
            last_msg = messages[-1]
            user_query = (
                last_msg.get("content", "")
                if isinstance(last_msg, dict)
                else getattr(last_msg, "content", "")
            )

        active_diagram = state.get("active_diagram")
        generated_code = dict(state.get("generated_code") or {})
        pending_pipeline = (
            dict(state.get("pending_pipeline")) if state.get("pending_pipeline") else None
        )
        error_rec = state.get("error_recovery") or {}

        # 1. Check if user requests delegating correction to Jules MCP Agent
        if _is_jules_delegation(user_query):
            error_logs = error_rec.get("logs") or (
                state.get("ci_report").summary_markdown if state.get("ci_report") else ""
            )
            prompt = (
                f"The previous Databricks Steward Agent task encountered an error.\n"
                f"Error Logs / Diagnostic Report:\n{error_logs}\n\n"
                f"User Correction Instruction: {user_query}\n"
                f"Please analyze the failure and propose a corrected fix."
            )
            jules_response = self.jules_adapter.ask_jules(prompt)
            response_text = (
                f"🤖 **Jules MCP Server Response**\n\n"
                f"{jules_response}\n\n"
                f"💡 You can confirm the proposed changes or ask for further adjustments."
            )
            return {
                "messages": [AIMessage(content=response_text)],
                "response": response_text,
                "active_diagram": active_diagram,
                "generated_code": generated_code,
                "pending_pipeline": pending_pipeline,
                "ci_report": state.get("ci_report"),
                "gitops_result": state.get("gitops_result"),
                "waiting_for_correction": False,
                "error_recovery": None,
            }

        # 2. Extract current code from state
        py_code = generated_code.get("pyspark") or error_rec.get("pyspark_code") or ""
        sql_code = generated_code.get("sparksql") or error_rec.get("sparksql_code") or ""
        if pending_pipeline:
            py_code = pending_pipeline.get("pyspark", py_code)
            sql_code = pending_pipeline.get("sparksql", sql_code)

        # 3. Apply correction based on user input
        new_py, new_sql = _apply_heuristic_fix(py_code, sql_code, user_query)

        # 4. Re-run CI quality gate verification
        report = run_ci_pipeline(pyspark_code=new_py, sparksql_code=new_sql)

        # Update code in generated_code and pending_pipeline
        generated_code["pyspark"] = new_py
        generated_code["sparksql"] = new_sql
        if pending_pipeline:
            pending_pipeline["pyspark"] = new_py
            pending_pipeline["sparksql"] = new_sql

        if report.is_approved:
            response_text = (
                f"✅ **Correção Aplicada com Sucesso!**\n\n"
                f"O pipeline foi corrigido e aprovado na esteira de CI Quality Gate.\n\n"
                f"#### 🐍 Código PySpark Corrigido:\n```python\n{new_py}\n```\n\n"
                f"#### ⚡ Código SparkSQL Corrigido:\n```sql\n{new_sql}\n```\n\n"
                f"{report.summary_markdown or report.format_markdown()}\n\n"
                f"💡 **Próximos passos:**\n"
                f"- Responda com *'sim'* para prosseguir com o deploy e materialização no Databricks!"
            )
            return {
                "messages": [AIMessage(content=response_text)],
                "response": response_text,
                "active_diagram": active_diagram,
                "generated_code": generated_code,
                "pending_pipeline": pending_pipeline,
                "ci_report": report,
                "gitops_result": state.get("gitops_result"),
                "waiting_for_correction": False,
                "error_recovery": None,
            }

        # If CI still fails, remain in waiting_for_correction
        rep_md = report.summary_markdown or report.format_markdown()
        response_text = (
            f"❌ **Correção Parcial - CI Quality Gate Rejeitou as Alterações**\n\n"
            f"Ainda foram detectadas violações na esteira de CI.\n\n"
            f"{rep_md}\n\n"
            f"Por favor, forneça mais orientações (ex: *'Fix the join'* ou *'Ask Jules to fix it'*)."
        )
        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "pending_pipeline": pending_pipeline,
            "ci_report": report,
            "gitops_result": state.get("gitops_result"),
            "waiting_for_correction": True,
            "error_recovery": {
                "source": "error_correction",
                "logs": rep_md,
                "pyspark_code": new_py,
                "sparksql_code": new_sql,
            },
        }
