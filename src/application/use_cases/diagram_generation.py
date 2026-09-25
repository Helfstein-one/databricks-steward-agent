from typing import Any

from langchain_core.messages import AIMessage

from src.agent.intent import _extract_entity_from_query, _extract_query_text
from src.agent.state import AgentState
from src.agent.tools import generate_diagram
from src.config import settings
from src.semantic.registry import SemanticRegistry


class DiagramGenerationUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_query = state.get("user_query") or _extract_query_text(messages)
        q_lower = (user_query or "").strip().lower()

        active_diagram = state.get("active_diagram")
        generated_code = state.get("generated_code")
        ci_report = state.get("ci_report")
        gitops_result = state.get("gitops_result")
        pending_pipeline = state.get("pending_pipeline")

        reg = SemanticRegistry(settings.semantic_models_path)
        has_medallion = reg.get_domain("databricks_medallion") is not None
        ent = _extract_entity_from_query(user_query)
        if (
            "lineage" in q_lower
            or "fluxo" in q_lower
            or "medallion" in q_lower
            or "medalhao" in q_lower
            or "medalhão" in q_lower
        ):
            diag = generate_diagram(
                "lineage", domain="databricks_medallion" if has_medallion else None
            )
        elif ent:
            diag = generate_diagram("er", domain=ent)
        elif ("sales" in q_lower or "vendas" in q_lower) and not has_medallion:
            diag = generate_diagram("er", domain="sales_lakehouse")
        elif (
            "credit" in q_lower or "credito" in q_lower or "crédito" in q_lower
        ) and not has_medallion:
            diag = generate_diagram("er", domain="corporate_credit")
        elif has_medallion:
            diag = generate_diagram("er", domain="databricks_medallion")
        elif "sales" in q_lower or "vendas" in q_lower:
            diag = generate_diagram("er", domain="sales_lakehouse")
        else:
            diag = generate_diagram("er", domain="corporate_credit")

        active_diagram = diag
        response_text = diag

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "pending_pipeline": pending_pipeline,
        }
