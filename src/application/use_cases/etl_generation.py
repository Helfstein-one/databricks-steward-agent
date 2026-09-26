from typing import Any

from langchain_core.messages import AIMessage

from src.agent.intent import (
    _extract_entity_from_query,
    _extract_query_text,
    _resolve_anaphoric_entity,
)
from src.agent.state import AgentState
from src.config import settings
from src.semantic.models import EntityModel
from src.semantic.registry import SemanticRegistry
from src.visualizer.mermaid import generate_etl_flowchart


class ETLGenerationUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_query = state.get("user_query") or _extract_query_text(messages)
        q_lower = (user_query or "").strip().lower()

        generated_code = state.get("generated_code")
        ci_report = state.get("ci_report")
        gitops_result = state.get("gitops_result")

        layer = "gold" if "gold" in q_lower else "bronze" if "bronze" in q_lower else "silver"
        reg = SemanticRegistry(settings.semantic_models_path)
        has_medallion = reg.get_domain("databricks_medallion") is not None
        ent = _extract_entity_from_query(user_query)
        if not ent:
            ent = _resolve_anaphoric_entity(user_query, messages, state)

        if ent:
            if (
                has_medallion
                and layer == "gold"
                and ent in ("customers", "customer", "cliente", "clientes", "usuarios", "user")
            ):
                entity_name = "medallion_gold_customer_kpis"
            else:
                entity_name = ent
        elif has_medallion:
            if layer == "bronze":
                entity_name = "medallion_bronze_transactions"
            elif layer == "gold":
                entity_name = "medallion_gold_sales_kpis"
            else:
                entity_name = "medallion_silver_transactions"
        elif "sales" in q_lower or "order" in q_lower or "venda" in q_lower:
            entity_name = "orders"
        elif "transaction" in q_lower or "transac" in q_lower:
            entity_name = "silver_transactions" if layer == "silver" else "bronze_raw_transactions"
        else:
            entity_name = "facilities"

        ent_obj = reg.get_entity(entity_name)
        if not ent_obj:
            from src.databricks.introspector import _build_mock_entities

            mock_ents = {e.name: e for e in _build_mock_entities()}
            ent_obj = mock_ents.get(entity_name)

        if not ent_obj:
            ent_obj = EntityModel(name=entity_name, table_name=entity_name)

        table_name = ent_obj.table_name or ent_obj.name
        flowchart_code = generate_etl_flowchart(ent_obj, layer=layer)
        flowchart_md = f"```mermaid\n{flowchart_code}\n```"

        pending_pipeline = {
            "product_name": table_name,
            "layer": layer,
            "source_entity": entity_name,
            "entity_name": ent_obj.name,
            "flowchart": flowchart_code,
            "status": "proposed",
        }

        confirmation_prompt = (
            "\n\n---\n"
            "❓ **Deseja aprovar esta proposta de ETL e gerar o código PySpark/SQL com validação de CI e deploy?**\n"
            "👉 *Digite **'sim'** ou **'confirmar'** para executar o ciclo de vida completo!*"
        )
        response_text = (
            f"### 🎨 Proposta Visual de Pipeline ETL ({layer.upper()} Layer): {table_name}\n\n"
            f"{flowchart_md}"
            f"{confirmation_prompt}"
        )

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": flowchart_md,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "pending_pipeline": pending_pipeline,
        }
