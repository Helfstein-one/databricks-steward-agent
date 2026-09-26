from typing import Any

from langchain_core.messages import AIMessage

from src.agent.intent import _extract_table_or_entity
from src.agent.state import AgentState
from src.config import settings
from src.semantic.models import EntityModel
from src.semantic.registry import SemanticRegistry
from src.visualizer.mermaid import generate_etl_flowchart


class ETLGenerationUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_query = messages[-1].get("content", "") if messages and isinstance(messages[-1], dict) else getattr(messages[-1], "content", "") if messages else state.get("user_query", "")
        thread_id = state.get("thread_id", "default_thread")

        entity_name = _extract_table_or_entity(user_query)

        layer = "silver"
        if "gold" in user_query.lower() or "business" in user_query.lower():
            layer = "gold"
        elif "bronze" in user_query.lower():
            layer = "bronze"

        reg = SemanticRegistry(settings.semantic_models_path)
        if not entity_name:
            if layer == "gold":
                entity_name = "medallion_gold_sales_kpis"
            else:
                entity_name = "medallion_silver_transactions"

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

        # Save checkpoint of the proposal
        try:
            from src.agent.checkpoint import get_checkpoint_manager
            chk_mgr = get_checkpoint_manager()
            saved_chk = chk_mgr.save_checkpoint(
                thread_id=thread_id,
                entity_name=entity_name,
                pyspark_code="",
                sparksql_code="",
                ci_status="PROPOSED",
                metadata={"table_name": table_name, "layer": layer, "flowchart": flowchart_code},
            )
            chk_id = saved_chk.get("checkpoint_id", "")
            chk_msg = f"\n💾 **Proposta v1 salva no banco de histórico conversacional (`{chk_id}`)**"
        except ImportError:
            chk_msg = ""

        confirmation_prompt = (
            "\n\n---\n"
            "❓ **Deseja aprovar esta proposta de ETL e gerar o código PySpark/SQL com validação de CI e deploy?**\n"
            "👉 *Digite **'sim'** ou **'confirmar'** para executar o ciclo de vida completo!*"
        )
        response_text = (
            f"### 🎨 Proposta Visual de Pipeline ETL ({layer.upper()} Layer): {table_name}\n\n"
            f"{flowchart_md}"
            f"{chk_msg}"
            f"{confirmation_prompt}"
        )

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": flowchart_md,
            "pending_pipeline": pending_pipeline,
        }
