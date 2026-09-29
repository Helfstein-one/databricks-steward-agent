from typing import Any

from langchain_core.messages import AIMessage

from src.agent.parsers import _extract_table_or_entity, _resolve_anaphoric_entity
from src.agent.state import AgentState
from src.config import settings
from src.semantic.models import EntityModel
from src.semantic.registry import SemanticRegistry
from src.visualizer.mermaid import generate_etl_flowchart
from src.etl.generator import generate_medallion_pipeline


class ETLGenerationUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_query = state.get("user_query") or (messages[-1].get("content", "") if messages and isinstance(messages[-1], dict) else getattr(messages[-1], "content", "") if messages else "")
        thread_id = state.get("thread_id", "default_thread")

        entity_name = _resolve_anaphoric_entity(user_query, messages, state)
        if not entity_name:
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
        if layer == "gold" and "silver" in table_name:
            table_name = table_name.replace("silver", "gold") + "_kpis"
        elif layer == "silver" and "bronze" in table_name:
            table_name = table_name.replace("bronze", "silver")
        ent_obj.table_name = table_name
        
        flowchart_code = generate_etl_flowchart(ent_obj, layer=layer)
        flowchart_md = f"```mermaid\n{flowchart_code}\n```"
        
        pipeline = generate_medallion_pipeline(ent_obj, layer=layer)
        p_py = pipeline.pyspark_code
        p_sql = pipeline.sparksql_code

        pending_pipeline = {
            "product_name": table_name,
            "layer": layer,
            "source_entity": entity_name,
            "entity_name": ent_obj.name,
            "flowchart": flowchart_code,
            "pyspark": p_py,
            "sparksql": p_sql,
            "status": "proposed",
        }

        # Save checkpoint of the proposal
        try:
            from src.agent.checkpoint import get_checkpoint_manager
            chk_mgr = get_checkpoint_manager()
            saved_chk = chk_mgr.save_checkpoint(
                thread_id=thread_id,
                entity_name=entity_name,
                pyspark_code=p_py,
                sparksql_code=p_sql,
                ci_status="PROPOSED",
                metadata={"table_name": table_name, "layer": layer, "flowchart": flowchart_code},
            )
            chk_id = saved_chk.get("checkpoint_id", "")
            chk_msg = f"\n💾 **Proposta v1 salva no banco de histórico conversacional (`{chk_id}`)**"
        except ImportError:
            chk_msg = ""

        confirmation_prompt = (
            "\n\n---\n"
            "❓ **Deseja aprovar esta proposta e fazer o deploy no Databricks?**\n"
            "👉 *Digite **'sim'** ou **'confirmar'** para executar o deploy com CI!*"
        )
        response_text = (
            f"### 🎨 Proposta Visual de Pipeline ETL ({layer.upper()} Layer): {table_name}\n\n"
            f"{flowchart_md}\n\n"
            f"### 💻 Código Gerado para Revisão\n"
            f"#### PySpark Pipeline\n```python\n{p_py}\n```\n\n"
            f"#### SparkSQL DDL & Ingestion\n```sql\n{p_sql}\n```\n"
            f"{chk_msg}"
            f"{confirmation_prompt}"
        )

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": flowchart_md,
            "pending_pipeline": pending_pipeline,
        }
