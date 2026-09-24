
import re
from typing import Any
from langchain_core.messages import AIMessage
from src.agent.state import AgentState
from src.semantic.registry import SemanticRegistry
from src.config import settings
from src.etl.generator import generate_medallion_pipeline
from src.agent.tools import generate_etl_pipeline
from src.agent.intent import _extract_entity_from_query, _resolve_anaphoric_entity, _extract_query_text

class ETLGenerationUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_query = state.get("user_query") or _extract_query_text(messages)
        q_lower = (user_query or "").strip().lower()

        active_diagram = state.get("active_diagram")
        generated_code = state.get("generated_code")
        ci_report = state.get("ci_report")
        gitops_result = state.get("gitops_result")
        pending_pipeline = state.get("pending_pipeline")

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

        if ent_obj:
            pipeline = generate_medallion_pipeline(ent_obj, layer=layer)
            generated_code = {
                "pyspark": pipeline.pyspark_code,
                "sparksql": pipeline.sparksql_code,
                "table_name": pipeline.table_name,
                "layer": pipeline.layer,
            }
            pending_pipeline = {
                "product_name": pipeline.table_name,
                "pyspark": pipeline.pyspark_code,
                "sparksql": pipeline.sparksql_code,
                "source_entity": entity_name,
            }
            confirmation_prompt = (
                "\n\n---\n"
                "❓ **Deseja confirmar e disparar a esteira de CI, auto commit & push na branch `main` e criação do Job no Databricks?**\n"
                "👉 *Digite **'sim'** ou **'confirmar'** para executar o ciclo de vida completo!*"
            )
            response_text = (
                f"### Generated Medallion Pipeline: {pipeline.table_name} ({pipeline.layer})\n\n"
                f"#### PySpark Pipeline\n```python\n{pipeline.pyspark_code}\n```\n\n"
                f"#### SparkSQL DDL & Ingestion\n```sql\n{pipeline.sparksql_code}\n```{confirmation_prompt}"
            )
        else:
            response_text = generate_etl_pipeline(entity_name, layer=layer)
            confirmation_prompt = (
                "\n\n---\n"
                "❓ **Deseja confirmar e disparar a esteira de CI, auto commit & push na branch `main` e criação do Job no Databricks?**\n"
                "👉 *Digite **'sim'** ou **'confirmar'** para executar o ciclo de vida completo!*"
            )
            response_text += confirmation_prompt

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "pending_pipeline": pending_pipeline,
        }
