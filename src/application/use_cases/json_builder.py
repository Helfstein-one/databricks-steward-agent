import json
from typing import Any

from langchain_core.messages import AIMessage

from src.agent.checkpoint import get_checkpoint_manager
from src.agent.state import AgentState
from src.etl.generator import generate_medallion_pipeline
from src.semantic.models import EntityModel


class JsonBuilderUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        query = state.get("user_query", "")
        start = query.find("{")
        end = query.rfind("}")
        if start != -1 and end != -1:
            try:
                payload = json.loads(query[start : end + 1])
                
                # Suporte ao formato antigo (Visual Builder v1)
                if "flow" in payload:
                    flow = payload.get("flow", [])
                    source = next((n["value"] for n in flow if n["type"] == "source"), "unknown")
                    transforms = [n["value"] for n in flow if n["type"] == "transform"]
                    layer = "gold" if "GROUP BY" in transforms else "silver"
                    ent_obj = EntityModel(name=source, table_name=source)
                    pipeline = generate_medallion_pipeline(ent_obj, layer=layer)
                    
                    response_text = f"### 🚀 ETL PySpark Gerado a partir do Widget\n\nIdentifiquei a origem `{source}` e as transformações `{', '.join(transforms)}`. Aqui está o código resultante:\n\n```python\n{pipeline.pyspark_code}\n```\n\nPara fazer o deploy e commit, digite **confirmar**."
                    
                # Novo formato (Interactive ER Builder v2)
                elif "sources" in payload:
                    sources = payload.get("sources", [])
                    if not sources:
                        raise ValueError("No sources selected")
                        
                    primary_source = sources[0]["table"]
                    cols = sources[0]["columns"]
                    
                    layer = "silver"
                    has_agg = any(c.get("agg") for c in cols)
                    if has_agg:
                        layer = "gold"
                        
                    ent_obj = EntityModel(name=primary_source, table_name=primary_source)
                    pipeline = generate_medallion_pipeline(ent_obj, layer=layer)
                    
                    # Generate a custom description based on their visual selection
                    desc = f"Identifiquei a seleção de `{primary_source}` com {len(cols)} colunas configuradas na Interface Visual.\n"
                    for c in cols:
                        ops = []
                        if c.get("alias"): ops.append(f"Alias: {c['alias']}")
                        if c.get("cast"): ops.append(f"Cast: {c['cast']}")
                        if c.get("agg"): ops.append(f"Agg: {c['agg']}")
                        if c.get("mask"): ops.append(f"Mask: {c['mask']}")
                        if c.get("expr"): ops.append(f"SQL: {c['expr']}")
                        desc += f"- **{c['name']}**: " + (", ".join(ops) if ops else "Manter") + "\n"
                        
                    response_text = f"### 🚀 Arquitetura Interativa Compreendida\n\n{desc}\n\n**Código PySpark Base Gerado:**\n```python\n{pipeline.pyspark_code}\n```\n\nPara aplicar essas transformações customizadas e aprovar o deploy GitOps, digite **confirmar**."

                thread_id = state.get("thread_id", "default_thread")
                chk_mgr = get_checkpoint_manager()
                chk_mgr.save_checkpoint(
                    thread_id=thread_id,
                    entity_name=ent_obj.name,
                    pyspark_code=pipeline.pyspark_code,
                    sparksql_code=pipeline.sparksql_code,
                    ci_status="APPROVED",
                    metadata={"table_name": pipeline.table_name, "layer": layer},
                )

                return {
                    "messages": [AIMessage(content=response_text)],
                    "response": response_text,
                    "generated_code": {
                        "pyspark": pipeline.pyspark_code,
                        "sparksql": pipeline.sparksql_code,
                        "table_name": pipeline.table_name,
                        "layer": layer,
                    },
                    "pending_pipeline": {
                        "product_name": pipeline.table_name,
                        "pyspark": pipeline.pyspark_code,
                        "sparksql": pipeline.sparksql_code,
                        "source_entity": ent_obj.name,
                        "layer": layer,
                    },
                }
            except Exception as e:
                return {"messages": [AIMessage(content=f"Payload JSON inválido: {e}")], "response": f"Payload JSON inválido: {e}"}

        return {"messages": [AIMessage(content="Payload JSON inválido.")], "response": "Payload JSON inválido."}
