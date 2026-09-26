from typing import Any
import json
from langchain_core.messages import AIMessage
from src.agent.state import AgentState
from src.etl.generator import generate_medallion_pipeline
from src.semantic.models import EntityModel
from src.agent.checkpoint import get_checkpoint_manager

class JsonBuilderUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        query = state.get("user_query", "")
        # Extract JSON payload
        start = query.find("{")
        end = query.rfind("}")
        if start != -1 and end != -1:
            try:
                payload = json.loads(query[start:end+1])
                flow = payload.get("flow", [])
                
                source = next((n["value"] for n in flow if n["type"] == "source"), "unknown_source")
                transforms = [n["value"] for n in flow if n["type"] == "transform"]
                
                layer = "gold" if "GROUP BY" in transforms else "silver"
                ent_obj = EntityModel(name=source, table_name=source)
                
                pipeline = generate_medallion_pipeline(ent_obj, layer=layer)
                
                thread_id = state.get("thread_id", "default_thread")
                chk_mgr = get_checkpoint_manager()
                chk_mgr.save_checkpoint(
                    thread_id=thread_id,
                    entity_name=source,
                    pyspark_code=pipeline.pyspark_code,
                    sparksql_code=pipeline.sparksql_code,
                    ci_status="APPROVED",
                    metadata={"table_name": pipeline.table_name, "layer": layer},
                )
                
                response_text = f"### 🚀 ETL PySpark Gerado a partir do Widget\n\nIdentifiquei a origem `{source}` e as transformações `{', '.join(transforms)}`. Aqui está o código resultante:\n\n```python\n{pipeline.pyspark_code}\n```\n\nPara fazer o deploy e commit, digite **confirmar**."
                
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
                        "source_entity": source,
                        "layer": layer
                    }
                }
            except Exception:
                pass
                
        return {"messages": [AIMessage(content="Payload JSON inválido.")], "response": "Payload JSON inválido."}
