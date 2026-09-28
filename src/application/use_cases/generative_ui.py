import base64
import json
from typing import Any

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.config import settings
from src.semantic.registry import SemanticRegistry

class GenerativeUIUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        # Carregar o schema do Auto-Discovery para injetar no widget
        registry = SemanticRegistry(settings.semantic_models_path)
        schema_list = []
        for entity in registry.entities.values():
            schema_list.append({
                "name": entity.name,
                "columns": [{"name": c.name, "type": c.type} for c in entity.columns]
            })
            
        schema_json_str = json.dumps(schema_list)
        
        # Carregar e injetar o schema no HTML
        with open("src/ux/widgets/interactive_er_builder.html", "r", encoding="utf-8") as f:
            html_content = f.read()
            
        html_content = html_content.replace("__SCHEMA_JSON__", schema_json_str)
        
        # Converter para Base64 para embedar no Open WebUI iframe
        html_bytes = html_content.encode("utf-8")
        html_b64 = base64.b64encode(html_bytes).decode("utf-8")
        
        # Ajustar o CSS base do iframe (remover fundo branco)
        data_uri = f"data:text/html;base64,{html_b64}"
        iframe_md = f'<iframe src="{data_uri}" width="100%" height="700px" style="border:none; border-radius: 8px; background: transparent;" allow="fullscreen" allowfullscreen></iframe>'
        
        response_text = (
            "### 📐 Interactive Semantic ER Builder\n\n"
            "Abra o construtor interativo abaixo. Você pode selecionar colunas específicas, renomeá-las, "
            "aplicar funções analíticas (como Group By/Count) ou criar expressões SQL nativas.\n\n"
            "Quando terminar, clique em **Send to AI** e cole o `[ETL_BUILDER_PAYLOAD]` para que eu implemente "
            f"a sua visão no Databricks.\n\n{iframe_md}"
        )
        
        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text
        }
