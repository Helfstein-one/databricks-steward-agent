from typing import Any
import base64
from langchain_core.messages import AIMessage
from src.agent.state import AgentState

class GenerativeUIUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        with open("src/ux/widgets/etl_builder.html", "rb") as f:
            html_bytes = f.read()
            html_b64 = base64.b64encode(html_bytes).decode("utf-8")
        
        # Open WebUI supports rendering iframes with base64 data URIs
        data_uri = f"data:text/html;base64,{html_b64}"
        iframe_md = f'<iframe src="{data_uri}" width="100%" height="600px" style="border:none; border-radius: 8px; background: white;"></iframe>'
        
        response_text = f"### 🧩 Visual ETL Builder\n\nArraste e solte os componentes abaixo. Quando finalizar, clique em **Save ETL** e cole o payload gerado na nossa conversa para eu montar as pipelines no Databricks.\n\n{iframe_md}"
        
        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text
        }
