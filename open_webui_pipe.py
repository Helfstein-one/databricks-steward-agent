"""Open WebUI Pipe interface for Databricks Steward Agent."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field

from src.agent.graph import create_steward_graph, get_local_chat_client
from src.config import settings


class Pipe:
    """Open WebUI Pipe connecting local models, Databricks, and LangGraph steward workflow."""

    class Valves(BaseModel):
        """Configurable valves exposed in the Open WebUI administrative panel."""

        DATABRICKS_HOST: str = Field(
            default_factory=lambda: settings.databricks_host,
            description="Databricks workspace URL (e.g. https://<workspace-id>.cloud.databricks.com)",
        )
        DATABRICKS_TOKEN: str = Field(
            default_factory=lambda: settings.databricks_token,
            description="Databricks Personal Access Token (PAT)",
        )
        DATABRICKS_WAREHOUSE_ID: str = Field(
            default_factory=lambda: settings.databricks_warehouse_id,
            description="Databricks SQL Warehouse HTTP Path / ID",
        )
        DATABRICKS_DEFAULT_CATALOG: str = Field(
            default_factory=lambda: settings.databricks_default_catalog,
            description="Default Unity Catalog name",
        )
        DATABRICKS_DEFAULT_SCHEMA: str = Field(
            default_factory=lambda: settings.databricks_default_schema,
            description="Default schema / database name",
        )
        LOCAL_LLM_BASE_URL: str = Field(
            default_factory=lambda: settings.local_llm_base_url,
            description="Local LLM OpenAI-compatible endpoint (e.g. Ollama or vLLM)",
        )
        LOCAL_LLM_MODEL: str = Field(
            default_factory=lambda: settings.local_llm_model,
            description="Local model identifier (e.g. qwen2.5-coder:7b)",
        )
        LOCAL_LLM_API_KEY: str = Field(
            default_factory=lambda: settings.local_llm_api_key,
            description="API key for local model serving (default: ollama)",
        )
        LOCAL_LLM_TEMPERATURE: float = Field(
            default_factory=lambda: settings.local_llm_temperature,
            description="Sampling temperature",
        )
        GITHUB_TOKEN: str = Field(
            default_factory=lambda: settings.github_token,
            description="GitHub PAT for GitOps Pull Request automation",
        )
        GITHUB_REPOSITORY: str = Field(
            default_factory=lambda: settings.github_repository,
            description="GitHub repository in owner/repo format",
        )
        GITHUB_BASE_BRANCH: str = Field(
            default_factory=lambda: settings.github_base_branch,
            description="Target base branch for Pull Requests",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.llm = get_local_chat_client(
            base_url=self.valves.LOCAL_LLM_BASE_URL,
            model=self.valves.LOCAL_LLM_MODEL,
            api_key=self.valves.LOCAL_LLM_API_KEY,
            temperature=self.valves.LOCAL_LLM_TEMPERATURE,
        )
        self.graph = create_steward_graph(llm=self.llm)

    def _sync_llm_with_valves(self) -> None:
        """Sync ChatOpenAI client with active administrative valves if overridden."""
        current_base = str(getattr(self.llm, "openai_api_base", ""))
        current_model = str(getattr(self.llm, "model_name", ""))
        if current_base != self.valves.LOCAL_LLM_BASE_URL or current_model != self.valves.LOCAL_LLM_MODEL:
            self.llm = get_local_chat_client(
                base_url=self.valves.LOCAL_LLM_BASE_URL,
                model=self.valves.LOCAL_LLM_MODEL,
                api_key=self.valves.LOCAL_LLM_API_KEY,
                temperature=self.valves.LOCAL_LLM_TEMPERATURE,
            )
            self.graph = create_steward_graph(llm=self.llm)

    def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
    ) -> str | Generator[str, None, None]:
        """Process incoming chat request and return steward response."""
        self._sync_llm_with_valves()
        messages = body.get("messages", [])
        prompt = ""
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                prompt = last_message.get("content", "")
            elif hasattr(last_message, "content"):
                prompt = str(last_message.content)

        state_input = {
            "messages": messages,
            "user_query": prompt,
        }

        try:
            result = self.graph.invoke(state_input)
            response_text = result.get("response", "")

            # Support streaming if requested
            if body.get("stream", False):
                def _stream_gen() -> Generator[str, None, None]:
                    chunk_size = 64
                    for i in range(0, len(response_text), chunk_size):
                        yield response_text[i : i + chunk_size]

                return _stream_gen()

            return response_text
        except Exception as e:  # noqa: BLE001
            return f"❌ Steward Agent Error: {e}"
