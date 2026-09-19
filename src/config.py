"""Configuration management for Databricks Steward Agent."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Carrega variáveis de .env se existir
load_dotenv()


class Settings(BaseModel):
    # Databricks
    databricks_host: str = Field(default_factory=lambda: os.getenv("DATABRICKS_HOST", ""))
    databricks_token: str = Field(default_factory=lambda: os.getenv("DATABRICKS_TOKEN", ""))
    databricks_warehouse_id: str = Field(
        default_factory=lambda: os.getenv("DATABRICKS_WAREHOUSE_ID", "")
    )
    databricks_default_catalog: str = Field(
        default_factory=lambda: os.getenv("DATABRICKS_DEFAULT_CATALOG", "main")
    )
    databricks_default_schema: str = Field(
        default_factory=lambda: os.getenv("DATABRICKS_DEFAULT_SCHEMA", "default")
    )

    # Local LLM
    local_llm_base_url: str = Field(
        default_factory=lambda: os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    )
    local_llm_model: str = Field(
        default_factory=lambda: os.getenv("LOCAL_LLM_MODEL", "qwen2.5-coder:7b")
    )
    local_llm_api_key: str = Field(default_factory=lambda: os.getenv("LOCAL_LLM_API_KEY", "ollama"))
    local_llm_temperature: float = Field(
        default_factory=lambda: float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1"))
    )

    # GitHub GitOps
    github_token: str = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    github_repository: str = Field(default_factory=lambda: os.getenv("GITHUB_REPOSITORY", ""))
    github_base_branch: str = Field(default_factory=lambda: os.getenv("GITHUB_BASE_BRANCH", "main"))

    # Paths and defaults
    semantic_models_path: Path = Field(
        default_factory=lambda: Path(os.getenv("SEMANTIC_MODELS_PATH", "./configs/semantic_models"))
    )
    default_query_limit: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_QUERY_LIMIT", "50"))
    )
    max_query_limit: int = Field(default_factory=lambda: int(os.getenv("MAX_QUERY_LIMIT", "200")))


settings = Settings()
