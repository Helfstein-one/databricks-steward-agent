"""Unit tests for configuration loading and environment overrides (Tier 1)."""

from pathlib import Path

from src.config import Settings, settings


def test_default_settings():
    """Verify default values of Settings."""
    cfg = Settings()
    assert cfg.databricks_default_catalog == "workspace"
    assert cfg.databricks_default_schema == "default"
    assert "localhost" in cfg.local_llm_base_url or "127.0.0.1" in cfg.local_llm_base_url
    assert cfg.local_llm_model == "qwen2.5-coder:7b"
    assert cfg.local_llm_temperature == 0.1
    assert cfg.github_base_branch == "workspace"
    assert cfg.default_query_limit == 50
    assert cfg.max_query_limit == 200
    assert isinstance(cfg.semantic_models_path, Path)


def test_settings_env_override(monkeypatch):
    """Verify that environment variables properly override defaults."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://custom.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-custom-token")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "custom-warehouse-id")
    monkeypatch.setenv("DATABRICKS_DEFAULT_CATALOG", "analytics_catalog")
    monkeypatch.setenv("DATABRICKS_DEFAULT_SCHEMA", "gold_schema")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://ollama-service:11434/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "deepseek-coder:6.7b")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "custom-api-key")
    monkeypatch.setenv("LOCAL_LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_supersecrettoken")
    monkeypatch.setenv("GITHUB_REPOSITORY", "my-org/my-lakehouse")
    monkeypatch.setenv("GITHUB_BASE_BRANCH", "dev")
    monkeypatch.setenv("SEMANTIC_MODELS_PATH", "/custom/semantic/path")
    monkeypatch.setenv("DEFAULT_QUERY_LIMIT", "100")
    monkeypatch.setenv("MAX_QUERY_LIMIT", "500")

    cfg = Settings()
    assert cfg.databricks_host == "https://custom.cloud.databricks.com"
    assert cfg.databricks_token == "dapi-custom-token"
    assert cfg.databricks_warehouse_id == "custom-warehouse-id"
    assert cfg.databricks_default_catalog == "analytics_catalog"
    assert cfg.databricks_default_schema == "gold_schema"
    assert cfg.local_llm_base_url == "http://ollama-service:11434/v1"
    assert cfg.local_llm_model == "deepseek-coder:6.7b"
    assert cfg.local_llm_api_key == "custom-api-key"
    assert cfg.local_llm_temperature == 0.2
    assert cfg.github_token == "ghp_supersecrettoken"
    assert cfg.github_repository == "my-org/my-lakehouse"
    assert cfg.github_base_branch == "dev"
    assert str(cfg.semantic_models_path) == "/custom/semantic/path"
    assert cfg.default_query_limit == 100
    assert cfg.max_query_limit == 500


def test_global_settings_instance():
    """Verify that global singleton instance 'settings' is available and valid."""
    assert isinstance(settings, Settings)
    assert settings.default_query_limit > 0
    assert settings.max_query_limit >= settings.default_query_limit
