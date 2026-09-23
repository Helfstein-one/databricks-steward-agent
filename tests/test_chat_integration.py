"""Integration tests for Chat layer: Intent Router, Open WebUI Pipe, and Conversational Synthesizer."""

from __future__ import annotations

import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from open_webui_pipe import Pipe
from src.agent.graph import _synthesize_conversational_response
from src.application.intent_router import (
    classify_intent_with_llm,
    is_confirmation,
    is_data_preview_query,
)

# ============================================================================
# 1. Tests for Semantic Intent Router (src/application/intent_router.py)
# ============================================================================

def test_intent_router_regex_pre_checks() -> None:
    """Test regex pre-checks for confirmation and data preview before calling LLM."""
    assert is_confirmation("sim, pode rodar o pipeline") is True
    assert is_confirmation("não, cancelar") is False

    assert is_data_preview_query("consultar dados da tabela transactions") is True
    assert is_data_preview_query("como funciona o lakehouse?") is False


def test_intent_router_all_tags_with_mock_llm() -> None:
    """Test classification for all tags: PREVIEW, SCHEMA, DIAGRAM, ETL, CONFIRM, GREETING, OTHER."""
    tags = ["PREVIEW", "SCHEMA", "DIAGRAM", "ETL", "CONFIRM", "GREETING", "OTHER"]

    for tag in tags:
        def _fake_llm(messages: Any, t: str = tag) -> AIMessage:
            return AIMessage(content=f"  {t}\n")

        mock_llm = RunnableLambda(_fake_llm)

        # Query that doesn't trigger pre-check regexes
        query = "qual a arquitetura recomendada para este caso?"
        result = classify_intent_with_llm(query, mock_llm)
        assert result == tag


def test_intent_router_regex_overrides() -> None:
    """Test that regex pre-checks trigger PREVIEW and CONFIRM directly without LLM call."""
    called = False

    def _fake_llm(messages: Any) -> AIMessage:
        nonlocal called
        called = True
        return AIMessage(content="OTHER")

    mock_llm = RunnableLambda(_fake_llm)

    # Regex matches confirmation
    assert classify_intent_with_llm("sim, confirmo", mock_llm) == "CONFIRM"
    assert not called

    # Regex matches preview query
    assert classify_intent_with_llm("mostrar os dados da tabela", mock_llm) == "PREVIEW"
    assert not called


def test_intent_router_fallbacks() -> None:
    """Test fallback behavior on LLM exception, hallucinated tags, or unexpected JSON/formatting."""
    # 1. LLM raises Exception -> Fallback to OTHER
    def _error_llm(messages: Any) -> AIMessage:
        raise RuntimeError("LLM Connection Timeout")

    mock_error_llm = RunnableLambda(_error_llm)
    assert classify_intent_with_llm("qual o esquema do banco?", mock_error_llm) == "OTHER"

    # 2. LLM hallucinates an unknown tag / tag not in target list -> Fallback to OTHER
    def _hallucinated_llm(messages: Any) -> AIMessage:
        return AIMessage(content="UNKNOWN_TAG_HALLUCINATION")

    mock_hallucinated_llm = RunnableLambda(_hallucinated_llm)
    assert classify_intent_with_llm("me fale sobre o tempo", mock_hallucinated_llm) == "OTHER"

    # 3. LLM returns unexpected JSON / structured markdown wrapper with tag embedded
    def _json_llm(messages: Any) -> AIMessage:
        return AIMessage(content='{"classification": "SCHEMA", "confidence": 0.99}')

    mock_json_llm = RunnableLambda(_json_llm)
    assert classify_intent_with_llm("listar tabelas do catalogo", mock_json_llm) == "SCHEMA"

    # 4. Empty or whitespace query -> Fallback to OTHER directly
    assert classify_intent_with_llm("", mock_error_llm) == "OTHER"
    assert classify_intent_with_llm("   ", mock_error_llm) == "OTHER"


# ============================================================================
# 2. Dedicated Tests for open_webui_pipe.py
# ============================================================================

def test_open_webui_pipe_initialization_and_valves() -> None:
    """Test Pipe initialization, default Valves settings, and sync logic on valve changes."""
    pipe = Pipe()
    assert hasattr(pipe, "valves")
    assert pipe.valves.LOCAL_LLM_MODEL is not None
    assert pipe._synced_base_url == pipe.valves.LOCAL_LLM_BASE_URL
    assert pipe._synced_model == pipe.valves.LOCAL_LLM_MODEL

    # Simulate changing administrative valves
    pipe.valves.LOCAL_LLM_MODEL = "llama3:8b"
    pipe.pipe({"messages": [{"role": "user", "content": "1"}]})
    assert pipe._synced_model == "llama3:8b"


def test_open_webui_pipe_message_formats_dict_and_object() -> None:
    """Test handling of messages structured as dicts and objects with content attributes."""
    pipe = Pipe()

    # 1. Dict message format
    body_dict = {"messages": [{"role": "user", "content": "1"}]}
    res_dict = pipe.pipe(body_dict)
    assert isinstance(res_dict, str)
    assert "Unity Catalog" in res_dict or "Databricks" in res_dict

    # 2. Object with content attribute format
    class MessageObj:
        def __init__(self, content: str) -> None:
            self.content = content

    body_obj = {"messages": [MessageObj("3")]}
    res_obj = pipe.pipe(body_obj)
    assert isinstance(res_obj, str)
    assert "erDiagram" in res_obj or "mermaid" in res_obj or "Diagram" in res_obj


def test_open_webui_pipe_streaming_and_non_streaming() -> None:
    """Test non-streaming response string vs streaming generator output."""
    pipe = Pipe()

    # Non-streaming request
    res_non_stream = pipe.pipe({"messages": [{"role": "user", "content": "1"}], "stream": False})
    assert isinstance(res_non_stream, str)

    # Streaming request
    res_stream = pipe.pipe({"messages": [{"role": "user", "content": "1"}], "stream": True})
    assert not isinstance(res_stream, str)
    assert isinstance(res_stream, Generator)

    chunks = list(res_stream)
    reconstructed = "".join(chunks)
    assert reconstructed == res_non_stream


def test_open_webui_pipe_exception_handling() -> None:
    """Test exception handling in Pipe.pipe returning formatted error string."""
    pipe = Pipe()
    with patch.object(pipe.graph, "invoke", side_effect=RuntimeError("Graph Execution Failed")):
        res = pipe.pipe({"messages": [{"role": "user", "content": "test"}]})
        assert isinstance(res, str)
        assert res.startswith("❌ Steward Agent Error:")
        assert "Graph Execution Failed" in res


# ============================================================================
# 3. Tests for Conversational Synthesizer Wrapper (_synthesize_conversational_response)
# ============================================================================

def test_conversational_synthesizer_preserves_markdown_tables_code_and_diagrams() -> None:
    """Test that synthesizer wrapper preserves Markdown tables, PySpark/SQL code, and Mermaid diagrams."""
    raw_markdown = (
        "### 📊 Modelo de Entidade-Relacionamento\n\n"
        "| Tabela Origem | Chave Origem (FK) | Relacionamento | Tabela Destino | Chave Destino (PK) |\n"
        "|---|---|---|---|---|\n"
        "| transactions | customer_id | N:1 | customers | customer_id |\n\n"
        "```python\ndef process(df):\n    return df.filter(df['status'] == 'COMPLETED')\n```\n\n"
        "```sql\nSELECT * FROM medallion_silver_transactions;\n```\n\n"
        "```mermaid\nerDiagram\n    CUSTOMERS ||--o{ TRANSACTIONS : places\n```"
    )

    det_result = {
        "messages": [AIMessage(content=raw_markdown)],
        "response": raw_markdown,
    }

    # Mock client returning synthesized response wrapping the raw markdown
    synthesized_content = (
        "Olá! Aqui está o resumo do seu modelo:\n\n"
        f"{raw_markdown}\n\n"
        "Posso ajudar a gerar os pipelines referentes a este modelo?"
    )

    mock_client = MagicMock()
    mock_client.invoke.return_value = AIMessage(content=synthesized_content)

    clean_modules = {k: v for k, v in sys.modules.items() if k != "pytest"}
    with patch.dict("sys.modules", clean_modules, clear=True), patch.dict("os.environ", {"PYTEST_CURRENT_TEST": ""}):
        res = _synthesize_conversational_response(
            det_result, mock_client, "desenhar modelo semântico"
        )

    # Assert Markdown components are intact in output
    out_resp = res["response"]
    assert "| Tabela Origem |" in out_resp
    assert "```python" in out_resp
    assert "```sql" in out_resp
    assert "```mermaid" in out_resp
    assert "CUSTOMERS ||--o{ TRANSACTIONS" in out_resp
    assert "Olá! Aqui está o resumo" in out_resp


def test_conversational_synthesizer_fallback_on_corrupted_output() -> None:
    """Test fallback to deterministic output when LLM strips code blocks or fails."""
    raw_markdown = (
        "### Pipeline Code\n\n"
        "```python\ndef process(df):\n    return df\n```"
    )

    det_result = {
        "messages": [AIMessage(content=raw_markdown)],
        "response": raw_markdown,
    }

    # Case 1: LLM hallucinated and removed the ```python block
    corrupted_content = "Aqui está o seu pipeline sem o código."
    mock_client_corrupt = MagicMock()
    mock_client_corrupt.invoke.return_value = AIMessage(content=corrupted_content)

    clean_modules = {k: v for k, v in sys.modules.items() if k != "pytest"}
    with patch.dict("sys.modules", clean_modules, clear=True), patch.dict("os.environ", {"PYTEST_CURRENT_TEST": ""}):
        res1 = _synthesize_conversational_response(
            det_result.copy(), mock_client_corrupt, "gerar etl"
        )
    assert res1["response"] == raw_markdown  # Fallback triggered!

    # Case 2: LLM raises exception
    mock_client_error = MagicMock()
    mock_client_error.invoke.side_effect = RuntimeError("Synthesis Timeout")

    with patch.dict("sys.modules", clean_modules, clear=True), patch.dict("os.environ", {"PYTEST_CURRENT_TEST": ""}):
        res2 = _synthesize_conversational_response(
            det_result.copy(), mock_client_error, "gerar etl"
        )
    assert res2["response"] == raw_markdown  # Fallback triggered!
