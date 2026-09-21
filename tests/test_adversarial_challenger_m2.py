"""Empirical Challenger M2 Adversarial Test Suite.

Stress-tests:
1. Negation rejection guard in _is_confirmation
2. Anaphoric keyword resolution in _resolve_anaphoric_entity (with and without keywords)
3. End-to-end journey resilience, parsing, and failure mode behavior
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from open_webui_pipe import Pipe
from src.agent.graph import (
    _is_confirmation,
    _resolve_anaphoric_entity,
    steward_node,
)
from src.agent.state import AgentState

# ==============================================================================
# 1. ADVERSARIAL CHALLENGE: Negation Rejection Guard (_is_confirmation)
# ==============================================================================


@pytest.mark.parametrize(
    "neg_query",
    [
        "não pode executar",
        "não confirmar",
        "cancelar",
        "cancela",
        "não quero executar",
        "não aprovo",
        "não prosseguir",
        "jamais executar isso",
        "nunca autorizei isso",
        "nem pensar em confirmar",
        "cancelar imediatamente",
        "cancela tudo",
        "não autorizo",
        "nao pode rodar",
        "nao confirmar",
        "não, cancelar",
        "nem execute",
        "NÃO CONFIRMAR",
        "CANCELAR",
        "Não pode executar!",
    ],
)
def test_adversarial_negation_rejection(neg_query: str) -> None:
    """Empirically verify that negative expressions containing confirmation keywords return False."""
    assert _is_confirmation(neg_query) is False, (
        f"Negation rejection failed for query: '{neg_query}'"
    )


@pytest.mark.parametrize(
    "empty_or_neutral",
    [
        "",
        "   ",
        "olá",
        "qual a modelagem de customers?",
        "desenhar diagrama",
        "listar tabelas",
        "ajuda",
        "1",
        "quem é você?",
    ],
)
def test_neutral_and_empty_queries_not_confirmed(empty_or_neutral: str) -> None:
    """Verify neutral, query, and empty inputs return False for _is_confirmation."""
    assert _is_confirmation(empty_or_neutral) is False


@pytest.mark.parametrize(
    "pos_query",
    [
        "sim",
        "confirmar",
        "confirmo",
        "aprovar",
        "aprovo",
        "pode executar",
        "executa",
        "executar",
        "confirmado",
        "prosseguir",
        "ok",
        "yes",
        "positivo",
        "autorizado",
        "pode rodar",
        "pode seguir",
        "manda bala",
        "pode fazer",
        "vamos lá",
        "bora",
        "s",
        "y",
        "SIM, PODE PROSSEGUIR",
        "sim, pode prosseguir",
    ],
)
def test_valid_affirmative_confirmations(pos_query: str) -> None:
    """Verify valid positive affirmations return True."""
    assert _is_confirmation(pos_query) is True


def test_adversarial_negation_prevents_deployment_in_graph() -> None:
    """Adversarial test: Ensure that sending a negative query when pending_pipeline exists does NOT deploy."""
    pending_mock = {
        "product_name": "test_adversarial_gold_product",
        "pyspark": "print('exploit')",
        "sparksql": "SELECT 1;",
        "source_entity": "customers",
    }

    state: AgentState = {
        "messages": [
            HumanMessage(content="propor um etl gold a partir dessa tabela"),
            AIMessage(content="Deseja confirmar e disparar a esteira de CI?"),
        ],
        "user_query": "não pode executar, cancelar",
        "pending_pipeline": pending_mock,
    }

    # If deploy_and_materialize_data_product were called, mock will catch it
    with patch("src.agent.tools.deploy_and_materialize_data_product") as mock_deploy:
        output = steward_node(state)

        # deploy must NOT have been called!
        assert mock_deploy.call_count == 0, (
            "Security Guard Failed: deploy called on negative query!"
        )

        # state must not report deployment success
        assert output.get("job_result") is None
        assert "Ciclo de Vida do Data Product Concluído com Sucesso" not in output["response"]


# ==============================================================================
# 2. ADVERSARIAL CHALLENGE: Anaphoric Entity Resolution (_resolve_anaphoric_entity)
# ==============================================================================


@pytest.mark.parametrize(
    "anaphoric_query",
    [
        "propor um etl gold a partir dessa tabela",
        "criar pipeline desta tabela",
        "etl a partir desse modelo",
        "usar dados deste modelo",
        "propor etl da tabela acima",
        "fazer gold da tabela anterior",
        "propor pipeline para a mesma tabela",
        "gerar etl dos dados dela",
        "usar informações dele",
        "PROPOR ETL DESSA TABELA",
    ],
)
def test_anaphoric_resolution_with_keywords_from_state(anaphoric_query: str) -> None:
    """Verify anaphoric queries resolve to previewed table from state."""
    state: AgentState = {
        "messages": [],
        "user_query": anaphoric_query,
        "preview_data": {"table_name": "workspace.default.customers", "limit": 10},
    }
    resolved = _resolve_anaphoric_entity(anaphoric_query, [], state)
    assert resolved == "customers", f"Failed to resolve 'customers' from query: '{anaphoric_query}'"


@pytest.mark.parametrize(
    "non_anaphoric_query",
    [
        "propor um etl gold",
        "gerar pipeline gold",
        "criar etl",
        "quero um etl",
        "fazer ingestao",
        "executar pipeline",
        "qual a modelagem?",
        "olá",
        "ajuda",
        "",
    ],
)
def test_anaphoric_resolution_without_keywords_returns_none(non_anaphoric_query: str) -> None:
    """Adversarial test: Queries WITHOUT anaphoric keywords MUST return None, even if preview_data exists."""
    state: AgentState = {
        "messages": [
            AIMessage(content="### 📊 Amostra de Dados da Tabela: `workspace.default.customers`")
        ],
        "user_query": non_anaphoric_query,
        "preview_data": {"table_name": "workspace.default.customers", "limit": 10},
    }
    resolved = _resolve_anaphoric_entity(non_anaphoric_query, state["messages"], state)
    assert resolved is None, (
        f"False positive anaphoric resolution for query without anaphoric keyword: '{non_anaphoric_query}'"
    )


def test_anaphoric_resolution_from_message_history_fallback() -> None:
    """Verify resolution works via message history when state['preview_data'] is missing."""
    messages = [
        HumanMessage(content="consulte os dados da tabela orders"),
        AIMessage(
            content="### 📊 Amostra de Dados da Tabela: `workspace.default.orders` (Top 10 registros)\n| id | date |"
        ),
    ]
    query = "propor um etl gold a partir dessa tabela"
    resolved = _resolve_anaphoric_entity(query, messages, state=None)
    assert resolved == "orders", (
        f"Failed to resolve 'orders' from message history, got '{resolved}'"
    )


def test_anaphoric_resolution_recency_in_multi_turn_history() -> None:
    """Adversarial test: In a multi-turn conversation with multiple previews, resolve to the MOST RECENT entity."""
    messages = [
        HumanMessage(content="consulte os dados da tabela products"),
        AIMessage(
            content="### 📊 Amostra de Dados da Tabela: `workspace.default.products`\n| id | name |"
        ),
        HumanMessage(content="agora consulte os dados da tabela customers"),
        AIMessage(
            content="### 📊 Amostra de Dados da Tabela: `workspace.default.customers`\n| customer_id | name |"
        ),
    ]
    query = "propor um etl gold a partir dessa tabela"
    resolved = _resolve_anaphoric_entity(query, messages, state=None)
    assert resolved == "customers", (
        f"Recency failure: expected 'customers' (most recent), but got '{resolved}'"
    )


def test_graph_routing_with_vs_without_anaphoric_reference() -> None:
    """Adversarial test: Compare full graph execution with vs without anaphora.

    - Turn 2 WITH 'dessa tabela' -> resolves to customers -> generates 'medallion_gold_customer_kpis'
    - Turn 2 WITHOUT 'dessa tabela' ('propor um etl gold') -> falls back to default 'medallion_gold_sales_kpis'
    """
    messages_after_turn1 = [
        {"role": "user", "content": "consulte os dados da tabela customers"},
        {
            "role": "assistant",
            "content": "### 📊 Amostra de Dados da Tabela: `workspace.default.customers` (Top 10 registros)\n| customer_id |",
        },
    ]

    # Scenario A: WITH anaphora
    state_with_anaphora: AgentState = {
        "messages": messages_after_turn1,
        "user_query": "propor um etl gold a partir dessa tabela",
        "preview_data": {"table_name": "workspace.default.customers", "limit": 10},
    }
    out_with = steward_node(state_with_anaphora)
    pending_with = out_with.get("pending_pipeline")
    assert pending_with is not None
    assert "medallion_gold_customer_kpis" in pending_with["product_name"]
    assert pending_with["source_entity"] == "medallion_gold_customer_kpis"
    assert "medallion_gold_customer_kpis" in out_with["response"]

    # Scenario B: WITHOUT anaphora
    state_without_anaphora: AgentState = {
        "messages": messages_after_turn1,
        "user_query": "propor um etl gold",
        "preview_data": {"table_name": "workspace.default.customers", "limit": 10},
    }
    out_without = steward_node(state_without_anaphora)
    pending_without = out_without.get("pending_pipeline")
    assert pending_without is not None
    assert "medallion_gold_sales_kpis" in pending_without["product_name"]
    assert pending_without["source_entity"] == "medallion_gold_sales_kpis"
    assert "medallion_gold_sales_kpis" in out_without["response"]


# ==============================================================================
# 3. ADVERSARIAL CHALLENGE: test_e2e_journey.py Resilience & Structure
# ==============================================================================


def test_e2e_journey_parser_resilience() -> None:
    """Verify Job ID and Run ID regex extraction handles various formatting permutations."""
    test_outputs = [
        "- **Job ID:** `776130343161247` (Run: `190789938461936`)",
        "- **Job ID:** 776130343161247 (Run: 190789938461936)",
        "Databricks Job ID: `123456`",
        "Job ID: 987654 (Run: `456789`)",
    ]

    for output in test_outputs:
        job_match = re.search(r"Job ID[:\*]*\s*`?(\d+)`?", output)
        assert job_match is not None, f"Failed to extract Job ID from: '{output}'"
        assert int(job_match.group(1)) > 0

        run_match = re.search(r"Run[:\*]*\s*`?(\d+)`?", output)
        if "Run" in output:
            assert run_match is not None, f"Failed to extract Run ID from: '{output}'"
            assert int(run_match.group(1)) > 0


def test_e2e_journey_missing_credentials_graceful_exit() -> None:
    """Verify run_e2e_journey returns False gracefully when required environment tokens are missing."""
    import test_e2e_journey

    with patch("test_e2e_journey.settings.databricks_host", None):
        assert test_e2e_journey.run_e2e_journey() is False

    with patch("test_e2e_journey.settings.databricks_token", None):
        assert test_e2e_journey.run_e2e_journey() is False


def test_open_webui_pipe_multi_turn_state_preservation() -> None:
    """Verify Open WebUI Pipe retains conversation state across turns."""
    pipe = Pipe()

    messages: list[dict[str, str]] = []

    # Turn 1
    messages.append({"role": "user", "content": "consulte os dados da tabela customers"})
    resp1 = pipe.pipe({"messages": messages, "stream": False})
    assert isinstance(resp1, str)
    assert len(resp1) > 0
    messages.append({"role": "assistant", "content": resp1})

    # Turn 2
    messages.append({"role": "user", "content": "propor um etl gold a partir dessa tabela"})
    resp2 = pipe.pipe({"messages": messages, "stream": False})
    assert isinstance(resp2, str)
    assert "medallion_gold_customer_kpis" in resp2
    assert "```python" in resp2
    assert "```sql" in resp2
    messages.append({"role": "assistant", "content": resp2})

    # Turn 2.5: User rejects / cancels
    messages.append({"role": "user", "content": "não pode executar, cancelar"})
    resp_reject = pipe.pipe({"messages": messages, "stream": False})
    # Must NOT execute deployment
    assert "Ciclo de Vida do Data Product Concluído com Sucesso" not in resp_reject
    assert "Job ID:" not in resp_reject


def test_adversarial_anaphoric_malformed_messages_robustness() -> None:
    """Adversarial test: Ensure _resolve_anaphoric_entity does not crash on malformed message objects."""
    malformed_messages = [
        None,
        12345,
        {},
        {"role": "assistant"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": 42},
        {"content": ""},
        MagicMock(content=None),
        MagicMock(spec=[]),  # Object with no content attribute
    ]
    # Query has anaphoric keyword but history contains malformed entries
    result = _resolve_anaphoric_entity("propor etl dessa tabela", malformed_messages, state=None)
    assert result is None


@pytest.mark.parametrize(
    "obfuscated_negation",
    [
        "   não!  ",
        "NÃO...",
        "(não pode executar)",
        "cancelar!",
        "nem pensar!!",
        "não, obrigado",
        "jamais!",
        "nunca!",
    ],
)
def test_adversarial_obfuscated_negations(obfuscated_negation: str) -> None:
    """Verify punctuation and casing surrounding negation keywords are safely recognized."""
    assert _is_confirmation(obfuscated_negation) is False


def test_e2e_journey_step_failure_handling(mock_env) -> None:
    """Verify that simulated Step failures in run_e2e_journey raise AssertionError as expected."""
    import test_e2e_journey

    # Mock Pipe.pipe to return an empty response in Step 1
    mock_pipe = MagicMock()
    mock_pipe.pipe.return_value = ""

    with (
        patch("test_e2e_journey.settings.databricks_host", "https://test.databricks.com"),
        patch("test_e2e_journey.settings.databricks_token", "test_token"),
        patch("test_e2e_journey.Pipe", return_value=mock_pipe),
        pytest.raises(AssertionError, match="Step 1: Empty response received"),
    ):
        test_e2e_journey.run_e2e_journey()
