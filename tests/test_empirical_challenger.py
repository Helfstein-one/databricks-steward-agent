"""Empirical stress test suite by Challenger M1_1.

Tests LangGraph multi-turn flow, anaphoric reference resolution,
confirmation regexes, edge cases, error resilience, and Pipe integration.
"""

from __future__ import annotations

from typing import Any

import pytest

from open_webui_pipe import Pipe
from src.agent.graph import (
    _extract_entity_from_query,
    _extract_table_or_entity,
    _is_confirmation,
    _is_data_preview_query,
    _resolve_anaphoric_entity,
    create_steward_graph,
)
from src.gitops.git_client import GitClient

# ==============================================================================
# 1. Turn 1: Preview Query Variations
# ==============================================================================

@pytest.mark.parametrize(
    "query,expected_entity",
    [
        ("consulte os dados da tabela customers", "customers"),
        ("consultar os dados da tabela customers", "customers"),
        ("mostrar dados customers", "customers"),
        ("mostre os dados da tabela customers", "customers"),
        ("ver dados da tabela customers", "customers"),
        ("exibir registros da tabela customers", "customers"),
        ("amostra da tabela customers", "customers"),
        ("preview da tabela customers", "customers"),
        ("amostra de dados da tabela customers", "customers"),
        ("quero ver os dados de customers", "customers"),
        ("consulte os dados da tabela workspace.default.customers", "workspace.default.customers"),
        ("mostrar dados da tabela workspace.default.customers LIMIT 5", "workspace.default.customers"),
        ("select * from customers", "customers"),
        ("SELECT * FROM workspace.default.customers LIMIT 10", "workspace.default.customers"),
    ],
)
def test_turn1_preview_variations(query: str, expected_entity: str) -> None:
    assert _is_data_preview_query(query), f"Failed to detect preview query: {query}"
    extracted = _extract_table_or_entity(query)
    assert extracted == expected_entity, f"Extracted '{extracted}' != expected '{expected_entity}' for '{query}'"


# ==============================================================================
# 2. Turn 2: Gold Anaphora Resolution Variations (R1 Journey)
# ==============================================================================

TURN1_ASSISTANT_MD = (
    "### 📊 Amostra de Dados da Tabela: `workspace.default.customers` (Top 10 registros)\n\n"
    "| customer_id | name | segment |\n"
    "|---|---|---|\n"
    "| 1 | Corp A | Enterprise |\n\n"
    "💡 *Dica: Para gerar um pipeline ETL a partir desta tabela, peça: 'propor etl a partir de customers'.*"
)

@pytest.mark.parametrize(
    "turn2_query,expected_resolved",
    [
        ("propor um etl gold a partir dessa tabela", "customers"),
        ("gerar etl gold desta tabela", "customers"),
        ("gerar pipeline etl gold a partir dessa tabela", "customers"),
        ("propor etl gold da tabela acima", "customers"),
        ("propor etl gold dessa tabela", "customers"),
        ("quero um etl gold desta tabela", "customers"),
        ("crie um pipeline gold dessa tabela", "customers"),
        ("propor pipeline gold desta tabela", "customers"),
    ],
)
def test_turn2_gold_anaphora_resolution(turn2_query: str, expected_resolved: str) -> None:
    messages = [
        {"role": "user", "content": "consulte os dados da tabela customers"},
        {"role": "assistant", "content": TURN1_ASSISTANT_MD},
        {"role": "user", "content": turn2_query},
    ]

    # Test resolving from message history alone (as in stateless Open WebUI)
    resolved = _resolve_anaphoric_entity(turn2_query, messages, state={})
    assert resolved == expected_resolved, f"Failed resolving anaphora for '{turn2_query}': got '{resolved}'"

    # Test full graph execution
    graph = create_steward_graph()
    res = graph.invoke({"messages": messages})

    resp = res.get("response", "")
    assert "```python" in resp and ("def " in resp or "process" in resp), (
        f"Graph did not generate PySpark code for '{turn2_query}'. Response: {resp[:200]}"
    )
    assert "```sql" in resp and "CREATE " in resp, (
        f"Graph did not generate SparkSQL DDL for '{turn2_query}'. Response: {resp[:200]}"
    )
    assert res.get("pending_pipeline") is not None, f"pending_pipeline is None for '{turn2_query}'"


# ==============================================================================
# 3. Turn 3: Affirmative Confirmation Variations
# ==============================================================================

@pytest.mark.parametrize(
    "confirm_query",
    [
        "sim, pode prosseguir",
        "confirmar",
        "pode rodar",
        "sim",
        "bora",
        "pode executar",
        "ok",
        "yes",
        "positivo",
        "autorizado",
        "pode seguir",
        "manda bala",
        "pode fazer",
        "vamos lá",
        "s",
        "y",
    ],
)
def test_turn3_affirmative_confirmations(confirm_query: str) -> None:
    assert _is_confirmation(confirm_query), f"Failed to match valid confirmation: '{confirm_query}'"


# ==============================================================================
# 4. Adversarial Findings: Vulnerabilities and Traps
# ==============================================================================

def test_vulnerability_negative_confirmation_trap() -> None:
    """Demonstrate vulnerability: Negative phrases containing confirmation words trigger deployment!

    When a user says 'não confirmar' or 'não quero executar', regex pattern
    matches 'confirmar' or 'executar' and treats it as affirmative confirmation!
    """
    trapped_negatives = [
        "não confirmar",
        "não quero executar",
        "não pode executar",
        "não aprovo",
    ]
    # Verify that negative phrases containing confirmation words are rejected as non-confirmations
    for q in trapped_negatives:
        assert not _is_confirmation(q), f"Expected '{q}' to be rejected as confirmation"


def test_vulnerability_silver_bronze_layer_entity_collision() -> None:
    """Demonstrate vulnerability: 'silver' and 'bronze' in ENTITY_ALIAS_MAP steal anaphoric references.

    In 'gerar etl silver dessa tabela', 'silver' is matched as entity name
    (medallion_silver_transactions) instead of layer, bypassing anaphora resolution for 'dessa tabela'.
    """
    silver_collision = _extract_entity_from_query("gerar etl silver dessa tabela")
    bronze_collision = _extract_entity_from_query("propor etl bronze dessa tabela")
    gold_no_collision = _extract_entity_from_query("propor etl gold dessa tabela")

    assert silver_collision == "medallion_silver_transactions"
    assert bronze_collision == "medallion_bronze_transactions"
    assert gold_no_collision is None  # Gold does not collide!


# ==============================================================================
# 5. Open WebUI Pipe End-to-End Simulation
# ==============================================================================

def test_pipe_3_turn_journey() -> None:
    """Simulate exact 3-turn user interaction through Open WebUI Pipe."""
    pipe = Pipe()

    # Turn 1: Preview
    body1 = {
        "messages": [
            {"role": "user", "content": "consulte os dados da tabela customers"}
        ],
        "stream": False,
    }
    resp1 = pipe.pipe(body1)
    assert isinstance(resp1, str)
    assert "Amostra de Dados" in resp1 or "customers" in resp1, f"Turn 1 failed: {resp1[:200]}"

    # Turn 2: ETL Proposal
    body2 = {
        "messages": [
            {"role": "user", "content": "consulte os dados da tabela customers"},
            {"role": "assistant", "content": resp1},
            {"role": "user", "content": "propor um etl gold a partir dessa tabela"},
        ],
        "stream": False,
    }
    resp2 = pipe.pipe(body2)
    assert isinstance(resp2, str)
    assert "PySpark" in resp2 or "CREATE OR REPLACE TABLE" in resp2 or "Medallion Pipeline" in resp2, (
        f"Turn 2 failed: {resp2[:200]}"
    )

    # Turn 3: Confirmation
    body3 = {
        "messages": [
            {"role": "user", "content": "consulte os dados da tabela customers"},
            {"role": "assistant", "content": resp1},
            {"role": "user", "content": "propor um etl gold a partir dessa tabela"},
            {"role": "assistant", "content": resp2},
            {"role": "user", "content": "sim, pode prosseguir"},
        ],
        "stream": False,
    }
    resp3 = pipe.pipe(body3)
    assert isinstance(resp3, str)
    # Check that lifecycle was executed: CI report, GitOps, and Databricks Job
    assert "Esteira de CI Quality Gate" in resp3 or "CI Quality Gate" in resp3, (
        f"Turn 3 missing CI report: {resp3[:300]}"
    )
    assert "Databricks Workflow Job" in resp3 or "Job ID" in resp3, (
        f"Turn 3 missing Databricks Job info: {resp3[:300]}"
    )
    assert "GitOps" in resp3, f"Turn 3 missing GitOps info: {resp3[:300]}"


# ==============================================================================
# 6. Edge Cases: Malformed Inputs, Extreme Strings, ReDoS, and State Leakage
# ==============================================================================

def test_graph_empty_messages() -> None:
    graph = create_steward_graph()
    res = graph.invoke({"messages": []})
    assert res is not None
    assert "response" in res
    assert "messages" in res


def test_graph_none_content() -> None:
    graph = create_steward_graph()
    res = graph.invoke({"messages": [{"role": "user", "content": ""}]})
    assert res is not None
    assert "response" in res


def test_graph_massive_input_resilience() -> None:
    """Stress test with 50k characters to check for regex catastrophic backtracking (ReDoS)."""
    graph = create_steward_graph()
    huge_input = "consulte os dados da tabela " + ("a" * 50000) + " customers"
    res = graph.invoke({"messages": [{"role": "user", "content": huge_input}]})
    assert res is not None
    assert "response" in res


def test_graph_special_characters_resilience() -> None:
    """Test SQL injection-like strings and unicode emojis."""
    graph = create_steward_graph()
    queries = [
        "consulte os dados da tabela customers; DROP TABLE workspace.default.customers;--",
        "propor um etl gold a partir dessa tabela 🚀🔥💻 ' OR '1'='1",
        "sim, pode prosseguir \x00\x01\x02",
    ]
    for q in queries:
        res = graph.invoke({"messages": [{"role": "user", "content": q}]})
        assert res is not None
        assert "response" in res
        assert not res.get("response", "").startswith("❌ Steward Agent Error")


# ==============================================================================
# 7. GitOps Force Add Verification
# ==============================================================================

def test_git_client_force_add(tmp_path: Any) -> None:
    """Verify GitClient commits files staged with -f even if ignored by gitignore."""
    git_client = GitClient(repo_path=tmp_path)
    git_client._run_git(["init"])

    # Create a .gitignore that ignores 'pipelines/'
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("pipelines/\n")
    git_client._run_git(["add", ".gitignore"])
    git_client._run_git(["-c", "user.name=Test", "-c", "user.email=t@test.com", "commit", "-m", "add gitignore"])

    # Now use commit_and_push_to_main on a file inside pipelines/
    files = {"pipelines/test_prod/schema.sql": "SELECT 1;"}
    res = git_client.commit_and_push_to_main(
        files=files,
        message="feat: add pipeline",
        remote="origin",
    )
    assert res["branch"] == "main"
    assert "pipelines/test_prod/schema.sql" in res["files"]

    # Verify file is actually in git repo
    ls_files = git_client._run_git(["ls-files", "pipelines/test_prod/schema.sql"])
    assert "pipelines/test_prod/schema.sql" in ls_files.stdout
