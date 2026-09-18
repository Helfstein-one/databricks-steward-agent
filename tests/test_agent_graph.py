"""Unit and component integration tests for LangGraph state graph and Open WebUI Pipe (Tier 3)."""

from collections.abc import Generator

from open_webui_pipe import Pipe
from src.agent.graph import create_steward_graph, steward_node
from src.agent.state import AgentState
from src.agent.tools import STEWARD_TOOLS

# ==============================================================================
# Agent State & Tools Tests
# ==============================================================================


def test_agent_state_schema():
    """Verify AgentState TypedDict contains expected workflow keys."""
    state: AgentState = {
        "messages": [],
        "catalog_context": None,
        "semantic_entities": [],
        "active_diagram": None,
        "generated_code": None,
        "ci_report": None,
        "gitops_result": None,
        "user_query": "test query",
        "response": None,
    }
    assert state["user_query"] == "test query"


def test_steward_tools_registry():
    """Verify tool functions are exposed in STEWARD_TOOLS registry."""
    tool_names = [t["name"] for t in STEWARD_TOOLS]
    assert "inspect_unity_catalog" in tool_names
    assert "load_semantic_models" in tool_names
    assert "query_semantic_layer" in tool_names
    assert "generate_diagram" in tool_names
    assert "generate_etl_pipeline" in tool_names
    assert "run_ci" in tool_names
    assert "submit_gitops_pr" in tool_names


# ==============================================================================
# LangGraph Graph & Node Routing Tests
# ==============================================================================


def test_create_steward_graph():
    """Verify compilation of LangGraph StateGraph workflow."""
    graph = create_steward_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_steward_node_routing_diagram():
    """Verify user query asking for ER diagram routes to Mermaid generator."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Por favor desenhar o diagrama ER do lakehouse"}],
        "user_query": "Por favor desenhar o diagrama ER do lakehouse",
    }
    output = steward_node(state)
    assert "response" in output
    assert "```mermaid" in output["response"]
    assert "erDiagram" in output["response"]
    assert output["active_diagram"] is not None


def test_steward_node_routing_catalog():
    """Verify user query asking for catalog inspection invokes Unity Catalog tool."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Quais sao as tabelas no unity catalog?"}],
        "user_query": "Quais sao as tabelas no unity catalog?",
    }
    output = steward_node(state)
    assert "Discovered" in output["response"]
    assert "entities" in output["response"]


def test_steward_node_routing_etl():
    """Verify user query asking for ETL pipeline generates PySpark and SparkSQL."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Gerar pipeline ETL silver para transactions"}],
        "user_query": "Gerar pipeline ETL silver para transactions",
    }
    output = steward_node(state)
    assert "PySpark Pipeline" in output["response"]
    assert "SparkSQL DDL" in output["response"]
    assert "run_silver_pipeline" in output["response"]


def test_steward_node_routing_ci():
    """Verify user query asking for CI quality check runs CI pipeline."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Validar codigo na esteira de CI com Ruff e SQLFluff"}],
        "user_query": "Validar codigo na esteira de CI com Ruff e SQLFluff",
    }
    output = steward_node(state)
    assert "CI Quality Gate Report" in output["response"]


def test_steward_node_routing_gitops():
    """Verify user query asking for GitOps PR triggers PR creation."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Abrir pull request via GitOps"}],
        "user_query": "Abrir pull request via GitOps",
    }
    output = steward_node(state)
    assert "PR" in output["response"]


def test_steward_node_default_greeting():
    """Verify generic query returns steward capabilities help menu."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Ola, quem e voce?"}],
        "user_query": "Ola, quem e voce?",
    }
    output = steward_node(state)
    assert "Databricks Steward Agent" in output["response"]
    assert "Unity Catalog Introspection" in output["response"]
    assert "Automated GitOps" in output["response"]


# ==============================================================================
# Open WebUI Pipe Tests
# ==============================================================================


def test_open_webui_pipe_valves_configuration():
    """Verify Valves model exposes proper defaults and configuration fields."""
    pipe = Pipe()
    assert hasattr(pipe, "valves")
    assert pipe.valves.DATABRICKS_DEFAULT_CATALOG == "main"
    assert pipe.valves.DATABRICKS_DEFAULT_SCHEMA == "default"
    assert pipe.valves.LOCAL_LLM_MODEL == "qwen2.5-coder:7b"
    assert pipe.valves.GITHUB_BASE_BRANCH == "main"


def test_open_webui_pipe_execution_non_streaming():
    """Verify Pipe.pipe returns full string response when stream=False."""
    pipe = Pipe()
    body = {
        "messages": [{"role": "user", "content": "Desenhar modelo de diagramas"}],
        "stream": False,
    }
    result = pipe.pipe(body)
    assert isinstance(result, str)
    assert "```mermaid" in result


def test_open_webui_pipe_execution_streaming():
    """Verify Pipe.pipe returns generator yielding chunks when stream=True."""
    pipe = Pipe()
    body = {
        "messages": [{"role": "user", "content": "Listar tabelas do catalogo"}],
        "stream": True,
    }
    result = pipe.pipe(body)
    assert isinstance(result, Generator)

    chunks = list(result)
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "Discovered" in full_text
