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
        "messages": [
            {"role": "user", "content": "Validar codigo na esteira de CI com Ruff e SQLFluff"}
        ],
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
    assert pipe.valves.DATABRICKS_DEFAULT_CATALOG == "workspace"
    assert pipe.valves.DATABRICKS_DEFAULT_SCHEMA == "default"
    assert pipe.valves.LOCAL_LLM_MODEL == "llama3.2:3b"
    assert pipe.valves.GITHUB_BASE_BRANCH == "workspace"


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


def test_steward_node_numeric_menu_shortcuts():
    """Verify numeric menu options 1 to 6 route to expected capabilities."""
    # Option 1: Unity Catalog
    out1 = steward_node({"messages": [{"role": "user", "content": "1"}], "user_query": "1"})
    assert "Discovered" in out1["response"]

    # Option 2: Semantic Models
    out2 = steward_node({"messages": [{"role": "user", "content": "2"}], "user_query": "2"})
    assert "Domain" in out2["response"]

    # Option 3: Mermaid Diagram
    out3 = steward_node({"messages": [{"role": "user", "content": "3"}], "user_query": "3"})
    assert "```mermaid" in out3["response"]

    # Option 4: ETL Pipeline
    out4 = steward_node({"messages": [{"role": "user", "content": "4"}], "user_query": "4"})
    assert "PySpark Pipeline" in out4["response"]

    # Option 5: CI Quality Gate
    out5 = steward_node({"messages": [{"role": "user", "content": "5"}], "user_query": "5"})
    assert "CI Quality Gate Report" in out5["response"]

    # Option 6: GitOps PR
    out6 = steward_node({"messages": [{"role": "user", "content": "6"}], "user_query": "6"})
    assert "PR" in out6["response"]


def test_steward_node_title_request():
    """Verify background UI title generation requests return concise titles without loops."""
    title_prompts = [
        "Generate a brief 3-5 word title for this chat based on the conversation:",
        "Summarize the conversation in 3-5 words for a chat title",
        "Generate a title for this chat",
    ]
    for prompt in title_prompts:
        out = steward_node(
            {"messages": [{"role": "user", "content": prompt}], "user_query": prompt}
        )
        assert "Databricks Steward - Governança" in out["response"]
        assert "Hello!" not in out["response"]


def test_steward_node_portuguese_greetings():
    """Verify conversational Portuguese greetings return the friendly stewardship welcome menu."""
    for greet in ["olá", "oi", "bom dia", "boa tarde"]:
        out = steward_node({"messages": [{"role": "user", "content": greet}], "user_query": greet})
        assert "Databricks Steward Agent" in out["response"]
        assert "Unity Catalog Introspection" in out["response"]
        assert "Dica:" in out["response"]


def test_steward_node_conceptual_explanations():
    """Verify conceptual questions return rich educational explanations instead of raw metadata dumps."""
    # Semantic layer
    sem_out = steward_node(
        {
            "messages": [{"role": "user", "content": "o que é camada semantica ?"}],
            "user_query": "o que é camada semantica ?",
        }
    )
    assert "Camada Semântica (Semantic Layer)" in sem_out["response"]
    assert "Fonte Única da Verdade" in sem_out["response"]
    assert "corporate_credit" in sem_out["response"]

    # Unity Catalog
    uc_out = steward_node(
        {
            "messages": [{"role": "user", "content": "o que é o unity catalog?"}],
            "user_query": "o que é o unity catalog?",
        }
    )
    assert "Unity Catalog" in uc_out["response"]
    assert "3 Níveis" in uc_out["response"]

    # Medallion Architecture
    med_out = steward_node(
        {
            "messages": [{"role": "user", "content": "como funciona a arquitetura medalhao?"}],
            "user_query": "como funciona a arquitetura medalhao?",
        }
    )
    assert "Arquitetura Medalhão" in med_out["response"]
    assert "Bronze" in med_out["response"]
    assert "Silver" in med_out["response"]
    assert "Gold" in med_out["response"]

    # CI Quality Gate
    ci_out = steward_node(
        {
            "messages": [{"role": "user", "content": "para que serve a esteira de ci?"}],
            "user_query": "para que serve a esteira de ci?",
        }
    )
    assert "Esteira de CI" in ci_out["response"]
    assert "Ruff" in ci_out["response"]
    assert "SQLFluff" in ci_out["response"]

    # GitOps
    git_out = steward_node(
        {
            "messages": [{"role": "user", "content": "o que é gitops no lakehouse?"}],
            "user_query": "o que é gitops no lakehouse?",
        }
    )
    assert "GitOps" in git_out["response"]
    assert "Pull Request" in git_out["response"]

    # Mermaid
    m_out = steward_node(
        {
            "messages": [{"role": "user", "content": "o que sao diagramas mermaid?"}],
            "user_query": "o que sao diagramas mermaid?",
        }
    )
    assert "Mermaid.js" in m_out["response"]
    assert "erDiagram" in m_out["response"]


def test_steward_node_entity_modeling_queries():
    """Verify specific table modeling queries return detailed columns, PK, metrics, and erDiagram."""
    # 1. Customers modeling
    cust_out = steward_node(
        {
            "messages": [{"role": "user", "content": "qual a modelagem de customers"}],
            "user_query": "qual a modelagem de customers",
        }
    )
    assert "### 📐 Modelagem de Dados: `customers`" in cust_out["response"]
    assert "main.sales.customers" in cust_out["response"]
    assert "sales_lakehouse" in cust_out["response"]
    assert "customer_id" in cust_out["response"]
    assert "total_customers" in cust_out["response"]
    assert "erDiagram" in cust_out["response"]
    assert cust_out["active_diagram"] is not None
    assert "erDiagram" in cust_out["active_diagram"]

    # 2. Orders table modeling
    orders_out = steward_node(
        {
            "messages": [{"role": "user", "content": "qual a modelagem da tabela orders"}],
            "user_query": "qual a modelagem da tabela orders",
        }
    )
    assert "### 📐 Modelagem de Dados: `orders`" in orders_out["response"]
    assert "main.sales.orders" in orders_out["response"]
    assert "order_id" in orders_out["response"]
    assert "total_orders" in orders_out["response"]
    assert "erDiagram" in orders_out["active_diagram"]

    # 3. Facilities schema
    fac_out = steward_node(
        {
            "messages": [{"role": "user", "content": "schema de facilities"}],
            "user_query": "schema de facilities",
        }
    )
    assert "### 📐 Modelagem de Dados: `facilities`" in fac_out["response"]
    assert "main.corporate_credit.credit_facilities" in fac_out["response"]
    assert "facility_id" in fac_out["response"]
    assert "utilization_rate" in fac_out["response"]
    assert "erDiagram" in fac_out["active_diagram"]

    # 4. Impairments structure
    imp_out = steward_node(
        {
            "messages": [{"role": "user", "content": "estrutura da tabela impairments"}],
            "user_query": "estrutura da tabela impairments",
        }
    )
    assert "### 📐 Modelagem de Dados: `impairments`" in imp_out["response"]
    assert "main.corporate_credit.impairments" in imp_out["response"]
    assert "impairment_id" in imp_out["response"]


def test_generate_diagram_modelagem_keyword():
    """Verify diagram_type='modelagem' generates erDiagram instead of lineage flowchart."""
    from src.agent.tools import generate_diagram

    diag_modelagem = generate_diagram("modelagem")
    assert "erDiagram" in diag_modelagem
    assert "graph LR" not in diag_modelagem

    diag_cust = generate_diagram("er", domain="customers")
    assert "erDiagram" in diag_cust
    assert "customers {" in diag_cust
