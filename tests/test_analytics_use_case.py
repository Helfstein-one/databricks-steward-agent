"""Unit and integration tests for NL2SQL Auto-Run for Semantic Analytics Questions (AnalyticsUseCase)."""

from unittest.mock import patch

from src.agent.graph import create_steward_graph
from src.agent.intent import _is_analytics_query
from src.application.use_cases.others import AnalyticsUseCase
from src.semantic.registry import SemanticRegistry


def test_is_analytics_query_detection():
    """Verify regex and intent detection for natural language analytics business questions."""
    assert _is_analytics_query("What was the total revenue yesterday?") is True
    assert _is_analytics_query("Qual o faturamento total por categoria?") is True
    assert _is_analytics_query("Qual o ticket médio por cliente?") is True
    assert _is_analytics_query("Qual o total de clientes VIP?") is True
    assert _is_analytics_query("Mostrar os registros da tabela orders") is False
    assert _is_analytics_query("O que é a camada semântica?") is False


def test_analytics_use_case_parsing(sample_yaml_dir):
    """Verify AnalyticsUseCase correctly parses natural language queries into semantic components."""
    use_case = AnalyticsUseCase()
    registry = SemanticRegistry(models_dir=sample_yaml_dir)

    # 1. Total revenue query with date filter
    ent, metrics, dims, filters = use_case._parse_nl_query("What was the total revenue yesterday?", registry)
    assert len(metrics) > 0
    assert ent is not None
    assert any("date = date_sub(current_date(), 1)" in f for f in filters)

    # 2. Revenue grouped by category
    ent, metrics, dims, filters = use_case._parse_nl_query("Qual o faturamento total por categoria?", registry)
    assert len(metrics) > 0
    assert "category" in dims

    # 3. VIP customers query
    ent, metrics, dims, filters = use_case._parse_nl_query("Qual o total de clientes VIP?", registry)
    assert len(metrics) > 0


def test_analytics_use_case_execution():
    """Verify AnalyticsUseCase compiles SQL, executes via DatabricksAdapter, and returns Markdown table."""
    use_case = AnalyticsUseCase()
    state = {"user_query": "Qual o faturamento total por categoria?"}

    mock_db_results = [
        {"category": "ELECTRONICS", "total_revenue": 150250.50},
        {"category": "FASHION", "total_revenue": 89400.00},
    ]

    with patch("src.application.use_cases.others.DatabricksAdapter") as MockAdapter:
        mock_instance = MockAdapter.return_value
        mock_instance.execute_query.return_value = mock_db_results

        res = use_case.execute(state)

        # Verify adapter call
        mock_instance.execute_query.assert_called_once()
        compiled_sql = mock_instance.execute_query.call_args[0][0]
        assert "SELECT" in compiled_sql
        assert "total_revenue" in compiled_sql
        assert "GROUP BY" in compiled_sql

        # Verify state output
        assert res.get("compiled_sql") == compiled_sql
        assert res.get("analytics_result") == mock_db_results

        response = res.get("response", "")
        assert "### 📊 Consulta Analítica Semântica (NL2SQL)" in response
        assert "Query SparkSQL Compilada via Camada Semântica:" in response
        assert "| category | total_revenue |" in response
        assert "| ELECTRONICS | 150250.5 |" in response
        assert "| FASHION | 89400.0 |" in response


def test_analytics_graph_routing_e2e():
    """Verify steward graph correctly routes natural language business questions to AnalyticsUseCase."""
    graph = create_steward_graph()
    mock_db_results = [{"total_revenue": 543210.00}]

    with patch("src.application.use_cases.others.DatabricksAdapter") as MockAdapter:
        mock_instance = MockAdapter.return_value
        mock_instance.execute_query.return_value = mock_db_results

        state = {"messages": [{"role": "user", "content": "What was the total revenue yesterday?"}]}
        res = graph.invoke(state)

        response = res.get("response", "")
        assert "### 📊 Consulta Analítica Semântica (NL2SQL)" in response
        assert "| total_revenue |" in response
        assert "| 543210.0 |" in response
