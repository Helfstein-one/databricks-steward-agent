"""Unit and integration tests for LLM-Enriched Semantic Auto-Discovery Crawler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.cli.discovery import enrich_entity_with_llm, main, run_discovery_crawler
from src.semantic.models import ColumnModel, EntityModel, SemanticDomainModel


@pytest.fixture
def sample_entity() -> EntityModel:
    return EntityModel(
        name="test_orders",
        table_name="main.default.test_orders",
        catalog="main",
        schema_name="default",
        primary_key="order_id",
        columns=[
            ColumnModel(name="order_id", type="string", primary_key=True, nullable=False),
            ColumnModel(name="customer_id", type="string", nullable=False),
            ColumnModel(name="amount", type="double", nullable=False),
            ColumnModel(name="quantity", type="int", nullable=False),
            ColumnModel(name="order_date", type="date", nullable=False),
        ],
    )


def test_enrich_entity_heuristic_fallback(sample_entity: EntityModel) -> None:
    enriched = enrich_entity_with_llm(sample_entity, llm=None)

    assert enriched.name == "test_orders"
    assert enriched.primary_key == "order_id"
    assert len(enriched.dimensions) == 5

    metric_names = [m.name for m in enriched.metrics]
    assert "total_test_orders_count" in metric_names
    assert "sum_amount" in metric_names
    assert "avg_amount" in metric_names
    assert "sum_quantity" in metric_names
    assert "avg_quantity" in metric_names


def test_enrich_entity_with_mock_llm(sample_entity: EntityModel) -> None:
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """```json
{
    "description": "E-commerce orders entity containing customer purchases.",
    "synonyms": ["orders", "pedidos"],
    "dimensions": [
        {"name": "order_id", "type": "string", "column": "order_id", "description": "Unique identifier for the order"},
        {"name": "customer_id", "type": "string", "column": "customer_id", "description": "Customer unique ID"},
        {"name": "amount", "type": "double", "column": "amount", "description": "Order monetary total"},
        {"name": "quantity", "type": "int", "column": "quantity", "description": "Number of items ordered"},
        {"name": "order_date", "type": "date", "column": "order_date", "description": "Date order was placed"}
    ],
    "metrics": [
        {"name": "total_revenue", "type": "sum", "sql": "SUM(amount)", "description": "Total gross order revenue"},
        {"name": "order_count", "type": "count", "sql": "COUNT(DISTINCT order_id)", "description": "Total number of orders"}
    ]
}
```"""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
        enriched = enrich_entity_with_llm(sample_entity, llm=mock_llm)

    assert enriched.description == "E-commerce orders entity containing customer purchases."
    assert "pedidos" in enriched.synonyms
    assert len(enriched.dimensions) == 5
    assert len(enriched.metrics) == 2
    assert enriched.metrics[0].name == "total_revenue"
    assert enriched.metrics[0].sql == "SUM(amount)"


def test_run_discovery_crawler_e2e(tmp_path: Path, sample_entity: EntityModel) -> None:
    out_yaml = tmp_path / "discovered_test.yaml"

    with patch("src.cli.discovery.DatabricksAdapter") as mock_adapter_cls:
        mock_adapter = MagicMock()
        mock_adapter.introspect_catalog.return_value = [sample_entity]
        mock_adapter_cls.return_value = mock_adapter

        domain_model = run_discovery_crawler(
            catalog="main",
            schema="test_schema",
            output_path=out_yaml,
            use_llm=False,
        )

        assert isinstance(domain_model, SemanticDomainModel)
        assert domain_model.domain == "auto_discovered_test_schema"
        assert domain_model.catalog == "main"
        assert domain_model.schema_name == "test_schema"
        assert len(domain_model.entities) == 1

        assert out_yaml.exists()
        with open(out_yaml, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        assert yaml_data["domain"] == "auto_discovered_test_schema"
        assert len(yaml_data["entities"]) == 1
        assert yaml_data["entities"][0]["name"] == "test_orders"


def test_cli_main_entrypoint(tmp_path: Path, sample_entity: EntityModel) -> None:
    out_yaml = tmp_path / "cli_out.yaml"

    test_args = [
        "discovery.py",
        "--catalog",
        "demo_cat",
        "--schema",
        "demo_sch",
        "--output",
        str(out_yaml),
        "--no-llm",
    ]

    with (
        patch("sys.argv", test_args),
        patch("src.cli.discovery.DatabricksAdapter") as mock_adapter_cls,
    ):
        mock_adapter = MagicMock()
        mock_adapter.introspect_catalog.return_value = [sample_entity]
        mock_adapter_cls.return_value = mock_adapter

        main()

        assert out_yaml.exists()
        with open(out_yaml, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        assert yaml_data["domain"] == "auto_discovered_demo_sch"
