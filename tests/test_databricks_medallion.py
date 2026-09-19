"""Tests for the live Databricks Medallion semantic domain and routing."""

from src.agent.graph import steward_node
from src.agent.tools import (
    format_entity_modeling,
    generate_diagram,
    query_semantic_layer,
)
from src.config import settings
from src.semantic.registry import SemanticRegistry


def test_databricks_medallion_domain_registered():
    """Verify databricks_medallion is loaded and indexed in the semantic registry."""
    reg = SemanticRegistry(settings.semantic_models_path)
    assert "databricks_medallion" in reg.domains

    domain = reg.get_domain("databricks_medallion")
    assert domain is not None
    assert domain.catalog == "workspace"
    assert domain.schema_name == "default"

    # Entities
    entity_names = [e.name for e in domain.entities]
    assert "medallion_bronze_transactions" in entity_names
    assert "medallion_silver_transactions" in entity_names
    assert "medallion_gold_sales_kpis" in entity_names
    assert "medallion_gold_customer_kpis" in entity_names


def test_databricks_medallion_entity_synonyms():
    """Verify Portuguese and business synonyms resolve to the real Databricks tables."""
    reg = SemanticRegistry(settings.semantic_models_path)

    # Bronze
    bronze_ent = reg.get_entity("transacoes_bronze")
    assert bronze_ent is not None
    assert bronze_ent.name == "medallion_bronze_transactions"

    # Silver
    silver_ent = reg.get_entity("transações")
    assert silver_ent is not None
    assert silver_ent.name == "medallion_silver_transactions"

    # Gold sales
    sales_ent = reg.get_entity("kpis_vendas")
    assert sales_ent is not None
    assert sales_ent.name == "medallion_gold_sales_kpis"

    # Gold customers
    cust_ent = reg.get_entity("gold_customer_kpis")
    assert cust_ent is not None
    assert cust_ent.name == "medallion_gold_customer_kpis"


def test_databricks_medallion_relationships():
    """Verify relationships between Bronze, Silver, and Gold tables."""
    reg = SemanticRegistry(settings.semantic_models_path)
    rels = reg.list_relationships("databricks_medallion")
    assert len(rels) >= 3

    rel_pairs = [(r.from_entity, r.to_entity, r.type) for r in rels]
    assert (
        "medallion_bronze_transactions",
        "medallion_silver_transactions",
        "one_to_one",
    ) in rel_pairs
    assert (
        "medallion_silver_transactions",
        "medallion_gold_customer_kpis",
        "many_to_one",
    ) in rel_pairs
    assert (
        "medallion_silver_transactions",
        "medallion_gold_sales_kpis",
        "many_to_one",
    ) in rel_pairs


def test_databricks_medallion_query_compilation():
    """Verify SparkSQL compilation for real Lakehouse metrics and dimensions."""
    sql = query_semantic_layer(
        entity_name="medallion_silver_transactions",
        metric_names=["total_revenue", "total_transactions"],
        group_by_dims=["category", "date"],
    )
    assert "SELECT" in sql
    assert "workspace.default.medallion_silver_transactions" in sql
    assert "SUM(amount) AS total_revenue" in sql
    assert "GROUP BY" in sql


def test_databricks_medallion_modeling_formatting():
    """Verify format_entity_modeling outputs complete real table details and ERD."""
    output = format_entity_modeling("transações")
    assert "medallion_silver_transactions" in output
    assert "workspace.default.medallion_silver_transactions" in output
    assert "transaction_id" in output
    assert "total_revenue" in output
    assert "erDiagram" in output
    assert "medallion_bronze_transactions" in output


def test_databricks_medallion_diagram_default():
    """Verify generate_diagram defaults to databricks_medallion when no domain is passed."""
    output = generate_diagram("er")
    assert "erDiagram" in output
    assert "medallion_silver_transactions" in output
    assert "medallion_gold_sales_kpis" in output
    assert "medallion_bronze_transactions" in output


def test_databricks_medallion_graph_routing_relations_query():
    """Verify user query asking for table relationships returns databricks_medallion model."""
    res = steward_node(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Como você gostaria de visualizar as relações entre as tabelas e schemas descobertas?",
                }
            ],
            "user_query": "Como você gostaria de visualizar as relações entre as tabelas e schemas descobertas?",
        }
    )
    resp = res["response"]
    assert "medallion_silver_transactions" in resp
    assert "medallion_bronze_transactions" in resp
    assert "corporate_credit" not in resp
    assert "facilities" not in resp
