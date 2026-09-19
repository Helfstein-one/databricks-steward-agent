"""Unit and component integration tests for Mermaid diagram generation (Tier 1)."""

from src.databricks.introspector import _build_mock_entities
from src.semantic.models import (
    ColumnModel,
    EntityModel,
    RelationshipModel,
)
from src.visualizer.mermaid import (
    _sanitize_id,
    _sanitize_type,
    generate_er_diagram,
    generate_lineage_diagram,
)


def test_sanitize_helpers():
    """Verify sanitization of identifiers and data types for Mermaid syntax."""
    assert _sanitize_id("main.default.my-table") == "main_default_my_table"
    assert _sanitize_type("DECIMAL(18, 2)") == "decimal_18__2_"
    assert _sanitize_type("") == "string"


def test_generate_er_diagram_crows_foot_notation():
    """Verify Mermaid erDiagram generates valid Crow's foot cardinality and PK/FK markers."""
    ent_parent = EntityModel(
        name="customers",
        primary_key="customer_id",
        columns=[
            ColumnModel(name="customer_id", type="string", primary_key=True),
            ColumnModel(name="name", type="string"),
            ColumnModel(name="tier", type="string"),
        ],
    )

    ent_child = EntityModel(
        name="orders",
        primary_key="order_id",
        columns=[
            ColumnModel(name="order_id", type="string", primary_key=True),
            ColumnModel(name="customer_id", type="string"),
            ColumnModel(name="amount", type="double"),
        ],
        relationships=[
            RelationshipModel(
                from_entity="orders",
                to_entity="customers",
                from_column="customer_id",
                to_column="customer_id",
                type="many_to_one",
                description="places",
            )
        ],
    )

    erd = generate_er_diagram([ent_parent, ent_child])

    assert erd.startswith("erDiagram")
    # Verify relations first
    assert 'orders }o--|| customers : "places"' in erd
    # Verify entity blocks
    assert "customers {" in erd
    assert "string customer_id PK" in erd
    assert "string name" in erd
    assert "orders {" in erd
    assert "string order_id PK" in erd
    assert "string customer_id FK" in erd
    assert "double amount" in erd


def test_generate_er_diagram_cardinality_variants():
    """Verify different cardinality mappings (one_to_many, one_to_one, many_to_many)."""
    e1 = EntityModel(name="table_a", primary_key="id")
    e2 = EntityModel(name="table_b", primary_key="id")

    rels = [
        RelationshipModel(
            from_entity="table_a",
            to_entity="table_b",
            from_column="id",
            to_column="id",
            type="one_to_many",
        ),
        RelationshipModel(
            from_entity="table_a",
            to_entity="table_b",
            from_column="code",
            to_column="code",
            type="one_to_one",
        ),
        RelationshipModel(
            from_entity="table_a",
            to_entity="table_b",
            from_column="tag",
            to_column="tag",
            type="many_to_many",
        ),
    ]

    erd = generate_er_diagram([e1, e2], relationships=rels)
    assert "table_a ||--o{ table_b :" in erd
    assert "table_a ||--|| table_b :" in erd
    assert "table_a }o--o{ table_b :" in erd


def test_generate_er_diagram_with_demo_entities():
    """Verify erDiagram with standard mock lakehouse entities."""
    demo_entities = _build_mock_entities()
    erd = generate_er_diagram(demo_entities)

    assert "erDiagram" in erd
    assert "bronze_raw_transactions {" in erd
    assert "silver_transactions {" in erd
    assert "gold_sales_kpis {" in erd
    assert "string transaction_id PK" in erd


def test_generate_lineage_diagram_medallion_subgraphs():
    """Verify Medallion lineage flowchart partitions into Bronze, Silver, and Gold subgraphs."""
    demo_entities = _build_mock_entities()
    lineage = generate_lineage_diagram(demo_entities)

    assert lineage.startswith("graph LR")
    assert "subgraph Bronze" in lineage
    assert "subgraph Silver" in lineage
    assert "subgraph Gold" in lineage
    assert "bronze_raw_transactions" in lineage
    assert "silver_transactions" in lineage
    assert "gold_sales_kpis" in lineage
    # Verify connectors
    assert "-->" in lineage


def test_generate_lineage_diagram_custom_layer_fallback():
    """Verify lineage generation for entities with inferred layer names from naming convention."""
    e_raw = EntityModel(name="raw_events", table_name="raw.events")
    e_clean = EntityModel(name="clean_events", table_name="clean.events")
    e_kpi = EntityModel(name="kpi_summary", table_name="kpi.summary")

    lineage = generate_lineage_diagram([e_raw, e_clean, e_kpi])
    assert "subgraph Bronze" in lineage
    assert "subgraph Silver" in lineage
    assert "subgraph Gold" in lineage
    assert "raw_events -->" in lineage
