"""Unit and component integration tests for declarative semantic layer (Tiers 1 & 2)."""

import pytest

from src.semantic.models import (
    ColumnModel,
    DimensionModel,
    EntityModel,
    MetricModel,
    RelationshipModel,
)
from src.semantic.registry import SemanticRegistry

# ==============================================================================
# Model Unit Tests (Tier 1)
# ==============================================================================


def test_column_model():
    """Verify ColumnModel creation and defaults."""
    col = ColumnModel(name="user_id", type="string", primary_key=True)
    assert col.name == "user_id"
    assert col.type == "string"
    assert col.primary_key is True
    assert col.nullable is True
    assert col.foreign_key is None


def test_dimension_model_default_column():
    """Verify DimensionModel automatically sets column to name if not provided."""
    dim = DimensionModel(name="sector", type="string", synonyms=["setor", "ramo"])
    assert dim.name == "sector"
    assert dim.column == "sector"
    assert "setor" in dim.synonyms


def test_dimension_model_explicit_column():
    """Verify DimensionModel preserves explicit column mappings."""
    dim = DimensionModel(name="economic_group", column="nm_economic_group", synonyms=["grupo"])
    assert dim.name == "economic_group"
    assert dim.column == "nm_economic_group"


def test_metric_model():
    """Verify MetricModel attributes and division safety expression."""
    metric = MetricModel(
        name="overdue_ratio_90d",
        entity="credit_facility",
        type="derived",
        sql="SUM(CASE WHEN days > 90 THEN bal ELSE 0 END) / NULLIF(SUM(bal), 0)",
        synonyms=["npl_90d"],
    )
    assert metric.name == "overdue_ratio_90d"
    assert metric.type == "derived"
    assert "NULLIF(" in metric.sql


def test_relationship_model():
    """Verify RelationshipModel validation and cardinality."""
    rel = RelationshipModel(
        from_entity="facilities",
        to_entity="counterparts",
        from_column="counterpart_id",
        to_column="counterpart_id",
        type="many_to_one",
    )
    assert rel.from_entity == "facilities"
    assert rel.to_entity == "counterparts"
    assert rel.type == "many_to_one"


def test_entity_model_normalization():
    """Verify EntityModel normalizes 'table' to 'table_name' and builds columns."""
    data = {
        "name": "counterpart",
        "table": "main.credit.counterparts",
        "primary_key": "counterpart_id",
        "dimensions": [
            {"name": "economic_group", "column": "nm_group"},
            {"name": "sector", "type": "string"},
        ],
        "synonyms": ["empresa", "cliente"],
    }
    entity = EntityModel(**data)
    assert entity.name == "counterpart"
    assert entity.table_name == "main.credit.counterparts"
    assert entity.primary_key == "counterpart_id"
    assert len(entity.columns) == 3  # PK + 2 dimensions
    col_names = [c.name for c in entity.columns]
    assert "counterpart_id" in col_names
    assert "nm_group" in col_names
    assert "sector" in col_names


# ==============================================================================
# Semantic Registry Tests (Tier 1 & 2)
# ==============================================================================


def test_registry_empty_initialization():
    """Verify SemanticRegistry initializes with empty collections."""
    registry = SemanticRegistry()
    assert len(registry.domains) == 0
    assert len(registry.entities) == 0
    assert len(registry.metrics) == 0
    assert len(registry.relationships) == 0
    assert "No semantic models registered." in registry.get_prompt_context()


def test_registry_load_directory(sample_yaml_dir):
    """Verify SemanticRegistry loads and indexes all YAML files from directory."""
    registry = SemanticRegistry(models_dir=sample_yaml_dir)

    assert "corporate_credit" in registry.domains
    assert "sales_lakehouse" in registry.domains

    # Verify entity registration
    assert registry.get_entity("counterpart") is not None
    assert registry.get_entity("credit_facility") is not None
    assert registry.get_entity("customer") is not None
    assert registry.get_entity("order") is not None

    # Verify metric registration
    assert registry.get_metric("total_exposure") is not None
    assert registry.get_metric("total_revenue") is not None
    assert registry.get_metric("overdue_ratio_90d") is not None


def test_registry_synonym_resolution(sample_yaml_dir):
    """Verify business synonyms resolve correctly to entities, dimensions, and metrics."""
    registry = SemanticRegistry(models_dir=sample_yaml_dir)

    # Entity synonym resolution
    entity_via_synonym = registry.get_entity("tomador")
    assert entity_via_synonym is not None
    assert entity_via_synonym.name == "counterpart"

    entity_via_synonym_2 = registry.get_entity("operacao")
    assert entity_via_synonym_2 is not None
    assert entity_via_synonym_2.name == "credit_facility"

    # Metric synonym resolution
    metric_via_synonym = registry.get_metric("exposicao_total")
    assert metric_via_synonym is not None
    assert metric_via_synonym.name == "total_exposure"

    metric_via_synonym_2 = registry.get_metric("faturamento")
    assert metric_via_synonym_2 is not None
    assert metric_via_synonym_2.name == "total_revenue"

    # Direct synonym lookup index
    resolved = registry.resolve_synonym("grupo")
    assert resolved is not None
    assert resolved["type"] == "dimension"
    assert resolved["target"] == "economic_group"


def test_registry_prompt_context_generation(sample_yaml_dir):
    """Verify serialization of prompt context for LLM grounding."""
    registry = SemanticRegistry(models_dir=sample_yaml_dir)
    context = registry.get_prompt_context("corporate_credit")

    assert "### Domain: corporate_credit" in context
    assert "Entity: `counterpart`" in context
    assert "Entity: `credit_facility`" in context
    assert "Table: `main.credit_risk.counterparts`" in context
    assert "Dimensions:" in context
    assert "Relationships (Joins):" in context


def test_registry_nonexistent_directory():
    """Verify non-existent directory does not crash the registry."""
    registry = SemanticRegistry(models_dir="/path/that/does/not/exist/at/all")
    assert len(registry.domains) == 0


def test_registry_invalid_file_handling(tmp_path):
    """Verify loading non-dictionary YAML raises appropriate error."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- list\n- not\n- dictionary", encoding="utf-8")

    registry = SemanticRegistry()
    with pytest.raises(TypeError):
        registry.load_file(bad_yaml)


# ==============================================================================
# Semantic Compiler & Graph Join Resolver Tests (Tier 2)
# ==============================================================================


def test_ensure_nullif_division_safety():
    """Verify division expressions are safely wrapped with NULLIF(..., 0)."""
    from src.semantic.compiler import ensure_nullif_division_safety

    safe_sql = ensure_nullif_division_safety("col_a / col_b")
    assert "NULLIF(" in safe_sql
    assert "NULLIF(col_b, 0)" in safe_sql

    # Already safe expression should not be doubly wrapped
    already_safe = "col_a / NULLIF(col_b, 0)"
    assert ensure_nullif_division_safety(already_safe) == already_safe


def test_graph_join_resolver():
    """Verify BFS shortest path join resolution between entities."""
    from src.semantic.compiler import GraphJoinResolver

    rels = [
        RelationshipModel(
            from_entity="facilities",
            to_entity="counterparts",
            from_column="counterpart_id",
            to_column="counterpart_id",
            type="many_to_one",
        ),
        RelationshipModel(
            from_entity="counterparts",
            to_entity="economic_groups",
            from_column="group_id",
            to_column="group_id",
            type="many_to_one",
        ),
    ]

    resolver = GraphJoinResolver(rels)
    # Direct path
    path_direct = resolver.find_shortest_path("facilities", "counterparts")
    assert path_direct is not None
    assert len(path_direct) == 1
    assert path_direct[0][0] == "counterparts"

    # Multi-hop path: facilities -> counterparts -> economic_groups
    path_multihop = resolver.find_shortest_path("facilities", "economic_groups")
    assert path_multihop is not None
    assert len(path_multihop) == 2
    assert path_multihop[0][0] == "counterparts"
    assert path_multihop[1][0] == "economic_groups"

    # Self path
    assert resolver.find_shortest_path("facilities", "facilities") == []

    # Disconnected entity
    assert resolver.find_shortest_path("facilities", "unrelated_table") is None


def test_compiler_query_compilation_with_nullif(sample_yaml_dir):
    """Verify compiler compiles query with NULLIF division safety and joins."""
    from src.semantic.compiler import SemanticQueryCompiler

    registry = SemanticRegistry(models_dir=sample_yaml_dir)
    compiler = SemanticQueryCompiler(registry)

    sql = compiler.compile_query(
        entity_name="credit_facility",
        metric_names=["total_exposure", "overdue_ratio_90d"],
        group_by_dims=["operation_type"],
        filters=["credit_facility.status = 'ACTIVE'"],
        order_by=["total_exposure DESC"],
        limit=25,
    )

    assert "SELECT" in sql
    assert "credit_facility.tp_operation AS operation_type" in sql
    assert "SUM(vl_outstanding_balance) AS total_exposure" in sql
    assert "NULLIF(" in sql
    assert "FROM main.credit_risk.facilities AS credit_facility" in sql
    assert "WHERE credit_facility.status = 'ACTIVE'" in sql
    assert "GROUP BY credit_facility.tp_operation" in sql
    assert "ORDER BY total_exposure DESC" in sql
    assert "LIMIT 25" in sql


def test_compiler_empty_request_raises_error(sample_yaml_dir):
    """Verify compiler raises ValueError when neither metrics nor dimensions are requested."""
    from src.semantic.compiler import SemanticQueryCompiler

    registry = SemanticRegistry(models_dir=sample_yaml_dir)
    compiler = SemanticQueryCompiler(registry)

    with pytest.raises(ValueError):
        compiler.compile_query(entity_name="credit_facility")
