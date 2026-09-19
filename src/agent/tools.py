"""LangGraph agent tools exposing Databricks, Semantic, Mermaid, CI, and GitOps capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ci.runner import run_ci_pipeline
from src.config import settings
from src.databricks.introspector import introspect_catalog
from src.etl.generator import generate_medallion_pipeline
from src.gitops.github_pr import create_data_product_pr
from src.semantic.compiler import SemanticQueryCompiler
from src.semantic.registry import SemanticRegistry
from src.visualizer.mermaid import generate_er_diagram, generate_lineage_diagram


def inspect_unity_catalog(catalog: str = "main", schema: str = "default") -> str:
    """Introspect Databricks Unity Catalog tables, columns, and constraints."""
    entities = introspect_catalog(catalog=catalog, schema=schema)
    lines = [f"Discovered {len(entities)} entities in {catalog}.{schema}:"]
    for ent in entities:
        col_summary = ", ".join([f"{c.name} ({c.type})" for c in ent.columns[:6]])
        lines.append(f"- **{ent.name}** (layer: {ent.layer or 'unassigned'}): {col_summary}")
    return "\n".join(lines)


def load_semantic_models(path: str | Path | None = None) -> str:
    """Load and format semantic domain ontologies from YAML files."""
    if path and Path(path).exists() and Path(path).is_dir():
        models_path = Path(path)
    else:
        models_path = settings.semantic_models_path
    reg = SemanticRegistry(models_path)
    return reg.get_prompt_context()


def query_semantic_layer(
    entity_name: str,
    metric_names: list[str] | str,
    group_by_dims: list[str] | str,
) -> str:
    """Compile business metrics and dimensions into an optimized SparkSQL query."""
    reg = SemanticRegistry(settings.semantic_models_path)
    compiler = SemanticQueryCompiler(reg)

    if not entity_name:
        available = list(reg.entities.keys())
        entity_name = available[0] if available else "facilities"

    if isinstance(metric_names, str):
        metric_names = [m.strip() for m in metric_names.split(",") if m.strip()]
    if isinstance(group_by_dims, str):
        group_by_dims = [d.strip() for d in group_by_dims.split(",") if d.strip()]

    try:
        sql = compiler.compile_query(
            entity_name=entity_name,
            metric_names=metric_names,
            group_by_dims=group_by_dims,
        )
        return f"```sql\n{sql}\n```"
    except Exception as e:  # noqa: BLE001
        return f"Error compiling semantic query: {e}"


def generate_diagram(diagram_type: str = "er", domain: str | None = None) -> str:
    """Generate Mermaid erDiagram (Crow's foot) or Medallion lineage flowchart."""
    reg = SemanticRegistry(settings.semantic_models_path)

    # If domain specified, retrieve domain entities
    if domain:
        dom = reg.get_domain(domain)
        entities = dom.entities if dom else reg.list_entities()
        relationships = dom.relationships if dom else reg.list_relationships()
    else:
        entities = reg.list_entities()
        relationships = reg.list_relationships()

    if not entities:
        from src.databricks.introspector import _build_mock_entities
        entities = _build_mock_entities()
        relationships = []

    if diagram_type.lower() in ("er", "erd", "erdiagram"):
        mermaid_code = generate_er_diagram(entities, relationships)
    else:
        mermaid_code = generate_lineage_diagram(entities)

    return f"```mermaid\n{mermaid_code}\n```"


def generate_etl_pipeline(entity_name: str, layer: str = "silver") -> str:
    """Generate Medallion PySpark and SparkSQL pipelines for a given entity."""
    reg = SemanticRegistry(settings.semantic_models_path)
    entity = reg.get_entity(entity_name)

    if not entity:
        from src.databricks.introspector import _build_mock_entities
        mock_ents = {e.name: e for e in _build_mock_entities()}
        entity = mock_ents.get(entity_name)

    if not entity:
        return f"Entity '{entity_name}' not found."

    pipeline = generate_medallion_pipeline(entity, layer=layer)
    return (
        f"### Generated Medallion Pipeline: {pipeline.table_name} ({pipeline.layer})\n\n"
        f"#### PySpark Pipeline\n```python\n{pipeline.pyspark_code}\n```\n\n"
        f"#### SparkSQL DDL & Ingestion\n```sql\n{pipeline.sparksql_code}\n```"
    )


def run_ci(pyspark_code: str | None = None, sparksql_code: str | None = None) -> str:
    """Execute CI quality gate verification with Ruff, SQLFluff, and Anti-pattern rules."""
    report = run_ci_pipeline(pyspark_code=pyspark_code, sparksql_code=sparksql_code)
    return report.summary_markdown or report.format_markdown()


def submit_gitops_pr(
    product_name: str,
    pyspark_code: str,
    sparksql_code: str,
    diagram_md: str = "",
) -> str:
    """Execute CI check and open GitHub PR if quality checks pass."""
    report = run_ci_pipeline(pyspark_code=pyspark_code, sparksql_code=sparksql_code)
    if not report.is_approved:
        return f"❌ PR Creation Blocked: CI Quality Gate Rejected the pipeline.\n\n{report.summary_markdown}"

    files = {
        f"pipelines/{product_name}/etl.py": pyspark_code,
        f"pipelines/{product_name}/schema.sql": sparksql_code,
    }

    result = create_data_product_pr(
        product_name=product_name,
        files=files,
        ci_report=report,
        diagram_md=diagram_md,
        dry_run=True,
    )

    return (
        f"✅ PR Successfully Created!\n\n"
        f"- **Branch:** `{result.branch_name}`\n"
        f"- **Commit SHA:** `{result.commit_sha}`\n"
        f"- **PR URL:** [{result.pr_url}]({result.pr_url})\n\n"
        f"#### CI Gate Report\n{report.summary_markdown}"
    )


from langchain_core.tools import tool

STEWARD_TOOLS: list[dict[str, Any]] = [
    {
        "name": "inspect_unity_catalog",
        "description": "Discover Unity Catalog catalogs, schemas, tables, and columns.",
        "func": inspect_unity_catalog,
    },
    {
        "name": "load_semantic_models",
        "description": "Inspect registered business entities, dimensions, and metrics.",
        "func": load_semantic_models,
    },
    {
        "name": "query_semantic_layer",
        "description": "Compile analytical queries using semantic business metrics and dimensions.",
        "func": query_semantic_layer,
    },
    {
        "name": "generate_diagram",
        "description": "Draw Mermaid ER diagrams or medallion lineage flowcharts.",
        "func": generate_diagram,
    },
    {
        "name": "generate_etl_pipeline",
        "description": "Generate PySpark and SparkSQL pipelines for Bronze, Silver, or Gold layers.",
        "func": generate_etl_pipeline,
    },
    {
        "name": "run_ci",
        "description": "Execute Ruff and SQLFluff CI quality gate and anti-pattern checks.",
        "func": run_ci,
    },
    {
        "name": "submit_gitops_pr",
        "description": "Run CI and create GitHub feature branch, commit, and Pull Request.",
        "func": submit_gitops_pr,
    },
]


@tool
def inspect_unity_catalog_tool(catalog: str = "main", schema_name: str = "default") -> str:
    """Discover Unity Catalog catalogs, schemas, tables, and columns."""
    return inspect_unity_catalog(catalog=catalog, schema=schema_name)


@tool
def load_semantic_models_tool(path: str | None = None) -> str:
    """Inspect registered business entities, dimensions, and metrics in the semantic layer."""
    return load_semantic_models(path=path)


@tool
def query_semantic_layer_tool(
    entity_name: str,
    metric_names: list[str] | str,
    group_by_dims: list[str] | str,
) -> str:
    """Compile analytical queries using semantic business metrics and dimensions."""
    return query_semantic_layer(entity_name=entity_name, metric_names=metric_names, group_by_dims=group_by_dims)


@tool
def generate_diagram_tool(diagram_type: str = "er", domain: str | None = None) -> str:
    """Draw Mermaid ER diagrams or medallion lineage flowcharts."""
    return generate_diagram(diagram_type=diagram_type, domain=domain)


@tool
def generate_etl_pipeline_tool(entity_name: str, layer: str = "silver") -> str:
    """Generate PySpark and SparkSQL pipelines for Bronze, Silver, or Gold layers."""
    return generate_etl_pipeline(entity_name=entity_name, layer=layer)


@tool
def run_ci_tool(pyspark_code: str | None = None, sparksql_code: str | None = None) -> str:
    """Execute Ruff and SQLFluff CI quality gate and anti-pattern checks."""
    return run_ci(pyspark_code=pyspark_code, sparksql_code=sparksql_code)


@tool
def submit_gitops_pr_tool(
    product_name: str,
    pyspark_code: str,
    sparksql_code: str,
    diagram_md: str = "",
) -> str:
    """Run CI and create GitHub feature branch, commit, and Pull Request."""
    return submit_gitops_pr(
        product_name=product_name,
        pyspark_code=pyspark_code,
        sparksql_code=sparksql_code,
        diagram_md=diagram_md,
    )


LANGCHAIN_TOOLS = [
    inspect_unity_catalog_tool,
    load_semantic_models_tool,
    query_semantic_layer_tool,
    generate_diagram_tool,
    generate_etl_pipeline_tool,
    run_ci_tool,
    submit_gitops_pr_tool,
]


def get_langchain_tools() -> list[Any]:
    """Retrieve LangChain tools for agent model binding."""
    return list(LANGCHAIN_TOOLS)

