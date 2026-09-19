"""LangGraph agent tools exposing Databricks, Semantic, Mermaid, CI, and GitOps capabilities."""

from __future__ import annotations

import re
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

ENTITY_ALIAS_MAP: dict[str, str] = {
    # Customers / Clientes
    "customers": "customers",
    "customer": "customers",
    "cliente": "customers",
    "clientes": "customers",
    "compradores": "customers",
    "buyers": "customers",
    # Orders / Pedidos
    "orders": "orders",
    "order": "orders",
    "pedido": "orders",
    "pedidos": "orders",
    "sales_orders": "orders",
    # Order Items / Itens de Pedido
    "order_items": "order_items",
    "order-items": "order_items",
    "order items": "order_items",
    "order_item": "order_items",
    "order item": "order_items",
    "item de pedido": "order_items",
    "itens de pedido": "order_items",
    "itens do pedido": "order_items",
    # Products / Produtos
    "products": "products",
    "product": "products",
    "produto": "products",
    "produtos": "products",
    # Facilities / Linhas de Crédito
    "facilities": "facilities",
    "facility": "facilities",
    "credit_facilities": "facilities",
    "credit facilities": "facilities",
    "linha de credito": "facilities",
    "linhas de credito": "facilities",
    "linha de crédito": "facilities",
    "linhas de crédito": "facilities",
    "facilidades": "facilities",
    "contracts": "facilities",
    # Borrowers / Tomadores
    "borrowers": "borrowers",
    "borrower": "borrowers",
    "tomador": "borrowers",
    "tomadores": "borrowers",
    # Impairments / Provisões
    "impairments": "impairments",
    "impairment": "impairments",
    "provisao": "impairments",
    "provisão": "impairments",
    "provisoes": "impairments",
    "provisões": "impairments",
    "perdas": "impairments",
    # Lakehouse Medallion tables
    "medallion_bronze_transactions": "medallion_bronze_transactions",
    "bronze_transactions": "medallion_bronze_transactions",
    "bronze_raw_transactions": "medallion_bronze_transactions",
    "transacoes_bronze": "medallion_bronze_transactions",
    "transações_bronze": "medallion_bronze_transactions",
    "transações bronze": "medallion_bronze_transactions",
    "transacoes bronze": "medallion_bronze_transactions",
    "raw_transactions": "medallion_bronze_transactions",
    "bronze": "medallion_bronze_transactions",
    "medallion_silver_transactions": "medallion_silver_transactions",
    "silver_transactions": "medallion_silver_transactions",
    "transacoes_silver": "medallion_silver_transactions",
    "transações_silver": "medallion_silver_transactions",
    "transações silver": "medallion_silver_transactions",
    "transacoes silver": "medallion_silver_transactions",
    "transactions": "medallion_silver_transactions",
    "transações": "medallion_silver_transactions",
    "transacoes": "medallion_silver_transactions",
    "transação": "medallion_silver_transactions",
    "transacao": "medallion_silver_transactions",
    "silver": "medallion_silver_transactions",
    "medallion_gold_sales_kpis": "medallion_gold_sales_kpis",
    "gold_sales_kpis": "medallion_gold_sales_kpis",
    "gold_sales": "medallion_gold_sales_kpis",
    "sales_kpis": "medallion_gold_sales_kpis",
    "kpis_vendas": "medallion_gold_sales_kpis",
    "kpis de vendas": "medallion_gold_sales_kpis",
    "vendas_gold": "medallion_gold_sales_kpis",
    "vendas gold": "medallion_gold_sales_kpis",
    "medallion_gold_customer_kpis": "medallion_gold_customer_kpis",
    "gold_customer_kpis": "medallion_gold_customer_kpis",
    "gold_customers": "medallion_gold_customer_kpis",
    "customer_kpis": "medallion_gold_customer_kpis",
    "kpis_clientes": "medallion_gold_customer_kpis",
    "kpis de clientes": "medallion_gold_customer_kpis",
    "clientes_gold": "medallion_gold_customer_kpis",
    "clientes gold": "medallion_gold_customer_kpis",
}

DOMAIN_ALIAS_MAP: dict[str, str] = {
    "sales": "sales_lakehouse",
    "vendas": "sales_lakehouse",
    "retail": "sales_lakehouse",
    "varejo": "sales_lakehouse",
    "sales_lakehouse": "sales_lakehouse",
    "credit": "corporate_credit",
    "credito": "corporate_credit",
    "crédito": "corporate_credit",
    "corporate_credit": "corporate_credit",
    "corporate": "corporate_credit",
    "wholesale": "corporate_credit",
    "medallion": "databricks_medallion",
    "medalhão": "databricks_medallion",
    "medalhao": "databricks_medallion",
    "databricks": "databricks_medallion",
    "databricks_medallion": "databricks_medallion",
    "lakehouse": "databricks_medallion",
    "workspace": "databricks_medallion",
}


def inspect_unity_catalog(catalog: str | None = None, schema: str | None = None) -> str:
    """Introspect Databricks Unity Catalog tables, columns, and constraints."""
    from src.databricks.client import DatabricksCEClient

    client = DatabricksCEClient()
    cat_to_use = catalog or settings.databricks_default_catalog or "workspace"
    sch_to_use = schema or settings.databricks_default_schema or "default"
    mode_str = (
        "🟢 Conectado ao Databricks Real via SDK"
        if client.is_configured()
        else "🟡 Modo Demonstração Offline (defina DATABRICKS_HOST e DATABRICKS_TOKEN no .env para conectar ao seu workspace)"
    )

    entities = introspect_catalog(catalog=cat_to_use, schema=sch_to_use, client=client)
    actual_cat = entities[0].catalog if entities else cat_to_use
    actual_sch = entities[0].schema_name if entities else sch_to_use
    lines = [
        f"Discovered {len(entities)} entities in {actual_cat}.{actual_sch} (*{mode_str}*):",
    ]
    for ent in entities:
        col_summary = ", ".join([f"`{c.name}` ({c.type})" for c in ent.columns[:6]])
        lines.append(f"- **`{ent.name}`** (Camada: `{ent.layer or 'unassigned'}`): {col_summary}")

    lines.append(
        "\n💡 *Nota: Estas são as tabelas físicas inspecionadas no Unity Catalog. "
        "Para ver os modelos de dados e Data Products da Camada Semântica (ex: `medallion_silver_transactions`, `medallion_gold_sales_kpis`), digite `2` ou peça a modelagem da tabela.*"
    )
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


def format_entity_modeling(entity_name: str) -> str:
    """Format detailed data modeling, dimensions, metrics, and ER diagram for a specific entity."""
    reg = SemanticRegistry(settings.semantic_models_path)
    clean_name = (entity_name or "").strip().lower()

    # Normalize prefixes like "tabela ", "table ", "entidade ", "entity ", "da tabela ", "de "
    clean_name = re.sub(
        r"^(a\s+|o\s+|da\s+|do\s+|de\s+)?(tabela|table|entidade|entity)\s+(da\s+|do\s+|de\s+)?",
        "",
        clean_name,
    ).strip()
    clean_name = re.sub(r"^(de\s+|da\s+|do\s+)", "", clean_name).strip()

    target_name = ENTITY_ALIAS_MAP.get(clean_name, clean_name)
    entity = reg.get_entity(target_name)

    if not entity:
        # Search for any known entity alias within the string
        sorted_aliases = sorted(ENTITY_ALIAS_MAP.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            pattern = r"(?:\b|_)" + re.escape(alias) + r"(?:\b|_)"
            if re.search(pattern, clean_name):
                matched_target = ENTITY_ALIAS_MAP[alias]
                e_found = reg.get_entity(matched_target)
                if e_found:
                    entity = e_found
                    target_name = matched_target
                    break

    if not entity:
        if target_name.endswith("s"):
            entity = reg.get_entity(target_name[:-1])
        else:
            entity = reg.get_entity(target_name + "s")

    if not entity:
        from src.databricks.introspector import introspect_catalog

        real_ents = {e.name.lower(): e for e in introspect_catalog()}
        e_obj = real_ents.get(target_name.lower())
        if e_obj:
            cols = "\n".join(
                [
                    f"| `{c.name}` | `{c.type.upper()}` | {c.description or 'Coluna de dados'} |"
                    for c in e_obj.columns
                ]
            )
            return (
                f"### 📐 Modelagem da Tabela Lakehouse: `{e_obj.name}`\n\n"
                f"- **Tabela Física:** `{e_obj.catalog}.{e_obj.schema_name}.{e_obj.name}`\n"
                f"- **Camada Medalhão:** `{e_obj.layer or 'unassigned'}`\n"
                f"- **Chave Primária:** `{e_obj.primary_key or 'Não definida explicitamente'}`\n\n"
                f"#### 📋 Colunas Físicas do Unity Catalog:\n"
                f"| Coluna | Tipo | Descrição |\n"
                f"|---|---|---|\n"
                f"{cols}\n\n"
                f"💡 **Próximos passos:**\n"
                f"- Peça *'gerar pipeline etl para {e_obj.name}'* para criar a ingestão PySpark / SparkSQL.\n"
                f"- Peça *'validar código na esteira de ci'* para testar as regras de engenharia de dados."
            )
        return f"Entidade ou tabela '{entity_name}' não encontrada no catálogo nem na camada semântica."

    domain_name = ""
    for d_key, d_obj in reg.domains.items():
        if any(e.name == entity.name for e in d_obj.entities):
            domain_name = d_key
            break

    dim_rows = []
    for d in entity.dimensions:
        is_pk = " **[PK]**" if d.name == entity.primary_key else ""
        if d.description:
            desc = d.description
        elif is_pk:
            desc = "Chave primária única da entidade"
        elif d.name.endswith("_id"):
            desc = f"Identificador de relacionamento ({d.name})"
        elif d.name.endswith("_date") or d.name.endswith("_at"):
            desc = f"Data / timestamp temporal ({d.name})"
        elif d.name in (
            "status",
            "segment",
            "country",
            "channel",
            "category",
            "currency",
            "product_type",
        ):
            desc = f"Atributo categórico de negócio ({d.name})"
        else:
            desc = f"Coluna dimensional {d.name}"
        dim_rows.append(f"| `{d.name}` | `{d.type.upper()}` |{is_pk} {desc} |")
    dim_table = "\n".join(dim_rows)

    metric_lines = []
    if entity.metrics:
        for m in entity.metrics:
            m_desc = f" — {m.description}" if m.description else ""
            metric_lines.append(f"- **`{m.name}`**: `{m.sql}`{m_desc}")
        metrics_section = "\n".join(metric_lines)
    else:
        metrics_section = "_Nenhuma métrica agregada registrada diretamente nesta entidade._"

    all_rels = reg.list_relationships(domain_name)
    relevant_rels = [
        r for r in all_rels if r.from_entity == entity.name or r.to_entity == entity.name
    ]
    if relevant_rels:
        rel_lines = [
            f"- `{r.from_entity}.{r.from_column}` → `{r.to_entity}.{r.to_column}` (`{r.type}`)"
            for r in relevant_rels
        ]
        rels_section = "\n".join(rel_lines)
    else:
        rels_section = "_Entidade sem chaves estrangeiras diretas registradas._"

    related_names = set(
        [entity.name]
        + [r.from_entity for r in relevant_rels]
        + [r.to_entity for r in relevant_rels]
    )
    related_entities = [reg.get_entity(n) for n in related_names if reg.get_entity(n)]
    diag = generate_er_diagram(related_entities, relevant_rels)

    domain_label = f"`{domain_name}`" if domain_name else "Lakehouse"

    return (
        f"### 📐 Modelagem de Dados: `{entity.name}`\n\n"
        f"- **Tabela Lakehouse:** `{entity.table_name}`\n"
        f"- **Domínio Semântico:** {domain_label}\n"
        f"- **Chave Primária (PK):** `{entity.primary_key}`\n\n"
        f"#### 📋 Colunas e Dimensões:\n"
        f"| Coluna | Tipo | Descrição |\n"
        f"|---|---|---|\n"
        f"{dim_table}\n\n"
        f"#### 📊 Métricas Analíticas de Negócio:\n"
        f"{metrics_section}\n\n"
        f"#### 🔗 Relacionamentos (Joins):\n"
        f"{rels_section}\n\n"
        f"#### 📊 Diagrama de Entidade-Relacionamento:\n"
        f"```mermaid\n{diag}\n```\n\n"
        f"💡 **Próximos passos com `{entity.name}`:**\n"
        f"- Peça *'gerar pipeline etl para {entity.name}'* para criar a ingestão PySpark / SparkSQL.\n"
        f"- Peça *'compilar query de métricas para {entity.name}'* para gerar consultas analíticas com joins automáticos.\n"
        f"- Peça *'validar código na esteira de ci'* para testar as regras de engenharia de dados."
    )


def generate_diagram(diagram_type: str = "er", domain: str | None = None) -> str:
    """Generate Mermaid erDiagram (Crow's foot) or Medallion lineage flowchart."""
    reg = SemanticRegistry(settings.semantic_models_path)
    dt = (diagram_type or "er").strip().lower()

    # Determine if Lineage flowchart or ER diagram
    is_lineage = any(k in dt for k in ("lineage", "linhagem", "fluxo", "medallion", "flowchart"))

    entities = []
    relationships = []
    domain_title = "Camada Semântica Lakehouse"

    # If domain specified, resolve domain alias or entity focus
    if domain:
        clean_dom = domain.strip().lower()
        resolved_domain = DOMAIN_ALIAS_MAP.get(clean_dom, clean_dom)
        dom = reg.get_domain(resolved_domain)
        if dom:
            entities = dom.entities
            relationships = dom.relationships
            domain_title = f"Domínio Semântico: `{resolved_domain}`"
        else:
            target_ent = ENTITY_ALIAS_MAP.get(clean_dom, clean_dom)
            ent = reg.get_entity(target_ent)
            if ent:
                d_name = ""
                for d_k, d_v in reg.domains.items():
                    if any(e.name == ent.name for e in d_v.entities):
                        d_name = d_k
                        break
                all_rels = reg.list_relationships(d_name)
                relationships = [
                    r for r in all_rels if r.from_entity == ent.name or r.to_entity == ent.name
                ]
                related_names = set(
                    [ent.name]
                    + [r.from_entity for r in relationships]
                    + [r.to_entity for r in relationships]
                )
                entities = [reg.get_entity(n) for n in related_names if reg.get_entity(n)]
                domain_title = f"Entidade: `{ent.name}` ({d_name or 'Lakehouse'})"

    if not entities:
        med_dom = reg.get_domain("databricks_medallion")
        if med_dom:
            entities = med_dom.entities
            relationships = med_dom.relationships
            domain_title = "Arquitetura Medalhão Databricks (`workspace.default`)"
        else:
            entities = reg.list_entities()
            relationships = reg.list_relationships()

    if not entities:
        from src.databricks.introspector import _build_mock_entities

        entities = _build_mock_entities()
        relationships = []

    if is_lineage:
        if not entities or len(entities) == len(reg.list_entities()):
            med_dom = reg.get_domain("databricks_medallion")
            if med_dom:
                entities = med_dom.entities
        mermaid_code = generate_lineage_diagram(entities)
        return (
            f"### 🌊 Fluxo de Linhagem Medalhão (Bronze → Silver → Gold)\n\n"
            f"```mermaid\n{mermaid_code}\n```"
        )

    # Format relationships table for ER diagrams
    rel_rows = []
    cardinality_pt = {
        "many_to_one": "N:1 (Muitos para Um)",
        "one_to_many": "1:N (Um para Muitos)",
        "one_to_one": "1:1 (Um para Um)",
        "many_to_many": "N:M (Muitos para Muitos)",
    }
    for r in relationships:
        c_label = cardinality_pt.get(r.type.lower(), r.type)
        rel_rows.append(
            f"| `{r.from_entity}` | `{r.from_column}` | {c_label} | `{r.to_entity}` | `{r.to_column}` |"
        )

    table_section = ""
    if rel_rows:
        table_section = (
            "#### 🔗 Relações e Chaves Estrangeiras (Camada Semântica):\n"
            "| Tabela Origem | Chave Origem (FK) | Relacionamento | Tabela Destino | Chave Destino (PK) |\n"
            "|---|---|---|---|---|\n" + "\n".join(rel_rows) + "\n\n"
        )

    mermaid_code = generate_er_diagram(entities, relationships)
    return (
        f"### 📊 Modelo de Entidade-Relacionamento ({domain_title})\n\n"
        f"{table_section}"
        f"#### 📐 Diagrama Visual Mermaid (ERD):\n"
        f"```mermaid\n{mermaid_code}\n```"
    )


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
    {
        "name": "format_entity_modeling",
        "description": "Format detailed modeling, dimensions, metrics, and ER diagram for a table or entity.",
        "func": format_entity_modeling,
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
    return query_semantic_layer(
        entity_name=entity_name, metric_names=metric_names, group_by_dims=group_by_dims
    )


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


@tool
def inspect_entity_modeling_tool(entity_name: str) -> str:
    """Inspect detailed data modeling, dimensions, metrics, and ER diagram for a specific table or entity."""
    return format_entity_modeling(entity_name)


LANGCHAIN_TOOLS = [
    inspect_unity_catalog_tool,
    load_semantic_models_tool,
    query_semantic_layer_tool,
    generate_diagram_tool,
    generate_etl_pipeline_tool,
    run_ci_tool,
    submit_gitops_pr_tool,
    inspect_entity_modeling_tool,
]


def get_langchain_tools() -> list[Any]:
    """Retrieve LangChain tools for agent model binding."""
    return list(LANGCHAIN_TOOLS)
