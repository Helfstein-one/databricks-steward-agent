# Databricks Steward Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Databricks SDK](https://img.shields.io/badge/Databricks-SDK%200.28+-orange.svg)](https://docs.databricks.com/en/dev-tools/sdk-python.html)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Open WebUI](https://img.shields.io/badge/Open%20WebUI-Compatible-green.svg)](https://openwebui.com/)

**Databricks Steward Agent** is an end-to-end GenAI data stewardship, semantic modeling, PySpark ETL engineering, and GitOps automation solution designed for **Open WebUI** and local LLM runners (Ollama / vLLM via OpenAI-compatible endpoints) with **LangGraph** orchestration.

---

## 🏛️ Architectural Overview

```
                          ┌────────────────────────────┐
                          │   Open WebUI Chat UI       │
                          │   (open_webui_pipe.py)     │
                          └─────────────┬──────────────┘
                                        │
                                        ▼
                          ┌────────────────────────────┐
                          │     LangGraph Agent        │
                          │     (src/agent/graph.py)   │
                          └─────────────┬──────────────┘
                                        │
        ┌───────────────────┬───────────┴───────────┬───────────────────┐
        ▼                   ▼                       ▼                   ▼
┌──────────────┐    ┌──────────────┐        ┌──────────────┐    ┌──────────────┐
│  Databricks  │    │   Semantic   │        │   Mermaid    │    │  Medallion   │
│Unity Catalog │    │ Declarative  │        │ Visualizer   │    │ETL Generator │
│ (Introspect) │    │  (Registry)  │        │ (ERD/Lineage)│    │(Bronze/Silver│
└──────────────┘    └───────┬──────┘        └──────────────┘    │    /Gold)    │
                            │                                   └───────┬──────┘
                            ▼                                           │
                    ┌──────────────┐                                    ▼
                    │ SparkSQL     │                            ┌──────────────┐
                    │ Compiler     │                            │CI Quality    │
                    │(NULLIF Safety│                            │Gate (Ruff,   │
                    └──────────────┘                            │SQLFluff,Anti)│
                                                                └───────┬──────┘
                                                                        │
                                                                        ▼
                                                                ┌──────────────┐
                                                                │Automated     │
                                                                │GitOps Engine │
                                                                │(GitHub PR)   │
                                                                └──────────────┘
```

---

## 🚀 Key Capabilities

1. **Open WebUI Pipe & Local LLMs**: Standalone `open_webui_pipe.py` script featuring configurable administrative `Valves` connecting to local models (Ollama, vLLM) or OpenAI endpoints.
2. **Databricks Unity Catalog Introspection**: Inspects catalogs, schemas, tables, and constraints via Databricks SDK (`WorkspaceClient`), featuring an automatic offline demo mock fallback mode.
3. **Declarative Semantic Modeling**: Pydantic v2 schemas defining business domains, entities, dimensions, metrics, and relationships with YAML ontologies in `configs/semantic_models/`.
4. **Graph Join Resolver & Safe SparkSQL Compiler**: Shortest-path join resolution using BFS on entity relationship graphs, compiling analytical queries with automatic `NULLIF(..., 0)` division-by-zero protection.
5. **Interactive Mermaid.js Visualizations**: Native generation of strict Crow's foot `erDiagram` models and Medallion layer lineage flowcharts (`graph LR`) directly renderable in chat interfaces.
6. **Modular Medallion ETL Generation**: Generates production-ready, explicitly typed PySpark and SparkSQL modules across Bronze (raw ingestion), Silver (deduplication & cleansing), and Gold (business KPIs) layers with Delta Lake optimization templates (`OPTIMIZE`, `ZORDER BY`, `VACUUM`).
7. **Data Best Practices CI Quality Gate**: Automated validation runner executing:
   - Ruff linting for Python/PySpark.
   - SQLFluff with `sparksql` dialect for SQL queries.
   - Static analysis detecting big data anti-patterns (unbounded `.collect()`, accidental cross-joins, missing partitions, unbounded `.toPandas()`).
8. **Automated GitOps with GitHub PR**: Creates feature branches (`feature/data-product-<name>`), stages files, crafts Conventional Commits, and opens rich GitHub Pull Requests embedding data product summaries, Mermaid diagrams, and CI validation reports.

---

## 📦 Project Layout

```
├── configs/
│   └── semantic_models/
│       ├── corporate_credit.yaml       # Wholesale Banking semantic ontology
│       └── sales_lakehouse.yaml        # E-Commerce retail sales ontology
├── open_webui_pipe.py                  # Open WebUI Pipe integration script
├── src/
│   ├── config.py                       # Pydantic Settings & environment loader
│   ├── databricks/
│   │   ├── client.py                   # Databricks SDK WorkspaceClient wrapper
│   │   └── introspector.py             # Unity Catalog inspector & mock fallback
│   ├── semantic/
│   │   ├── models.py                   # Pydantic v2 domain, entity & metric models
│   │   ├── registry.py                 # Multi-file YAML registry & synonym resolver
│   │   └── compiler.py                 # Multi-table graph join & SparkSQL compiler
│   ├── visualizer/
│   │   └── mermaid.py                  # Crow's foot ERD & Medallion lineage generator
│   ├── etl/
│   │   ├── generator.py                # Bronze/Silver/Gold PySpark & SQL generator
│   │   └── templates.py                # Delta OPTIMIZE, ZORDER BY, VACUUM templates
│   ├── ci/
│   │   ├── anti_patterns.py            # Static AST rules (collect, cross-join, partitions)
│   │   ├── report.py                   # Pydantic CI report model & markdown table
│   │   └── runner.py                   # Programmatic Ruff, SQLFluff & CI runner
│   ├── gitops/
│   │   ├── git_client.py               # Feature branch, commit & push automation
│   │   └── github_pr.py                # PyGithub automated PR creation
│   └── agent/
│       ├── state.py                    # LangGraph AgentState TypedDict
│       ├── tools.py                    # Agent tools dispatching to subsystems
│       └── graph.py                    # LangGraph StateGraph coordination
├── tests/                              # Comprehensive test suite
├── .env.example                        # Template for environment configuration
├── pyproject.toml                      # Build config and tool definitions
└── README.md
```

---

## ⚙️ Configuration & Environment

Copy `.env.example` to `.env` and fill in your connection details:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `DATABRICKS_HOST` | Databricks workspace URL | `""` (runs in demo mock mode) |
| `DATABRICKS_TOKEN` | Databricks Personal Access Token (PAT) | `""` |
| `DATABRICKS_WAREHOUSE_ID` | Databricks SQL Warehouse ID | `""` |
| `LOCAL_LLM_BASE_URL` | Local LLM OpenAI API endpoint | `http://localhost:11434/v1` |
| `LOCAL_LLM_MODEL` | Local model name | `qwen2.5-coder:7b` |
| `GITHUB_TOKEN` | GitHub Personal Access Token | `""` (runs in PR simulation mode) |
| `GITHUB_REPOSITORY` | Target repository in `owner/repo` format | `""` |
| `SEMANTIC_MODELS_PATH` | Path to semantic models directory | `./configs/semantic_models` |

---

## 💻 Quickstart & Python Usage

### 1. Introspect Unity Catalog (Offline Demo or Live)
```python
from src.databricks.introspector import introspect_catalog

entities = introspect_catalog(catalog="main", schema="default")
for ent in entities:
    print(f"Discovered: {ent.name} (layer: {ent.layer})")
```

### 2. Query Semantic Layer
```python
from src.semantic.registry import SemanticRegistry
from src.semantic.compiler import SemanticQueryCompiler

registry = SemanticRegistry("configs/semantic_models")
compiler = SemanticQueryCompiler(registry)

sql = compiler.compile_query(
    entity_name="facilities",
    metric_names=["total_credit_limit", "utilization_rate"],
    group_by_dims=["product_type", "status"],
    filters=["status = 'ACTIVE'"],
    limit=10,
)
print(sql)
```

### 3. Generate Mermaid Diagrams
```python
from src.semantic.registry import SemanticRegistry
from src.visualizer.mermaid import generate_er_diagram, generate_lineage_diagram

reg = SemanticRegistry("configs/semantic_models")
domain = reg.get_domain("corporate_credit")

# Crow's foot ER diagram
erd = generate_er_diagram(domain.entities, domain.relationships)

# Medallion lineage flowchart
lineage = generate_lineage_diagram(domain.entities)
```

### 4. Generate Medallion ETL & Run CI Quality Gate
```python
from src.etl.generator import generate_medallion_pipeline
from src.ci.runner import run_ci_pipeline

# Generate Silver pipeline
pipeline = generate_medallion_pipeline(domain.entities[0], layer="silver")

# Run automated CI validation
report = run_ci_pipeline(
    pyspark_code=pipeline.pyspark_code,
    sparksql_code=pipeline.sparksql_code,
)
print("Approved:", report.is_approved)
print(report.summary_markdown)
```

### 5. Automated GitOps Pull Request
```python
from src.gitops.github_pr import create_data_product_pr

result = create_data_product_pr(
    product_name="credit-facilities-silver",
    files={
        "pipelines/silver_facilities.py": pipeline.pyspark_code,
        "pipelines/silver_facilities.sql": pipeline.sparksql_code,
    },
    ci_report=report,
    diagram_md=erd,
    dry_run=True,
)
print("PR URL:", result.pr_url)
```

---

## 🌐 Open WebUI Integration

1. In Open WebUI, navigate to **Admin Panel** > **Functions / Pipes**.
2. Click **Add New Pipe** and import or paste the contents of `open_webui_pipe.py`.
3. Configure your **Valves** (Databricks credentials, local LLM endpoints, GitHub repository).
4. Save and select the **Databricks Steward** model in your chat dropdown!

---

## 🧪 Testing & Verification

Run Ruff code quality checks and the test suite:

```bash
# Run Ruff linting
.venv/bin/ruff check src/

# Run PyTest test suite
.venv/bin/pytest -v tests/
```

---

## 📄 License
Apache-2.0
