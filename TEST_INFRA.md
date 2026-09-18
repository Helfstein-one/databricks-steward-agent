# Databricks Steward Agent — Test Infrastructure & Methodology (TEST_INFRA)

## 1. Overview & Test Philosophy

The Databricks Steward Agent test suite is engineered as an exhaustive, requirement-driven, opaque-box testing framework. It validates that the solution functions reliably as an enterprise-grade GenAI data stewardship, semantic modeling, ETL engineering, and GitOps automation platform integrated with Open WebUI.

The testing architecture follows a strict **4-Tier Progressive Verification Model**:
- **Tier 1 (Foundational Unit Tests)**: Verifies configurations, Pydantic v2 data models, YAML parsing, schema validation, and native diagram generation.
- **Tier 2 (Component Integration & Logic with Mocks)**: Verifies Databricks SDK client connectivity, Unity Catalog introspection, offline demo mock fallbacks, PySpark/SparkSQL ETL generation, Delta Lake optimization templates, CI quality gate execution (Ruff, SQLFluff, Anti-pattern detection), and GitOps operations (branching, commits, PyGithub PR).
- **Tier 3 (Workflow Integration & State Graphs)**: Verifies LangGraph agent state graph orchestration, tool invocation pipelines, and Open WebUI Pipe interface with configurable Valves.
- **Tier 4 (Real-World End-to-End Scenarios)**: Verifies complete lifecycle data stewardship scenarios from user prompt to Unity Catalog inspection, semantic modeling, Mermaid ERD/lineage visualization, Medallion ETL code generation, CI quality gate validation, and automated GitHub PR creation.

---

## 2. Feature Inventory Mapping to Test Tiers

The table below maps all 26 features identified in `PROJECT.md` to their corresponding test tier and test file:

| Feature ID | Feature Name | Test Tier | Test Module | Verification Scope |
|:---|:---|:---|:---|:---|
| **F-01** | Settings & Environment Configuration | Tier 1 | `tests/test_config.py` | Pydantic Settings loading, defaults, environment variable overrides, paths. |
| **F-02** | Databricks Workspace Client Wrapper | Tier 2 | `tests/test_databricks_client.py` | Connection verification, URL normalization, directory creation, base64 workspace file import. |
| **F-03** | Unity Catalog Introspection | Tier 2 | `tests/test_databricks_client.py` | Discovery of catalogs, schemas, tables, columns, primary/foreign keys via Databricks SDK. |
| **F-04** | Unity Catalog Offline Demo Mode | Tier 2 | `tests/test_databricks_client.py` | Mock entity fallback (`bronze_raw_transactions`, `silver_transactions`, `gold_sales_kpis`) when credentials are missing/unreachable. |
| **F-05** | Declarative Semantic Data Models | Tier 1 | `tests/test_semantic_layer.py` | Pydantic v2 validation of `DimensionModel`, `MetricModel`, `RelationshipModel`, `EntityModel`, `SemanticDomainModel`. |
| **F-06** | Multi-File Semantic Registry | Tier 1 | `tests/test_semantic_layer.py` | Dynamic loading of `*.yaml` from directory, schema validation, entity and metric indexing, caching. |
| **F-07** | Business Synonyms Resolver | Tier 1 | `tests/test_semantic_layer.py` | Mapping natural language business terms to physical tables and columns. |
| **F-08** | Graph Join Resolver | Tier 2 | `tests/test_semantic_layer.py` | Multi-table relationship graph, shortest-path join resolution across entities. |
| **F-09** | SparkSQL Semantic Query Compiler | Tier 2 | `tests/test_semantic_layer.py` | Compiling metrics and dimensions into SparkSQL with `NULLIF(..., 0)` division safety and `GROUP BY`. |
| **F-10** | Sample Semantic Models | Tier 1 | `tests/test_semantic_layer.py` | Loading and validation of `corporate_credit.yaml` and `sales_lakehouse.yaml`. |
| **F-11** | Mermaid erDiagram Generator | Tier 1 | `tests/test_mermaid.py` | Native Mermaid `erDiagram` with Crow's foot cardinality (`||--o{`, `||--||`, `}o--||`), attributes, PK/FK markers. |
| **F-12** | Mermaid Medallion Lineage Flow | Tier 1 | `tests/test_mermaid.py` | `graph LR` flowchart partitioned into Bronze, Silver, and Gold subgraphs. |
| **F-13** | Modular Medallion PySpark Generator | Tier 2 | `tests/test_etl_generator.py` | Generating Bronze (raw ingest), Silver (clean/dedup), Gold (KPI aggregate) PySpark pipelines. |
| **F-14** | SparkSQL Script Generator | Tier 2 | `tests/test_etl_generator.py` | Generating idempotent DDL/DML, Delta Lake table creation, merge scripts. |
| **F-15** | Delta Lake Optimization Templates | Tier 2 | `tests/test_etl_generator.py` | Generation of `OPTIMIZE`, `ZORDER BY`, and `VACUUM` statements with configurable retention. |
| **F-16** | Ruff Python Linter Wrapper | Tier 2 | `tests/test_ci_pipeline.py` | Programmatic invocation of Ruff check & format on generated PySpark code. |
| **F-17** | SQLFluff SparkSQL Linter Wrapper | Tier 2 | `tests/test_ci_pipeline.py` | Programmatic linting of SparkSQL code with `sparksql` dialect and violation parsing. |
| **F-18** | Data Anti-Pattern Detector | Tier 2 | `tests/test_ci_pipeline.py` | AST/regex detection of dangerous `.collect()`, accidental cross-joins, and missing partition filters. |
| **F-19** | Structured CI Quality Gate Report | Tier 2 | `tests/test_ci_pipeline.py` | Pydantic CIReport schema, overall approval status, diagnostics, Markdown summary. |
| **F-20** | Git Feature Branching & Commits | Tier 2 | `tests/test_gitops.py` | Dynamic branch creation `feature/data-product-<name>`, Conventional Commits formatting. |
| **F-21** | PyGithub Automated Pull Request | Tier 2 | `tests/test_gitops.py` | PyGithub PR creation embedding data product summary, Mermaid diagrams, and CI quality report. |
| **F-22** | GitOps Dry-Run / Mock Support | Tier 2 | `tests/test_gitops.py` | Safe dry-run and mock execution modes for offline and CI verification. |
| **F-23** | LangGraph Agent State & Tools | Tier 3 | `tests/test_agent_graph.py` | AgentState schema, state transitions, tool execution across catalog, semantic, ETL, CI, and GitOps nodes. |
| **F-24** | Open WebUI Pipe Script | Tier 3 | `tests/test_agent_graph.py` | `open_webui_pipe.py` exposing `Pipe` class, `Valves`, streaming generator, and error handling. |
| **F-25** | Comprehensive Unit Test Suite | Tiers 1-4 | `tests/*` | Complete test coverage executing cleanly under `.venv/bin/pytest -v tests/`. |
| **F-26** | Documentation & Delivery | Tier 4 | `tests/test_e2e_scenarios.py` | Verification of end-to-end user workflows and architectural integrity. |

---

## 3. Test Runner & Execution Semantics

The test suite is built on **PyTest 9.1+** and **pytest-asyncio**, executed within the provisioned `.venv` environment.

### Command Line Invocation
```bash
# Execute entire test suite with verbose output
.venv/bin/pytest -v tests/

# Execute a specific tier or test file
.venv/bin/pytest -v tests/test_config.py
.venv/bin/pytest -v tests/test_semantic_layer.py
.venv/bin/pytest -v tests/test_ci_pipeline.py
.venv/bin/pytest -v tests/test_e2e_scenarios.py

# Execute with keyword filter
.venv/bin/pytest -v -k "anti_pattern or gitops"

# Execute with short traceback for quick feedback
.venv/bin/pytest -v --tb=short tests/
```

### Test Isolation & Non-Pollution Guarantees
1. **No External Network Calls**: All external network interactions (Databricks SDK REST API, PyGithub API, OpenAI/local LLM HTTP endpoints) are cleanly mocked via `unittest.mock` and custom fixture fakes.
2. **Deterministic Temporary Storage**: All file system operations (temporary Python files for Ruff, temporary SQL files for SQLFluff, Git operations) utilize PyTest's built-in `tmp_path` fixture, guaranteeing zero pollution of the host workspace.
3. **Environment Isolation**: Tests utilizing environment variables use `monkeypatch` to set and restore environment state without side effects on subsequent tests.

---

## 4. Test Fixtures & Mock Architecture (`tests/conftest.py`)

`tests/conftest.py` provides centralized, reusable fixtures across all test suites:

1. **`mock_workspace_client`**:
   - Mocks `databricks.sdk.WorkspaceClient`.
   - Simulates `catalogs.list()`, `schemas.list()`, `tables.list()`, and `workspace.import_()`.
   - Provides realistic catalog, schema, table, and column metadata matching Unity Catalog specifications.

2. **`mock_pygithub`**:
   - Mocks `github.Github` and `github.Auth`.
   - Simulates `get_repo()`, `create_pull()`, PR branches, commit objects, and HTML URLs.

3. **`sample_semantic_domain` & `sample_yaml_dir`**:
   - Provides valid `SemanticDomainModel` instances with entities (`counterparts`, `facilities`), dimensions (`sector`, `rating`), metrics (`total_exposure`, `overdue_ratio`), and relationships.
   - Populates temporary directories with valid YAML models (`corporate_credit.yaml`, `sales_lakehouse.yaml`) and invalid YAML models for negative testing.

4. **`sample_code_snippets`**:
   - **Clean PySpark Code**: Properly structured Medallion functions with explicit schemas and Delta Lake operations.
   - **Flawed PySpark Code with Anti-Patterns**: Unbounded `.collect()`, `toPandas()` without limit, Cartesian `crossJoin()`.
   - **Syntax Error PySpark Code**: Python code with unclosed parentheses or malformed statements.
   - **Clean SparkSQL Code**: Formatted SQL with valid syntax and `NULLIF(..., 0)`.
   - **Flawed SparkSQL Code**: SQL with Cartesian products (`FROM t1, t2`), syntax errors, or divide-by-zero vulnerabilities.

---

## 5. Real-World Application Scenarios (Tier 4)

Tier 4 (`tests/test_e2e_scenarios.py`) simulates realistic end-to-end data stewardship lifecycles:

### Scenario 1: Conversational Lakehouse Data Product Lifecycle
1. **User Request**: User prompts Open WebUI to create a "Wholesale Credit Risk" data product.
2. **Catalog Introspection**: Steward Agent inspects Databricks Unity Catalog for existing credit tables.
3. **Semantic Modeling**: Agent loads/creates the semantic domain model with dimensions, metrics, and relationships.
4. **Mermaid Visualization**: Agent emits native Mermaid `erDiagram` and Medallion lineage diagram (`graph LR`) renderable in Open WebUI.
5. **ETL Code Generation**: Agent generates idempotent Bronze, Silver, and Gold PySpark and SparkSQL scripts.
6. **CI Quality Gate**: CI runner analyzes generated code with Ruff, SQLFluff, and anti-pattern detectors, generating an approved `CIReport`.
7. **GitOps Execution**: GitOps manager creates feature branch `feature/data-product-credit-risk`, commits code with Conventional Commit `feat(data-product): add credit risk pipeline`, and opens a PyGithub Pull Request embedding the diagrams and CI audit report.

### Scenario 2: Anti-Pattern Detection & Rejection Gate
1. **User Request**: User submits or requests a pipeline containing unsafe data engineering patterns (e.g. unbounded `.collect()` or accidental Cartesian join).
2. **CI Quality Gate**: CI runner evaluates the code, detects the anti-patterns, and issues a `REJECTED` report with line numbers and diagnostic details.
3. **Safety Enforcement**: GitOps workflow is strictly blocked; no branch is pushed and no Pull Request is opened.

### Scenario 3: Offline Demo Mode Resiliency
1. **Disconnected Environment**: Live Databricks credentials are not configured or the host is unreachable.
2. **Fallback Activation**: The agent gracefully switches to offline demo mode, utilizing `_build_mock_entities()` to provide instant catalog discovery and allow users to model, visualize, and test pipelines without live infrastructure.
