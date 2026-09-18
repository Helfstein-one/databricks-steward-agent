"""Adversarial white-box fuzzing and edge-case verification for Databricks Steward Agent."""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from open_webui_pipe import Pipe
from src.ci.anti_patterns import DataAntiPatternDetector
from src.ci.runner import run_ci_pipeline
from src.etl.generator import generate_medallion_pipeline
from src.etl.templates import (
    build_delta_create_table,
    build_delta_merge_query,
    build_optimize_query,
    build_vacuum_query,
)
from src.semantic.compiler import (
    GraphJoinResolver,
    SemanticQueryCompiler,
    ensure_nullif_division_safety,
)
from src.semantic.models import (
    ColumnModel,
    DimensionModel,
    EntityModel,
    MetricModel,
    RelationshipModel,
    SemanticDomainModel,
)
from src.semantic.registry import SemanticRegistry
from src.visualizer.mermaid import (
    _sanitize_id,
    _sanitize_type,
    generate_er_diagram,
    generate_lineage_diagram,
)

# ==============================================================================
# SECTION 1: SparkSQL Compilation & Delta Lake Statement Fuzzing
# ==============================================================================


class TestSparkSQLAndDeltaFuzzing:
    """Adversarial stress-testing of SQL compilation, division safety, and Delta queries."""

    @pytest.mark.parametrize(
        ("expr", "expected_contains"),
        [
            ("a / b", "a / NULLIF(b, 0)"),
            ("a / b / c", "a / NULLIF(b, 0) / NULLIF(c, 0)"),
            ("SUM(x) / COUNT(DISTINCT y)", "SUM(x) / NULLIF(COUNT(DISTINCT y), 0)"),
            ("revenue / NULLIF(cost, 0)", "revenue / NULLIF(cost, 0)"),
            ("(a + b) / (c - d)", "(a + b) / NULLIF((c - d), 0)"),
            ("a / 0", "a / NULLIF(0, 0)"),
            ("SELECT 'http://api.com/' AS url", "SELECT 'http://api.com/' AS url"),
            ("-- comment with slash /\nSELECT col FROM tbl", "SELECT col FROM tbl"),
            ("/* block comment / */ SELECT col FROM tbl", "SELECT col FROM tbl"),
            (
                "COALESCE(val, 0) / (denom_a + denom_b)",
                "COALESCE(val, 0) / NULLIF((denom_a + denom_b), 0)",
            ),
            ("SELECT 100", "SELECT 100"),
        ],
    )
    def test_nullif_division_safety_fuzzing(self, expr: str, expected_contains: str) -> None:
        """Verify NULLIF wrapping across varied mathematical expressions and comments."""
        safe_sql = ensure_nullif_division_safety(expr)
        assert expected_contains in safe_sql

    def test_nullif_division_safety_unary_operator_limitation(self) -> None:
        """Adversarial observation: unary minus/plus after slash breaks parser lookahead.

        In 'a / -b', the tokenizer at depth 0 stops immediately upon seeing '-',
        leaving denom_chars empty and causing the expression to remain unwrapped.
        """
        unwrapped = ensure_nullif_division_safety("a / -b")
        assert unwrapped == "a / -b"

    def test_graph_join_resolver_disconnected_components(self) -> None:
        """Verify BFS join resolver gracefully handles disconnected subgraphs without crashing."""
        rels = [
            RelationshipModel(
                name="rel1",
                from_entity="orders",
                to_entity="customers",
                from_column="customer_id",
                to_column="id",
                type="many_to_one",
            ),
            RelationshipModel(
                name="rel2",
                from_entity="logs",
                to_entity="servers",
                from_column="server_id",
                to_column="id",
                type="many_to_one",
            ),
        ]
        resolver = GraphJoinResolver(rels)
        # Connected pair
        path = resolver.find_shortest_path("orders", "customers")
        assert path is not None
        assert len(path) == 1
        assert path[0][0] == "customers"

        # Disconnected pair (island nodes)
        disconnected_path = resolver.find_shortest_path("orders", "servers")
        assert disconnected_path is None

        # Unknown entities
        assert resolver.find_shortest_path("orders", "unknown_tbl") is None
        assert resolver.find_shortest_path("unknown_start", "orders") is None

        # Self join
        assert resolver.find_shortest_path("orders", "orders") == []

    def test_graph_join_resolver_cyclic_graph(self) -> None:
        """Verify BFS join resolver does not get stuck in infinite cycles."""
        rels = [
            RelationshipModel(
                name="rel_ab",
                from_entity="a",
                to_entity="b",
                from_column="b_id",
                to_column="id",
                type="many_to_one",
            ),
            RelationshipModel(
                name="rel_bc",
                from_entity="b",
                to_entity="c",
                from_column="c_id",
                to_column="id",
                type="many_to_one",
            ),
            RelationshipModel(
                name="rel_ca",
                from_entity="c",
                to_entity="a",
                from_column="a_id",
                to_column="id",
                type="many_to_one",
            ),
        ]
        resolver = GraphJoinResolver(rels)
        path = resolver.find_shortest_path("a", "c")
        assert path is not None
        # Shortest path is direct backward step or forward via b
        assert len(path) in (1, 2)

    def test_semantic_compiler_adversarial_parameters(self) -> None:
        """Verify query compiler rejects invalid requests and compiles edge-case expressions."""
        reg = SemanticRegistry()
        ent = EntityModel(
            name="transactions",
            table_name="main.finance.transactions",
            primary_key="tx_id",
            columns=[
                ColumnModel(name="tx_id", type="string"),
                ColumnModel(name="user_id", type="string"),
                ColumnModel(name="amount", type="double"),
            ],
            dimensions=[
                DimensionModel(name="user_id", column="user_id"),
            ],
            metrics=[
                MetricModel(
                    name="avg_amount",
                    sql="SUM(amount) / COUNT(*)",
                    type="average",
                ),
            ],
        )
        dom = SemanticDomainModel(domain="finance", entities=[ent])
        reg.register_domain(dom)
        compiler = SemanticQueryCompiler(reg)

        # 1. Error on empty request
        with pytest.raises(ValueError, match="At least one metric or dimension"):
            compiler.compile_query("transactions", metric_names=[], group_by_dims=[])

        # 2. Error on missing entity
        with pytest.raises(ValueError, match="Entity 'non_existent' not found"):
            compiler.compile_query("non_existent", metric_names=["avg_amount"])

        # 3. Valid query compilation with NULLIF and explicit aliases
        sql = compiler.compile_query(
            "transactions",
            metric_names=["avg_amount"],
            group_by_dims=["user_id"],
            filters=["transactions.amount > 0", "transactions.user_id IS NOT NULL"],
            order_by=["avg_amount DESC"],
            limit=10,
        )
        assert "SELECT" in sql
        assert "NULLIF(COUNT(*), 0) AS avg_amount" in sql
        assert "FROM main.finance.transactions AS transactions" in sql
        assert "WHERE transactions.amount > 0 AND transactions.user_id IS NOT NULL" in sql
        assert "GROUP BY transactions.user_id" in sql
        assert "ORDER BY avg_amount DESC" in sql
        assert "LIMIT 10" in sql

    def test_delta_templates_fuzzing(self) -> None:
        """Verify Delta Lake query template generation under varied edge-case arguments."""
        # 1. build_optimize_query
        opt_no_zorder = build_optimize_query("my_tbl")
        assert opt_no_zorder == "OPTIMIZE my_tbl;"

        opt_with_zorder = build_optimize_query("my_tbl", zorder_columns=["col_a", "col_b"])
        assert opt_with_zorder == "OPTIMIZE my_tbl ZORDER BY (col_a, col_b);"

        opt_with_filter = build_optimize_query(
            "my_tbl",
            zorder_columns=["date_col"],
            where_clause="date_col >= '2026-01-01'",
        )
        assert "WHERE date_col >= '2026-01-01'" in opt_with_filter
        assert "ZORDER BY (date_col)" in opt_with_filter

        # 2. build_vacuum_query
        vac_default = build_vacuum_query("my_tbl")
        assert vac_default == "VACUUM my_tbl RETAIN 168 HOURS;"
        vac_custom = build_vacuum_query("my_tbl", retention_hours=72)
        assert vac_custom == "VACUUM my_tbl RETAIN 72 HOURS;"

        # 3. build_delta_create_table with special characters and comments
        cols = [
            {"name": "id", "type": "STRING"},
            {"name": "amount_cents", "type": "BIGINT"},
        ]
        create_sql = build_delta_create_table(
            table_name="catalog.schema.table",
            columns=cols,
            partition_by=["id"],
            comment="Table with single quote: it's safe",
        )
        assert "CREATE TABLE IF NOT EXISTS catalog.schema.table" in create_sql
        assert "`id` STRING" in create_sql
        assert "PARTITIONED BY (`id`)" in create_sql
        assert "COMMENT 'Table with single quote: it''s safe'" in create_sql

        # 4. build_delta_merge_query with wildcard fallback
        merge_sql = build_delta_merge_query(
            target_table="gold_tbl",
            source_table="silver_tbl",
            join_keys=["pk_id"],
        )
        assert "MERGE INTO gold_tbl AS target" in merge_sql
        assert "target.`pk_id` = source.`pk_id`" in merge_sql
        assert "WHEN MATCHED THEN UPDATE SET *" in merge_sql
        assert "WHEN NOT MATCHED THEN INSERT *" in merge_sql

    def test_etl_generator_layer_robustness(self) -> None:
        """Verify Medallion ETL code generation handles sparse entity definitions."""
        sparse_entity = EntityModel(
            name="sparse_events",
            table_name="raw_sparse_events",
            columns=[],
        )
        # Bronze generation with empty columns
        bronze_pipe = generate_medallion_pipeline(sparse_entity, layer="bronze")
        assert bronze_pipe.layer == "bronze"
        assert "run_bronze_pipeline" in bronze_pipe.pyspark_code
        assert "COPY INTO raw_sparse_events" in bronze_pipe.sparksql_code

        # Silver generation
        silver_pipe = generate_medallion_pipeline(sparse_entity, layer="silver")
        assert silver_pipe.layer == "silver"
        assert "run_silver_pipeline" in silver_pipe.pyspark_code

        # Gold generation
        gold_pipe = generate_medallion_pipeline(sparse_entity, layer="gold")
        assert gold_pipe.layer == "gold"
        assert "run_gold_pipeline" in gold_pipe.pyspark_code

        # Unsupported layer rejection
        with pytest.raises(ValueError, match="Unsupported layer"):
            generate_medallion_pipeline(sparse_entity, layer="platinum")


# ==============================================================================
# SECTION 2: Mermaid Diagram Generation Syntax & Fuzzing
# ==============================================================================


class TestMermaidDiagramFuzzing:
    """Stress-testing Mermaid erDiagram and Medallion lineage flowchart generation."""

    @pytest.mark.parametrize(
        ("input_name", "expected_clean"),
        [
            ("sales-lakehouse", "sales_lakehouse"),
            ("Customer Orders", "Customer_Orders"),
            ("main.catalog.tbl", "main_catalog_tbl"),
            ("order#123$", "order_123_"),
            ("table_with_underscore", "table_with_underscore"),
            ("123_numeric_start", "123_numeric_start"),
            ("SELECT", "SELECT"),
        ],
    )
    def test_sanitize_id_edge_cases(self, input_name: str, expected_clean: str) -> None:
        """Verify identifier sanitization eliminates characters illegal in Mermaid syntax."""
        cleaned = _sanitize_id(input_name)
        assert cleaned == expected_clean

    @pytest.mark.parametrize(
        ("type_str", "expected_clean"),
        [
            ("STRING", "string"),
            ("ARRAY<STRING>", "array_string_"),
            ("MAP<STRING, INT>", "map_string__int_"),
            ("DECIMAL(10, 2)", "decimal_10__2_"),
            ("", "string"),
        ],
    )
    def test_sanitize_type_edge_cases(self, type_str: str, expected_clean: str) -> None:
        """Verify data type sanitization handles complex parameterized types cleanly."""
        cleaned = _sanitize_type(type_str)
        assert cleaned == expected_clean
        assert " " not in cleaned

    def test_sanitize_type_whitespace_returns_underscores(self) -> None:
        """Adversarial observation: whitespace in type produces underscores rather than 'string'."""
        cleaned = _sanitize_type("   ")
        assert cleaned == "___"

    def test_er_diagram_unescaped_description_quotes_finding(self) -> None:
        """Adversarial observation: column descriptions with double quotes produce unescaped quotes.

        In Mermaid erDiagram, attributes are formatted as:
            {type} {name} {PK/FK} "{description}"
        If description contains double quotes, it yields:
            decimal_18__2_ balance_usd "Amount in "USD""
        which breaks Mermaid rendering.
        """
        ent = EntityModel(
            name="customer_account",
            columns=[
                ColumnModel(
                    name="balance_usd",
                    type="decimal(18, 2)",
                    description='Amount in "USD"',
                ),
            ],
        )
        er_code = generate_er_diagram([ent])
        assert 'balance_usd "Amount in "USD""' in er_code

    def test_er_diagram_with_adversarial_entities(self) -> None:
        """Verify ER diagram generation with special characters in entities, columns, descriptions."""
        ent1 = EntityModel(
            name="customer-account",
            table_name="lakehouse.sales.customer-account",
            primary_key="account_id",
            columns=[
                ColumnModel(
                    name="account_id",
                    type="string",
                    primary_key=True,
                    description="Unique customer ID",
                ),
                ColumnModel(
                    name="balance-usd",
                    type="decimal(18, 2)",
                    description="Amount in USD",
                ),
            ],
            relationships=[
                RelationshipModel(
                    name="account_orders",
                    from_entity="customer-account",
                    to_entity="order_items",
                    from_column="account_id",
                    to_column="customer_ref",
                    type="one_to_many",
                    description="Orders placed by account",
                )
            ],
        )
        ent2 = EntityModel(
            name="order_items",
            table_name="lakehouse.sales.order_items",
            primary_key="order_id",
            columns=[
                ColumnModel(name="order_id", type="string", primary_key=True),
                ColumnModel(name="customer_ref", type="string"),
            ],
        )

        er_code = generate_er_diagram([ent1, ent2])
        assert "erDiagram" in er_code
        # Cardinality symbol for one_to_many
        assert "customer_account ||--o{ order_items :" in er_code
        # Column declarations
        assert "customer_account {" in er_code
        assert "string account_id PK" in er_code
        assert "decimal_18__2_ balance_usd" in er_code
        assert "order_items {" in er_code

    def test_er_diagram_empty_entities(self) -> None:
        """Verify ER diagram generation gracefully outputs erDiagram header for empty list."""
        er_code = generate_er_diagram([])
        assert er_code.strip() == "erDiagram"

    def test_lineage_diagram_layer_partitioning(self) -> None:
        """Verify flowchart partitioning across Bronze, Silver, and Gold Medallion layers."""
        entities = [
            EntityModel(name="raw_events", layer="bronze"),
            EntityModel(name="clean_events", layer="silver"),
            EntityModel(name="kpi_summary", layer="gold"),
            EntityModel(name="lookup_table", layer="custom"),
        ]

        flowchart = generate_lineage_diagram(entities)
        assert "graph LR" in flowchart
        assert "subgraph Bronze [Bronze Layer: Raw Ingestion]" in flowchart
        assert "subgraph Silver [Silver Layer: Cleansed & Conformed]" in flowchart
        assert "subgraph Gold [Gold Layer: Aggregated Business KPIs]" in flowchart
        assert "subgraph Entities [Lakehouse Entities]" in flowchart
        # Default synthesized flow connections
        assert "raw_events -->|ETL Dedup & Clean| clean_events" in flowchart
        assert "clean_events -->|Aggregate KPIs| kpi_summary" in flowchart


# ==============================================================================
# SECTION 3: CI Quality Gate Approval & Rejection Behavior
# ==============================================================================


class TestCIQualityGateFuzzing:
    """Stress-testing CI quality gate under diverse code inputs, anti-patterns, and syntax faults."""

    def test_ci_pipeline_approves_pristine_code(self) -> None:
        """Verify clean code passes all checks with full approval."""
        py_code = (
            "from pyspark.sql import SparkSession\n"
            "def run_etl(spark: SparkSession) -> None:\n"
            "    df = spark.table('bronze_orders')\n"
            "    df_clean = df.filter(df['status'] == 'CONFIRMED')\n"
            "    df_clean.write.format('delta').mode('overwrite').saveAsTable('silver_orders')\n"
        )
        sql_code = (
            "CREATE TABLE IF NOT EXISTS silver_orders (\n"
            "    order_id STRING,\n"
            "    amount DOUBLE\n"
            ") USING DELTA;\n"
            "SELECT order_id, amount FROM silver_orders WHERE amount > 0;\n"
        )
        report = run_ci_pipeline(pyspark_code=py_code, sparksql_code=sql_code)
        assert report.is_approved is True
        assert report.ruff_status == "PASSED"
        assert report.sqlfluff_status == "PASSED"
        assert len(report.anti_patterns) == 0

    def test_ci_pipeline_rejects_pyspark_syntax_error(self) -> None:
        """Verify fatal Python syntax error immediately blocks approval."""
        bad_py = "def broken_code(df: return df"
        report = run_ci_pipeline(pyspark_code=bad_py)
        assert report.is_approved is False
        assert report.ruff_status == "FAILED"
        assert any(v.rule == "PYSPARK-SYNTAX-ERR" for v in report.anti_patterns)

    def test_ci_pipeline_rejects_sparksql_syntax_error(self) -> None:
        """Verify fatal SparkSQL parse error blocks approval."""
        bad_sql = "SELEC * FORM my_table WHERE"
        report = run_ci_pipeline(sparksql_code=bad_sql)
        assert report.is_approved is False
        assert report.sqlfluff_status == "FAILED"
        assert any(v.rule.startswith("PRS") for v in report.violations)

    def test_ci_pipeline_rejects_unbounded_collect(self) -> None:
        """Verify dangerous .collect() triggers SPARK-ANTI-001 and rejects pipeline."""
        code = (
            "from pyspark.sql import SparkSession\n"
            "spark = SparkSession.builder.getOrCreate()\n"
            "rows = spark.table('huge_table').collect()\n"
        )
        report = run_ci_pipeline(pyspark_code=code)
        assert report.is_approved is False
        assert any(v.rule == "SPARK-ANTI-001" for v in report.anti_patterns)

    def test_ci_pipeline_allows_bounded_collect(self) -> None:
        """Verify bounded .limit(N).collect() is permitted as safe."""
        code = (
            "from pyspark.sql import SparkSession\n"
            "spark = SparkSession.builder.getOrCreate()\n"
            "sample = spark.table('huge_table').limit(10).collect()\n"
        )
        report = run_ci_pipeline(pyspark_code=code)
        assert not any(v.rule == "SPARK-ANTI-001" for v in report.anti_patterns)

    def test_ci_pipeline_rejects_cross_join_variants(self) -> None:
        """Verify both PySpark .crossJoin() and SparkSQL CROSS JOIN are rejected."""
        py_cross = "df = df1.crossJoin(df2)\n"
        rep_py = run_ci_pipeline(pyspark_code=py_cross)
        assert rep_py.is_approved is False
        assert any(v.rule == "SPARK-ANTI-002" for v in rep_py.anti_patterns)

        sql_cross = "SELECT * FROM df1 CROSS JOIN df2;"
        rep_sql = run_ci_pipeline(sparksql_code=sql_cross)
        assert rep_sql.is_approved is False
        assert any(v.rule == "SQL-ANTI-001" for v in rep_sql.anti_patterns)

    def test_ci_pipeline_rejects_unbounded_to_pandas(self) -> None:
        """Verify unbounded .toPandas() triggers SPARK-ANTI-004 error."""
        code = "df = spark.table('large_table').toPandas()\n"
        report = run_ci_pipeline(pyspark_code=code)
        assert report.is_approved is False
        assert any(v.rule == "SPARK-ANTI-004" for v in report.anti_patterns)

    def test_ci_pipeline_rejects_unconstrained_delete(self) -> None:
        """Verify DELETE statement without WHERE clause is blocked."""
        bad_delete = "DELETE FROM customer_data;"
        report = run_ci_pipeline(sparksql_code=bad_delete)
        assert report.is_approved is False
        assert any(v.rule == "SQL-ANTI-003" for v in report.anti_patterns)

        # Valid delete with WHERE passes
        safe_delete = "DELETE FROM customer_data WHERE account_id = '123';"
        report_safe = run_ci_pipeline(sparksql_code=safe_delete)
        assert not any(v.rule == "SQL-ANTI-003" for v in report_safe.anti_patterns)

    def test_vulnerability_block_comment_bypasses_delete_check(self) -> None:
        """Adversarial observation: block comment /* WHERE */ bypasses SQL-ANTI-003.

        Because check_sparksql only strips single-line '--' comments and not '/* */',
        the substring 'WHERE' in a block comment fools the check into believing
        a valid WHERE clause exists.
        """
        tricky_delete = "DELETE FROM customer_data; /* WHERE 1=1 */"
        violations = DataAntiPatternDetector.check_sparksql(tricky_delete)
        # Empirically demonstrated: misses the anti-pattern!
        assert not any(v.rule == "SQL-ANTI-003" for v in violations)

    def test_vulnerability_update_without_where_not_implemented(self) -> None:
        """Adversarial observation: UPDATE without WHERE is not checked by detector.

        Even though docstring states 'Rule 3: DELETE or UPDATE without WHERE',
        the implementation regex only checks r'\\bDELETE\\s+FROM\\b'.
        """
        dangerous_update = "UPDATE customer_data SET active = false;"
        violations = DataAntiPatternDetector.check_sparksql(dangerous_update)
        assert len(violations) == 0

    def test_ci_pipeline_delta_maintenance_isolation(self) -> None:
        """Verify Delta OPTIMIZE and VACUUM statements do not cause SQLFluff parser failures."""
        sql = (
            "OPTIMIZE silver_sales ZORDER BY (transaction_date);\n"
            "VACUUM silver_sales RETAIN 168 HOURS;\n"
        )
        report = run_ci_pipeline(sparksql_code=sql)
        assert report.is_approved is True
        assert report.sqlfluff_status == "PASSED"

    def test_ci_pipeline_empty_and_whitespace_inputs(self) -> None:
        """Verify CI runner gracefully handles empty, None, or whitespace inputs."""
        rep_none = run_ci_pipeline(None, None)
        assert rep_none.is_approved is True
        assert rep_none.ruff_status == "SKIPPED"
        assert rep_none.sqlfluff_status == "SKIPPED"

        rep_ws = run_ci_pipeline("   \n   ", "  \t  ")
        assert rep_ws.is_approved is True
        assert rep_ws.ruff_status == "SKIPPED"
        assert rep_ws.sqlfluff_status == "SKIPPED"


# ==============================================================================
# SECTION 4: Open WebUI Pipe Execution (Streaming & Non-Streaming)
# ==============================================================================


class TestOpenWebUIPipeFuzzing:
    """Stress-testing Open WebUI Pipe execution across varied chat payloads and valves."""

    def test_pipe_empty_body_payload(self) -> None:
        """Verify pipe handles completely empty body without crashing."""
        pipe = Pipe()
        res = pipe.pipe({})
        assert isinstance(res, str)
        assert "Databricks Steward Agent" in res

    def test_pipe_malformed_messages_payloads(self) -> None:
        """Verify pipe handles missing, empty, or heterogeneous message entries."""
        pipe = Pipe()

        # 1. Empty message list
        res1 = pipe.pipe({"messages": []})
        assert "Databricks Steward Agent" in res1

        # 2. String list instead of dicts
        res2 = pipe.pipe({"messages": ["plain string instead of dict"]})
        assert isinstance(res2, str)

        # 3. Dict message without content
        res3 = pipe.pipe({"messages": [{"role": "user"}]})
        assert isinstance(res3, str)

        # 4. Content is None
        res4 = pipe.pipe({"messages": [{"role": "user", "content": None}]})
        assert isinstance(res4, str)

    def test_pipe_routing_across_domains(self) -> None:
        """Verify pipe correctly dispatches user intents based on natural language queries."""
        pipe = Pipe()

        # 1. Mermaid diagram request
        body_diag = {"messages": [{"role": "user", "content": "Gere um diagrama ER das vendas"}]}
        res_diag = pipe.pipe(body_diag)
        assert "erDiagram" in res_diag

        # 2. Unity Catalog introspection
        body_cat = {"messages": [{"role": "user", "content": "inspecionar catalogo unity"}]}
        res_cat = pipe.pipe(body_cat)
        assert "Discovered" in res_cat or "entities" in res_cat

        # 3. Semantic models lookup
        body_sem = {"messages": [{"role": "user", "content": "ver modelos semanticos e metricas"}]}
        res_sem = pipe.pipe(body_sem)
        assert "Semantic Domain" in res_sem or "Domain" in res_sem

        # 4. Medallion ETL pipeline generation
        body_etl = {"messages": [{"role": "user", "content": "gerar pipeline gold de facilities"}]}
        res_etl = pipe.pipe(body_etl)
        assert "Generated Medallion Pipeline" in res_etl
        assert "PySpark" in res_etl

        # 5. CI Quality Gate
        body_ci = {"messages": [{"role": "user", "content": "validar codigo na esteira de ci lint"}]}
        res_ci = pipe.pipe(body_ci)
        assert "CI Quality Gate Report" in res_ci

        # 6. GitOps PR (without the word 'pipeline' which triggers rule 4)
        body_pr = {"messages": [{"role": "user", "content": "criar pull request no github para o data product"}]}
        res_pr = pipe.pipe(body_pr)
        assert "PR Successfully Created" in res_pr or "Branch" in res_pr

    def test_vulnerability_pipe_routing_precedence_conflict(self) -> None:
        """Adversarial observation: 'pipeline' keyword in PR requests hijacks routing to ETL.

        When user prompt contains both 'pull request' and 'pipeline', Rule 4 (ETL)
        fires before Rule 6 (GitOps), preventing PR creation.
        """
        pipe = Pipe()
        body = {"messages": [{"role": "user", "content": "abrir pull request no github com o pipeline"}]}
        res = pipe.pipe(body)
        # Empirically demonstrated: routes to Medallion ETL, not GitOps PR
        assert "Generated Medallion Pipeline" in res

    def test_pipe_streaming_execution(self) -> None:
        """Verify streaming generator yields non-empty text chunks reconstituting full output."""
        pipe = Pipe()
        payload = {
            "messages": [{"role": "user", "content": "desenhar fluxo medallion lineage"}],
            "stream": True,
        }
        res_gen = pipe.pipe(payload)
        assert isinstance(res_gen, types.GeneratorType)

        chunks = list(res_gen)
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert "graph LR" in full_text
        assert "subgraph" in full_text

    def test_vulnerability_pipe_routing_language_gap_linhagem(self) -> None:
        """Adversarial observation: 'linhagem' falls back to ER diagram instead of graph LR.

        The sub-router checks 'if \"lineage\" in q_lower or \"fluxo\" in q_lower:',
        omitting 'linhagem', so Portuguese prompts like 'diagrama de linhagem'
        render erDiagram instead of Medallion lineage flowchart.
        """
        pipe = Pipe()
        payload = {"messages": [{"role": "user", "content": "desenhar diagrama mermaid de linhagem"}]}
        res = pipe.pipe(payload)
        assert "erDiagram" in res
        assert "graph LR" not in res

    def test_pipe_large_payload_resilience(self) -> None:
        """Verify pipe survives very large input text without memory or recursion failure."""
        pipe = Pipe()
        giant_query = "qual a linhagem? " + ("detalhes " * 10000)
        payload = {"messages": [{"role": "user", "content": giant_query}]}
        res = pipe.pipe(payload)
        assert isinstance(res, str)
        assert len(res) > 0

    def test_pipe_valves_configuration_and_override(self) -> None:
        """Verify administrative Valves instantiate with defaults and accept overrides."""
        pipe = Pipe()
        assert pipe.valves.LOCAL_LLM_MODEL is not None
        assert pipe.valves.GITHUB_BASE_BRANCH is not None

        pipe.valves.LOCAL_LLM_MODEL = "deepseek-coder:6.7b"
        pipe.valves.DATABRICKS_HOST = "https://custom-lakehouse.cloud.databricks.com"
        assert pipe.valves.LOCAL_LLM_MODEL == "deepseek-coder:6.7b"
        assert pipe.valves.DATABRICKS_HOST == "https://custom-lakehouse.cloud.databricks.com"

    def test_pipe_runtime_exception_handling(self) -> None:
        """Verify pipe catches unexpected runtime exceptions and returns safe error message."""
        pipe = Pipe()
        with patch.object(pipe.graph, "invoke", side_effect=RuntimeError("Simulated LLM network crash")):
            payload = {"messages": [{"role": "user", "content": "gerar pipeline"}]}
            res = pipe.pipe(payload)
            assert isinstance(res, str)
            assert "❌ Steward Agent Error:" in res
            assert "Simulated LLM network crash" in res
