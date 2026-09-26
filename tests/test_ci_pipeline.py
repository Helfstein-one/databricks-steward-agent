"""Unit and component integration tests for Data Best Practices CI quality gate (Tier 2)."""

from src.ci.anti_patterns import DataAntiPatternDetector
from src.ci.report import CIReport, Violation
from src.ci.runner import CIRunner, run_ci_pipeline

# ==============================================================================
# Anti-Pattern Detector Tests
# ==============================================================================


def test_anti_pattern_detects_unbounded_collect(flawed_pyspark_collect_code):
    """Verify detection of unbounded .collect() anti-pattern."""
    violations = DataAntiPatternDetector.check_pyspark(flawed_pyspark_collect_code)
    assert len(violations) >= 1
    collect_v = next((v for v in violations if v.rule == "SPARK-ANTI-001"), None)
    assert collect_v is not None
    assert "collect()" in collect_v.message
    assert collect_v.severity == "error"


def test_anti_pattern_allows_bounded_collect():
    """Verify that .limit(N).collect() is permitted and not flagged as anti-pattern."""
    bounded_code = """
def sample(spark):
    df = spark.table("main.default.tbl")
    return df.limit(10).collect()
"""
    violations = DataAntiPatternDetector.check_pyspark(bounded_code)
    assert not any(v.rule == "SPARK-ANTI-001" for v in violations)


def test_anti_pattern_detects_cross_join(flawed_pyspark_cross_join_code):
    """Verify detection of explicit .crossJoin() anti-pattern."""
    violations = DataAntiPatternDetector.check_pyspark(flawed_pyspark_cross_join_code)
    assert len(violations) >= 1
    join_v = next((v for v in violations if v.rule == "SPARK-ANTI-002"), None)
    assert join_v is not None
    assert "crossJoin" in join_v.message
    assert join_v.severity == "error"


def test_anti_pattern_detects_unbounded_topandas():
    """Verify detection of unbounded .toPandas() on distributed DataFrame."""
    pandas_code = """
def convert_data(df):
    return df.toPandas()
"""
    violations = DataAntiPatternDetector.check_pyspark(pandas_code)
    topandas_v = next((v for v in violations if v.rule == "SPARK-ANTI-004"), None)
    assert topandas_v is not None
    assert "toPandas" in topandas_v.message


def test_anti_pattern_sql_cross_join(flawed_sparksql_cross_join_code):
    """Verify detection of SQL cross joins or SELECT *."""
    violations = DataAntiPatternDetector.check_sparksql(flawed_sparksql_cross_join_code)
    assert any(v.rule == "SQL-ANTI-002" for v in violations)  # SELECT * warning


def test_anti_pattern_sql_explicit_cross_join():
    """Verify detection of explicit CROSS JOIN in SQL."""
    sql = "SELECT a.id, b.name FROM table_a CROSS JOIN table_b;"
    violations = DataAntiPatternDetector.check_sparksql(sql)
    assert any(v.rule == "SQL-ANTI-001" for v in violations)


def test_anti_pattern_sql_delete_without_where():
    """Verify detection of DELETE FROM without WHERE clause."""
    sql = "DELETE FROM main.credit.counterparts;"
    violations = DataAntiPatternDetector.check_sparksql(sql)
    assert any(v.rule == "SQL-ANTI-003" for v in violations)


# ==============================================================================
# Modular CI Runner Tests
# ==============================================================================


def test_modular_step_syntax_and_linting(clean_pyspark_code, clean_sparksql_code):
    """Verify Modular Step 1: Syntax & Linting."""
    r_stat, s_stat, _violations, anti = CIRunner.validate_syntax_and_linting(
        pyspark_code=clean_pyspark_code,
        sparksql_code=clean_sparksql_code,
    )
    assert r_stat == "PASSED"
    assert s_stat == "PASSED"
    assert len(anti) == 0


def test_modular_step_semantic_mapping():
    """Verify Modular Step 2: Semantic Validation."""
    # Test unknown column reference against explicit entity contract
    sql = "SELECT unknown_col_xyz FROM main.credit.facilities;"
    status, violations = CIRunner.validate_semantic_mapping(
        sparksql_code=sql, entity_name="facilities"
    )
    assert status == "FAILED"
    assert any(v.rule == "SEMANTIC-COL-UNMAPPED" for v in violations)


def test_modular_step_dry_run_execution_plan(clean_pyspark_code, clean_sparksql_code):
    """Verify Modular Step 3: Dry-Run Execution Plan validation."""
    status, violations = CIRunner.validate_dry_run_execution_plan(
        pyspark_code=clean_pyspark_code,
        sparksql_code=clean_sparksql_code,
    )
    assert status == "PASSED"
    assert len(violations) == 0

    # Malformed dry run
    bad_status, bad_violations = CIRunner.validate_dry_run_execution_plan(
        sparksql_code="INVALID_STMT_WITHOUT_SQL_KEYWORDS"
    )
    assert bad_status == "FAILED"
    assert any(v.rule == "DRYRUN-EXEC-FAIL" for v in bad_violations)


# ==============================================================================
# CI Runner Integration Tests
# ==============================================================================


def test_ci_pipeline_approves_clean_code(clean_pyspark_code, clean_sparksql_code):
    """Verify CI runner approves clean, properly structured PySpark and SparkSQL code."""
    report = run_ci_pipeline(
        pyspark_code=clean_pyspark_code,
        sparksql_code=clean_sparksql_code,
    )

    assert isinstance(report, CIReport)
    assert report.is_approved is True
    assert report.ruff_status == "PASSED"
    assert report.sqlfluff_status == "PASSED"
    assert report.semantic_status == "PASSED"
    assert report.dry_run_status == "PASSED"
    assert len(report.anti_patterns) == 0
    assert "APPROVED" in report.summary_markdown


def test_ci_pipeline_rejects_pyspark_syntax_error(flawed_pyspark_syntax_error_code):
    """Verify CI runner rejects code containing Python syntax errors."""
    report = run_ci_pipeline(pyspark_code=flawed_pyspark_syntax_error_code)

    assert report.is_approved is False
    assert report.ruff_status == "FAILED" or any(v.severity == "error" for v in report.violations)
    assert "REJECTED" in report.summary_markdown


def test_ci_pipeline_rejects_sparksql_syntax_error(flawed_sparksql_syntax_error_code):
    """Verify CI runner rejects code containing SparkSQL syntax errors."""
    report = run_ci_pipeline(sparksql_code=flawed_sparksql_syntax_error_code)

    assert report.is_approved is False
    assert report.sqlfluff_status == "FAILED" or any(
        v.severity == "error" for v in report.violations
    )
    assert "REJECTED" in report.summary_markdown


def test_ci_pipeline_rejects_anti_patterns(flawed_pyspark_collect_code):
    """Verify CI runner strictly rejects code triggering data anti-patterns."""
    report = run_ci_pipeline(pyspark_code=flawed_pyspark_collect_code)

    assert report.is_approved is False
    assert len(report.anti_patterns) >= 1
    assert "REJECTED" in report.summary_markdown
    assert "SPARK-ANTI-001" in report.summary_markdown


def test_ci_report_markdown_formatting():
    """Verify Markdown report generation format and diagnostic table."""
    report = CIReport(
        is_approved=False,
        ruff_status="FAILED",
        sqlfluff_status="PASSED",
        semantic_status="PASSED",
        dry_run_status="PASSED",
        anti_patterns=[
            Violation(
                rule="SPARK-ANTI-001",
                line=42,
                message="Dangerous collect() detected",
                severity="error",
            )
        ],
        violations=[
            Violation(
                rule="E999",
                line=10,
                message="SyntaxError in Python file",
                severity="error",
            )
        ],
    )

    md = report.format_markdown()
    assert "### Data Engineering CI Quality Gate Report" in md
    assert "❌ **REJECTED**" in md
    assert "| **Ruff (Python / PySpark)** | FAILED |" in md
    assert "| **SQLFluff (SparkSQL)** | PASSED |" in md
    assert "| **Data Anti-Patterns** | FAILED | 1 |" in md
    assert "| **Semantic Validation (Unity Catalog)** | PASSED |" in md
    assert "| **Dry-Run Execution Plan** | PASSED |" in md
    assert "`SPARK-ANTI-001`" in md
    assert "`E999`" in md
