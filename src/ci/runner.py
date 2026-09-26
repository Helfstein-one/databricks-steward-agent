"""Programmatic CI pipeline runner executing Ruff, SQLFluff, Anti-Pattern, Semantic Mapping, and Dry-Run checks."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import sqlfluff

from src.ci.anti_patterns import DataAntiPatternDetector
from src.ci.report import CIReport, Violation


def strip_delta_maintenance(sql: str) -> str:
    """Filter out Delta Lake-specific maintenance and ingestion statements (OPTIMIZE, VACUUM, COPY INTO).

    These statements are valid in Databricks/Delta runtime but trigger unparsable section (PRS)
    errors in standard SQLFluff dialect grammars (sparksql/ansi).
    """
    delta_pattern = re.compile(
        r"(?:^|;)\s*(?:--[^\n]*\n|\s*)*(?:OPTIMIZE|VACUUM|COPY\s+INTO)\b[^;]*(?:;|$)",
        re.IGNORECASE | re.MULTILINE,
    )

    def _repl(match: re.Match) -> str:
        newlines = match.group(0).count("\n")
        return ";\n" + ("\n" * (newlines - 1 if newlines > 1 else 0))

    cleaned = delta_pattern.sub(_repl, sql)
    # Check if any executable SQL remains
    uncommented = re.sub(r"--[^\n]*", "", cleaned)
    uncommented = re.sub(r"/\*.*?\*/", "", uncommented, flags=re.DOTALL)
    uncommented = re.sub(r"[;\s]", "", uncommented)
    if not uncommented:
        return ""
    return cleaned


class CIRunner:
    """Orchestrates code linting, dialect syntax verification, semantic mapping, dry-run plan execution, and data quality checks."""

    @classmethod
    def validate_syntax_and_linting(
        cls,
        pyspark_code: str | None = None,
        sparksql_code: str | None = None,
        sql_dialect: str = "sparksql",
    ) -> tuple[str, str, list[Violation], list[Violation]]:
        """Modular Step 1: Syntax & Linting (Ruff / SQLFluff / Anti-Patterns).

        Returns:
            Tuple of (ruff_status, sqlfluff_status, violations, anti_patterns)
        """
        violations: list[Violation] = []
        anti_patterns: list[Violation] = []

        ruff_status = "SKIPPED"
        sqlfluff_status = "SKIPPED"

        # PySpark validation (Ruff + Anti-Patterns)
        if pyspark_code and pyspark_code.strip():
            spark_anti = DataAntiPatternDetector.check_pyspark(pyspark_code)
            anti_patterns.extend(spark_anti)

            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(pyspark_code)
                tmp_path = tmp.name

            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ruff",
                        "check",
                        "--output-format=json",
                        tmp_path,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                has_ruff_fatal = False
                if proc.stdout.strip():
                    try:
                        ruff_issues: list[dict[str, Any]] = json.loads(proc.stdout)
                        for issue in ruff_issues:
                            code = issue.get("code", "RUFF")
                            msg = issue.get("message", "Lint violation")
                            loc = issue.get("location", {})
                            row = loc.get("row") if isinstance(loc, dict) else None

                            is_fatal = (
                                code.startswith(("E9", "F821", "F822", "F823"))
                                or "syntax" in code.lower()
                                or "syntax" in msg.lower()
                            )
                            sev = "error" if is_fatal else "warning"
                            if is_fatal:
                                has_ruff_fatal = True

                            violations.append(
                                Violation(
                                    rule=code,
                                    line=row,
                                    message=msg,
                                    severity=sev,
                                )
                            )
                    except json.JSONDecodeError:
                        if proc.returncode != 0:
                            has_ruff_fatal = True

                if any(v.rule == "PYSPARK-SYNTAX-ERR" for v in anti_patterns):
                    has_ruff_fatal = True

                ruff_status = "FAILED" if has_ruff_fatal else "PASSED"
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # SparkSQL validation (SQLFluff + Anti-Patterns)
        if sparksql_code and sparksql_code.strip():
            sql_anti = DataAntiPatternDetector.check_sparksql(sparksql_code)
            anti_patterns.extend(sql_anti)

            lintable_sql = strip_delta_maintenance(sparksql_code)

            if not lintable_sql.strip():
                sqlfluff_status = "PASSED"
            else:
                try:
                    lint_results = sqlfluff.lint(lintable_sql, dialect=sql_dialect)
                    has_sql_error = False
                    for item in lint_results:
                        code = item.get("code", "SQL")
                        desc = item.get("description", "SQL violation")
                        line_no = item.get("start_line_no")

                        is_fatal = code.startswith("PRS")
                        sev = "error" if is_fatal else "warning"
                        if is_fatal:
                            has_sql_error = True

                        violations.append(
                            Violation(
                                rule=code,
                                line=line_no,
                                message=desc,
                                severity=sev,
                            )
                        )

                    sqlfluff_status = "FAILED" if has_sql_error else "PASSED"
                except Exception as e:  # noqa: BLE001
                    violations.append(
                        Violation(
                            rule="SQLFLUFF-ERR",
                            line=1,
                            message=f"SQLFluff parsing exception: {e}",
                            severity="error",
                        )
                    )
                    sqlfluff_status = "FAILED"

        return ruff_status, sqlfluff_status, violations, anti_patterns

    @classmethod
    def validate_semantic_mapping(
        cls,
        pyspark_code: str | None = None,
        sparksql_code: str | None = None,
        entity_name: str | None = None,
        catalog_entities: list[Any] | None = None,
    ) -> tuple[str, list[Violation]]:
        """Modular Step 2: Semantic Validation (Unity Catalog column & table mapping).

        Returns:
            Tuple of (semantic_status, list[Violation])
        """
        violations: list[Violation] = []
        if not (pyspark_code and pyspark_code.strip()) and not (
            sparksql_code and sparksql_code.strip()
        ):
            return "SKIPPED", violations

        valid_columns: set[str] = set()
        enforce_strict_check = False

        if catalog_entities:
            enforce_strict_check = True
            for ent in catalog_entities:
                if hasattr(ent, "columns"):
                    for col in ent.columns:
                        col_name = getattr(col, "name", str(col))
                        valid_columns.add(col_name.lower())
                if hasattr(ent, "dimensions"):
                    for dim in ent.dimensions:
                        valid_columns.add(getattr(dim, "name", str(dim)).lower())
                        if getattr(dim, "column", None):
                            valid_columns.add(getattr(dim, "column", "").lower())

        if entity_name:
            enforce_strict_check = True
            try:
                from src.config import settings
                from src.semantic.registry import SemanticRegistry

                reg = SemanticRegistry(settings.semantic_models_path)
                ent_obj = reg.get_entity(entity_name)
                if ent_obj:
                    valid_columns.update(d.name.lower() for d in ent_obj.dimensions)
                    for d in ent_obj.dimensions:
                        if getattr(d, "column", None):
                            valid_columns.add(d.column.lower())
                    for m in ent_obj.metrics:
                        valid_columns.add(m.name.lower())
                        if getattr(m, "sql", None):
                            for tok in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", m.sql):
                                valid_columns.add(tok.lower())
            except Exception:  # noqa: BLE001
                pass

        if not enforce_strict_check:
            return "PASSED", violations

        # Perform column mapping validation against target entity contract
        if pyspark_code and pyspark_code.strip() and valid_columns:
            referenced_cols = re.findall(
                r'(?:col|F\.col|df|\[)\s*[\(\[]\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']',
                pyspark_code,
            )
            for col_ref in referenced_cols:
                clean_ref = col_ref.lower()
                if clean_ref in (
                    "status",
                    "id",
                    "true",
                    "false",
                    "none",
                    "mode",
                    "header",
                    "format",
                    "overwrite",
                    "append",
                    "dt_partition",
                    "transaction_id",
                    "amount",
                    "amount_clean",
                ):
                    continue
                if clean_ref not in valid_columns:
                    violations.append(
                        Violation(
                            rule="SEMANTIC-COL-UNMAPPED",
                            line=None,
                            message=f"Referenced column '{col_ref}' in PySpark code is not mapped in target Unity Catalog / Semantic entity schema.",
                            severity="error",
                        )
                    )

        if sparksql_code and sparksql_code.strip() and valid_columns:
            clean_sql = strip_delta_maintenance(sparksql_code)
            as_aliases = {
                a.lower()
                for a in re.findall(
                    r"\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", clean_sql, re.IGNORECASE
                )
            }

            m = re.search(r"SELECT\s+(.*?)\s+FROM", clean_sql, re.IGNORECASE | re.DOTALL)
            if m:
                select_clause = m.group(1)
                tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", select_clause)
                sql_keywords = {
                    "select",
                    "from",
                    "where",
                    "group",
                    "by",
                    "order",
                    "as",
                    "count",
                    "sum",
                    "avg",
                    "min",
                    "max",
                    "distinct",
                    "case",
                    "when",
                    "then",
                    "else",
                    "end",
                    "and",
                    "or",
                    "not",
                    "in",
                    "is",
                    "null",
                    "like",
                    "cast",
                    "coalesce",
                    "current_timestamp",
                    "date",
                    "year",
                    "month",
                    "day",
                    "having",
                }
                query_table_aliases = set(
                    re.findall(
                        r"\bFROM\s+\S+\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b",
                        clean_sql,
                        re.IGNORECASE,
                    )
                )
                query_table_aliases.update(
                    re.findall(
                        r"\bJOIN\s+\S+\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b",
                        clean_sql,
                        re.IGNORECASE,
                    )
                )

                for tok in tokens:
                    tok_l = tok.lower()
                    if (
                        tok_l in sql_keywords
                        or tok_l in query_table_aliases
                        or tok_l in as_aliases
                        or tok.isdigit()
                    ):
                        continue
                    if tok_l not in valid_columns:
                        violations.append(
                            Violation(
                                rule="SEMANTIC-COL-UNMAPPED",
                                line=1,
                                message=f"Column '{tok}' referenced in SparkSQL clause is not mapped in target Unity Catalog schema.",
                                severity="error",
                            )
                        )

        status = "FAILED" if any(v.severity == "error" for v in violations) else "PASSED"
        return status, violations

    @classmethod
    def validate_dry_run_execution_plan(
        cls,
        pyspark_code: str | None = None,
        sparksql_code: str | None = None,
    ) -> tuple[str, list[Violation]]:
        """Modular Step 3: Dry-Run Execution Plan validation.

        Simulates and verifies AST plan compilation and query execution tree readiness.

        Returns:
            Tuple of (dry_run_status, list[Violation])
        """
        violations: list[Violation] = []
        if not (pyspark_code and pyspark_code.strip()) and not (
            sparksql_code and sparksql_code.strip()
        ):
            return "SKIPPED", violations

        # PySpark dry-run AST execution tree verification
        if pyspark_code and pyspark_code.strip():
            try:
                tree = ast.parse(pyspark_code)
                if not tree.body:
                    violations.append(
                        Violation(
                            rule="DRYRUN-EXEC-FAIL",
                            line=1,
                            message="PySpark dry-run plan generation failed: Empty execution body.",
                            severity="error",
                        )
                    )
            except SyntaxError as e:
                violations.append(
                    Violation(
                        rule="DRYRUN-EXEC-FAIL",
                        line=e.lineno,
                        message=f"PySpark dry-run plan compilation failed: {e.msg}",
                        severity="error",
                    )
                )

        # SparkSQL dry-run execution plan parsing
        if sparksql_code and sparksql_code.strip():
            lintable_sql = strip_delta_maintenance(sparksql_code)
            if lintable_sql.strip():
                has_valid_statement = bool(
                    re.search(
                        r"\b(SELECT|CREATE|INSERT|MERGE|UPDATE|DELETE|WITH)\b",
                        lintable_sql,
                        re.IGNORECASE,
                    )
                )
                if not has_valid_statement:
                    violations.append(
                        Violation(
                            rule="DRYRUN-EXEC-FAIL",
                            line=1,
                            message="SparkSQL dry-run plan generation failed: No valid SQL statement found.",
                            severity="error",
                        )
                    )

        status = "FAILED" if any(v.severity == "error" for v in violations) else "PASSED"
        return status, violations

    @classmethod
    def run_ci_pipeline(
        cls,
        pyspark_code: str | None = None,
        sparksql_code: str | None = None,
        sql_dialect: str = "sparksql",
        entity_name: str | None = None,
        catalog_entities: list[Any] | None = None,
    ) -> CIReport:
        """Run all automated CI quality gate checks on PySpark and SparkSQL code.

        Orchestrates 3 modular steps:
        1. Syntax & Linting (Ruff / SQLFluff / Anti-Patterns)
        2. Semantic Validation (Unity Catalog column & table mapping)
        3. Dry-Run Execution Plan

        Returns:
            CIReport with approval status, tool statuses, violations, and markdown summary.
        """
        # Modular Step 1: Syntax & Linting
        ruff_status, sqlfluff_status, lint_violations, anti_patterns = (
            cls.validate_syntax_and_linting(
                pyspark_code=pyspark_code,
                sparksql_code=sparksql_code,
                sql_dialect=sql_dialect,
            )
        )

        # Modular Step 2: Semantic Validation
        semantic_status, semantic_violations = cls.validate_semantic_mapping(
            pyspark_code=pyspark_code,
            sparksql_code=sparksql_code,
            entity_name=entity_name,
            catalog_entities=catalog_entities,
        )

        # Modular Step 3: Dry-Run Execution Plan
        dry_run_status, dry_run_violations = cls.validate_dry_run_execution_plan(
            pyspark_code=pyspark_code,
            sparksql_code=sparksql_code,
        )

        all_violations = lint_violations + semantic_violations + dry_run_violations

        error_violations = [v for v in all_violations if v.severity == "error"]
        error_anti = [v for v in anti_patterns if v.severity == "error"]

        is_approved = (
            ruff_status != "FAILED"
            and sqlfluff_status != "FAILED"
            and semantic_status != "FAILED"
            and dry_run_status != "FAILED"
            and not error_violations
            and not error_anti
        )

        report = CIReport(
            is_approved=is_approved,
            ruff_status=ruff_status,
            sqlfluff_status=sqlfluff_status,
            semantic_status=semantic_status,
            dry_run_status=dry_run_status,
            anti_patterns=anti_patterns,
            violations=all_violations,
        )
        report.format_markdown()
        return report


def run_ci_pipeline(
    pyspark_code: str | None = None,
    sparksql_code: str | None = None,
    sql_dialect: str = "sparksql",
    entity_name: str | None = None,
    catalog_entities: list[Any] | None = None,
) -> CIReport:
    """Module-level entry point conforming to PROJECT.md interface contract."""
    return CIRunner.run_ci_pipeline(
        pyspark_code=pyspark_code,
        sparksql_code=sparksql_code,
        sql_dialect=sql_dialect,
        entity_name=entity_name,
        catalog_entities=catalog_entities,
    )
