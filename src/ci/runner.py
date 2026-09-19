"""Programmatic CI pipeline runner executing Ruff, SQLFluff, and Anti-Pattern checks."""

from __future__ import annotations

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
    """Orchestrates code linting, dialect syntax verification, and data engineering quality checks."""

    @classmethod
    def run_ci_pipeline(
        cls,
        pyspark_code: str | None = None,
        sparksql_code: str | None = None,
        sql_dialect: str = "sparksql",
    ) -> CIReport:
        """Run all automated CI quality gate checks on PySpark and SparkSQL code.

        Args:
            pyspark_code: Optional PySpark script content to validate.
            sparksql_code: Optional SparkSQL query content to validate.
            sql_dialect: SQLFluff dialect to use (default: 'sparksql').

        Returns:
            CIReport with approval status, tool statuses, violations, and markdown summary.
        """
        violations: list[Violation] = []
        anti_patterns: list[Violation] = []

        ruff_status = "SKIPPED"
        sqlfluff_status = "SKIPPED"

        # 1. PySpark validation (Ruff + Anti-Patterns)
        if pyspark_code and pyspark_code.strip():
            # Run anti-pattern detector
            spark_anti = DataAntiPatternDetector.check_pyspark(pyspark_code)
            anti_patterns.extend(spark_anti)

            # Run Ruff linter via subprocess
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

                            # Treat syntax errors and undefined symbols as blocking errors
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

                # If Python AST parser in anti-patterns found a syntax error, mark fatal
                if any(v.rule == "PYSPARK-SYNTAX-ERR" for v in anti_patterns):
                    has_ruff_fatal = True

                ruff_status = "FAILED" if has_ruff_fatal else "PASSED"
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # 2. SparkSQL validation (SQLFluff + Anti-Patterns)
        if sparksql_code and sparksql_code.strip():
            # Run anti-pattern detector on complete original query
            sql_anti = DataAntiPatternDetector.check_sparksql(sparksql_code)
            anti_patterns.extend(sql_anti)

            # Strip Delta maintenance statements (OPTIMIZE, VACUUM, COPY INTO)
            # so valid Delta pipelines are not rejected by SQLFluff parser limitations
            lintable_sql = strip_delta_maintenance(sparksql_code)

            if not lintable_sql.strip():
                # All statements were Delta Lake maintenance commands; valid Delta pipeline
                sqlfluff_status = "PASSED"
            else:
                # Run SQLFluff programmatic linter with configured dialect
                try:
                    lint_results = sqlfluff.lint(lintable_sql, dialect=sql_dialect)
                    has_sql_error = False
                    for item in lint_results:
                        code = item.get("code", "SQL")
                        desc = item.get("description", "SQL violation")
                        line_no = item.get("start_line_no")

                        # PRS = unparsable section / syntax error
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

        # Determine approval:
        # Rejected if Ruff has fatal syntax error, SQLFluff has parse error,
        # or any anti-pattern has error severity
        error_violations = [v for v in violations if v.severity == "error"]
        error_anti = [v for v in anti_patterns if v.severity == "error"]

        is_approved = (
            ruff_status != "FAILED"
            and sqlfluff_status != "FAILED"
            and not error_violations
            and not error_anti
        )

        report = CIReport(
            is_approved=is_approved,
            ruff_status=ruff_status,
            sqlfluff_status=sqlfluff_status,
            anti_patterns=anti_patterns,
            violations=violations,
        )
        report.format_markdown()
        return report


def run_ci_pipeline(
    pyspark_code: str | None = None,
    sparksql_code: str | None = None,
    sql_dialect: str = "sparksql",
) -> CIReport:
    """Module-level entry point conforming to PROJECT.md interface contract."""
    return CIRunner.run_ci_pipeline(
        pyspark_code=pyspark_code,
        sparksql_code=sparksql_code,
        sql_dialect=sql_dialect,
    )
