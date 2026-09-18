"""Static analysis rules detecting data engineering anti-patterns in PySpark and SparkSQL."""

from __future__ import annotations

import ast
import re

from src.ci.report import Violation


class DataAntiPatternDetector:
    """Detects dangerous big data patterns in PySpark and SparkSQL scripts."""

    @classmethod
    def check_pyspark(cls, code: str) -> list[Violation]:
        """Analyze PySpark code using AST and heuristic rules."""
        violations: list[Violation] = []
        if not code.strip():
            return violations

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(
                Violation(
                    rule="PYSPARK-SYNTAX-ERR",
                    line=e.lineno,
                    message=f"Python syntax error: {e.msg}",
                    severity="error",
                )
            )
            return violations

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Check method calls on DataFrames
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr

                # Rule 1: Dangerous unbounded .collect()
                if method_name == "collect":
                    # Check if preceded by .limit() in call chain
                    has_limit = False
                    curr = node.func.value
                    while isinstance(curr, ast.Call):
                        if isinstance(curr.func, ast.Attribute) and curr.func.attr in ("limit", "take", "head"):
                            has_limit = True
                            break
                        if isinstance(curr.func, ast.Attribute):
                            curr = curr.func.value
                        else:
                            break

                    if not has_limit:
                        violations.append(
                            Violation(
                                rule="SPARK-ANTI-001",
                                line=node.lineno,
                                message="Dangerous unbounded .collect() detected. In big data pipelines, .collect() pulls all partitions into driver memory, leading to Driver OOM.",
                                severity="error",
                            )
                        )

                # Rule 2: Accidental cross join
                elif method_name == "crossJoin":
                    violations.append(
                        Violation(
                            rule="SPARK-ANTI-002",
                            line=node.lineno,
                            message="Accidental or explicit .crossJoin() detected. Cartesian products explode cluster memory and shuffle IO.",
                            severity="error",
                        )
                    )
                elif method_name == "join":
                    # If join has only 1 arg (target df) and no join condition
                    if len(node.args) == 1 and not node.keywords:
                        violations.append(
                            Violation(
                                rule="SPARK-ANTI-002",
                                line=node.lineno,
                                message="Unconstrained .join() without join condition acts as a Cartesian cross join.",
                                severity="error",
                            )
                        )

                # Rule 4: Unbounded .toPandas()
                elif method_name == "toPandas":
                    has_limit = False
                    curr = node.func.value
                    while isinstance(curr, ast.Call):
                        if isinstance(curr.func, ast.Attribute) and curr.func.attr in ("limit", "take", "head"):
                            has_limit = True
                            break
                        if isinstance(curr.func, ast.Attribute):
                            curr = curr.func.value
                        else:
                            break

                    if not has_limit:
                        violations.append(
                            Violation(
                                rule="SPARK-ANTI-004",
                                line=node.lineno,
                                message="Unbounded .toPandas() will crash driver memory on large distributed datasets.",
                                severity="error",
                            )
                        )

        return violations

    @classmethod
    def check_sparksql(cls, sql: str) -> list[Violation]:
        """Analyze SparkSQL query for common SQL data anti-patterns."""
        violations: list[Violation] = []
        if not sql.strip():
            return violations

        # Mask comments by replacing non-newline characters with spaces to preserve line count
        def _mask_comment(m: re.Match) -> str:
            return "".join("\n" if c == "\n" else " " for c in m.group(0))

        sql_clean_comments = re.sub(r"--[^\n]*", _mask_comment, sql)
        sql_clean_comments = re.sub(r"/\*.*?\*/", _mask_comment, sql_clean_comments, flags=re.DOTALL)

        # Rule 1: CROSS JOIN (supports single-line and multiline)
        for match in re.finditer(r"\bCROSS\s+JOIN\b", sql_clean_comments, re.IGNORECASE):
            line_no = sql[: match.start()].count("\n") + 1
            violations.append(
                Violation(
                    rule="SQL-ANTI-001",
                    line=line_no,
                    message="Explicit CROSS JOIN detected. Avoid Cartesian products in distributed SQL.",
                    severity="error",
                )
            )

        # Rule 2: SELECT * in production queries (multiline-aware across full SQL)
        for match in re.finditer(r"\bSELECT\s+\*\s+FROM\b", sql_clean_comments, re.IGNORECASE):
            line_no = sql[: match.start()].count("\n") + 1
            violations.append(
                Violation(
                    rule="SQL-ANTI-002",
                    line=line_no,
                    message="Avoid 'SELECT *' in production lakehouse queries; project explicit columns.",
                    severity="warning",
                )
            )

        # Rule 3: DELETE or UPDATE without WHERE
        sql_clean_single = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        if re.search(r"\bDELETE\s+FROM\b", sql_clean_comments, re.IGNORECASE) and not re.search(r"\bWHERE\b", sql_clean_single, re.IGNORECASE):
            violations.append(
                Violation(
                    rule="SQL-ANTI-003",
                    line=1,
                    message="DELETE FROM statement without WHERE clause will truncate the table.",
                    severity="error",
                )
            )

        return violations
