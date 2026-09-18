"""Structured CI quality gate report and markdown formatter."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Violation(BaseModel):
    """Specific lint, syntax, or data anti-pattern violation."""

    rule: str
    line: int | None = None
    message: str
    severity: str = "error"  # "error", "warning", "info"


class CIReport(BaseModel):
    """Comprehensive CI quality gate validation report."""

    is_approved: bool
    ruff_status: str  # "PASSED", "FAILED", "SKIPPED"
    sqlfluff_status: str  # "PASSED", "FAILED", "SKIPPED"
    anti_patterns: list[Violation] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    summary_markdown: str = ""

    def format_markdown(self) -> str:
        """Format the report into a structured GitHub / Open WebUI markdown table."""
        status_badge = "✅ **APPROVED**" if self.is_approved else "❌ **REJECTED**"

        lines = [
            "### Data Engineering CI Quality Gate Report",
            f"**Overall Status:** {status_badge}",
            "",
            "| Check Suite | Status | Violations Count |",
            "|---|---|---|",
            f"| **Ruff (Python / PySpark)** | {self.ruff_status} | {len([v for v in self.violations if v.rule.startswith(('E', 'F', 'W', 'I', 'B', 'SIM', 'UP', 'RUFF'))])} |",
            f"| **SQLFluff (SparkSQL)** | {self.sqlfluff_status} | {len([v for v in self.violations if v.rule.startswith(('LT', 'CP', 'AL', 'RF', 'PRS', 'SQL'))])} |",
            f"| **Data Anti-Patterns** | {'PASSED' if not self.anti_patterns else 'FAILED'} | {len(self.anti_patterns)} |",
            "",
        ]

        all_issues = self.violations + self.anti_patterns
        if all_issues:
            lines.extend([
                "#### Diagnostic Details",
                "| Severity | Rule | Line | Description |",
                "|---|---|---|---|",
            ])
            for issue in all_issues:
                sev_icon = "🔴" if issue.severity == "error" else "🟡"
                line_str = str(issue.line) if issue.line is not None else "-"
                clean_msg = issue.message.replace("|", "/")
                lines.append(f"| {sev_icon} {issue.severity.upper()} | `{issue.rule}` | {line_str} | {clean_msg} |")
        else:
            lines.append("🎉 *All automated linting, syntax, and data engineering best practices checks passed cleanly!*")

        rendered = "\n".join(lines)
        self.summary_markdown = rendered
        return rendered
