"""LangGraph state representation for Databricks Steward Agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from src.ci.report import CIReport
from src.gitops.github_pr import GitOpsResult
from src.semantic.models import EntityModel


class AgentState(TypedDict, total=False):
    """Conversational and execution state tracked across LangGraph nodes."""

    messages: Annotated[list[Any], add_messages]
    catalog_context: str | None
    semantic_entities: list[EntityModel]
    active_diagram: str | None
    generated_code: dict[str, str] | None
    ci_report: CIReport | None
    gitops_result: GitOpsResult | None
    user_query: str | None
    response: str | None
    pending_pipeline: dict[str, Any] | None
    job_result: dict[str, Any] | None
    preview_data: dict[str, Any] | None
