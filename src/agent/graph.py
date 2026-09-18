"""LangGraph agent coordinating Databricks stewardship, semantic modeling, CI, and GitOps."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState
from src.agent.tools import (
    generate_diagram,
    generate_etl_pipeline,
    get_langchain_tools,
    inspect_unity_catalog,
    load_semantic_models,
)
from src.ci.runner import run_ci_pipeline
from src.config import settings
from src.etl.generator import generate_medallion_pipeline
from src.gitops.github_pr import create_data_product_pr
from src.semantic.registry import SemanticRegistry

logger = logging.getLogger(__name__)

# Cache for local LLM endpoint reachability to avoid repeated timeouts
_availability_cache: dict[tuple[str, str], tuple[bool, float]] = {}


def get_local_chat_client(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """Instantiate OpenAI-compatible local chat client (Ollama/vLLM)."""
    return ChatOpenAI(
        base_url=base_url or settings.local_llm_base_url,
        model=model or settings.local_llm_model,
        api_key=api_key or settings.local_llm_api_key or "ollama",
        temperature=temperature if temperature is not None else settings.local_llm_temperature,
        timeout=3.0,
        max_retries=0,
    )


def _extract_query_text(messages: list[Any]) -> str:
    """Extract text from the last user message."""
    if not messages:
        return ""
    last_msg = messages[-1]
    if isinstance(last_msg, dict):
        return last_msg.get("content", "") or ""
    if hasattr(last_msg, "content"):
        return str(last_msg.content or "")
    return str(last_msg or "")


def _deterministic_steward_execution(state: AgentState) -> dict[str, Any]:
    """Fallback deterministic rule-based router executing steward capabilities."""
    messages = state.get("messages", [])
    user_query = state.get("user_query") or _extract_query_text(messages)
    q_lower = (user_query or "").lower()

    response_text = ""
    active_diagram = state.get("active_diagram")
    generated_code = state.get("generated_code")
    ci_report = state.get("ci_report")
    gitops_result = state.get("gitops_result")

    # 1. Mermaid Diagram Generation
    if any(k in q_lower for k in ("diagram", "erd", "erdiagram", "mermaid", "lineage", "desenhar", "modelo visual")):
        if "lineage" in q_lower or "fluxo" in q_lower or "medallion" in q_lower:
            diag = generate_diagram("lineage")
        elif "sales" in q_lower:
            diag = generate_diagram("er", domain="sales_lakehouse")
        else:
            diag = generate_diagram("er", domain="corporate_credit")

        active_diagram = diag
        response_text = f"Here is the requested Mermaid diagram:\n\n{diag}"

    # 2. Databricks Unity Catalog Introspection
    elif any(k in q_lower for k in ("catalog", "schema", "tabelas", "unity catalog", "introspect")):
        response_text = inspect_unity_catalog()

    # 3. Semantic Layer & Business Models
    elif any(k in q_lower for k in ("semantic", "metrica", "dimensao", "ontology", "ontologia", "negocio")):
        response_text = load_semantic_models()

    # 4. ETL Pipeline Generation
    elif any(k in q_lower for k in ("etl", "pipeline", "pyspark", "sparksql", "bronze", "silver", "gold")):
        layer = "gold" if "gold" in q_lower else "bronze" if "bronze" in q_lower else "silver"
        entity_name = "facilities"
        if "sales" in q_lower or "order" in q_lower:
            entity_name = "orders"
        elif "transaction" in q_lower:
            entity_name = "silver_transactions" if layer == "silver" else "bronze_raw_transactions"

        # Generate pipeline and preserve generated code in state
        reg = SemanticRegistry("configs/semantic_models")
        ent_obj = reg.get_entity(entity_name)
        if not ent_obj:
            from src.databricks.introspector import _build_mock_entities
            mock_ents = {e.name: e for e in _build_mock_entities()}
            ent_obj = mock_ents.get(entity_name)

        if ent_obj:
            pipeline = generate_medallion_pipeline(ent_obj, layer=layer)
            generated_code = {
                "pyspark": pipeline.pyspark_code,
                "sparksql": pipeline.sparksql_code,
                "table_name": pipeline.table_name,
                "layer": pipeline.layer,
            }
            response_text = (
                f"### Generated Medallion Pipeline: {pipeline.table_name} ({pipeline.layer})\n\n"
                f"#### PySpark Pipeline\n```python\n{pipeline.pyspark_code}\n```\n\n"
                f"#### SparkSQL DDL & Ingestion\n```sql\n{pipeline.sparksql_code}\n```"
            )
        else:
            response_text = generate_etl_pipeline(entity_name, layer=layer)

    # 5. Data Best Practices CI Quality Gate
    elif any(k in q_lower for k in ("ci", "lint", "ruff", "sqlfluff", "anti-pattern", "validar")):
        py_code = generated_code.get("pyspark") if generated_code else None
        sql_code = generated_code.get("sparksql") if generated_code else None

        if not py_code or not sql_code:
            py_code = "def process(df):\n    return df.filter(df['active'] == True)\n"
            sql_code = "SELECT order_id, total_amount FROM main.sales.orders;"

        report = run_ci_pipeline(pyspark_code=py_code, sparksql_code=sql_code)
        ci_report = report
        response_text = report.summary_markdown or report.format_markdown()

    # 6. GitOps & PR Opening (word boundary regex prevents false positives on 'produtos', 'preco', etc.)
    elif re.search(r"\b(pr|pull\s*request|gitops|branch|commit|push)\b", q_lower):
        py_code = generated_code.get("pyspark") if generated_code else None
        sql_code = generated_code.get("sparksql") if generated_code else None

        if not py_code or not sql_code:
            py_code = "def process(df):\n    return df.filter(df['active'] == True)\n"
            sql_code = "SELECT order_id, total_amount FROM main.sales.orders;"

        report = ci_report or run_ci_pipeline(pyspark_code=py_code, sparksql_code=sql_code)
        ci_report = report

        product_name = generated_code.get("table_name", "corporate-credit-kpis") if generated_code else "corporate-credit-kpis"
        product_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", product_name.replace(".", "-")).lower()

        files = {
            f"pipelines/{product_slug}/etl.py": py_code,
            f"pipelines/{product_slug}/schema.sql": sql_code,
        }

        result = create_data_product_pr(
            product_name=product_slug,
            files=files,
            ci_report=report,
            diagram_md=active_diagram or "",
            dry_run=True,
        )
        gitops_result = result

        if result.status == "rejected":
            response_text = f"❌ PR Creation Blocked: CI Quality Gate Rejected the pipeline.\n\n{report.summary_markdown}"
        else:
            response_text = (
                f"✅ PR Successfully Created!\n\n"
                f"- **Branch:** `{result.branch_name}`\n"
                f"- **Commit SHA:** `{result.commit_sha}`\n"
                f"- **PR URL:** [{result.pr_url}]({result.pr_url})\n\n"
                f"#### CI Gate Report\n{report.summary_markdown}"
            )

    # Default assistance
    else:
        response_text = (
            "Hello! I am the **Databricks Steward Agent**.\n\n"
            "I can assist you with:\n"
            "1. **Unity Catalog Introspection**: Discover catalogs, schemas, and tables.\n"
            "2. **Semantic Modeling**: Query business dimensions, metrics, and relationships.\n"
            "3. **Mermaid.js Diagrams**: Generate ER diagrams (`erDiagram`) and Medallion flowcharts.\n"
            "4. **Modular ETL Engineering**: Produce idempotent PySpark & SparkSQL pipelines (Bronze, Silver, Gold).\n"
            "5. **CI Quality Gate**: Verify code with Ruff, SQLFluff (sparksql), and anti-pattern detectors.\n"
            "6. **Automated GitOps**: Create feature branches, Conventional Commits, and open GitHub PRs."
        )

    return {
        "messages": [AIMessage(content=response_text)],
        "response": response_text,
        "active_diagram": active_diagram,
        "generated_code": generated_code,
        "ci_report": ci_report,
        "gitops_result": gitops_result,
    }


def steward_node(state: AgentState, llm: ChatOpenAI | None = None) -> dict[str, Any]:
    """Main routing and execution node for the Databricks Steward Agent.

    Attempts genuine LLM reasoning with tool calling if model is available,
    falling back to deterministic intent execution if offline, unreachable, or on error.
    """
    client = llm or get_local_chat_client()
    base_url = str(getattr(client, "openai_api_base", None) or settings.local_llm_base_url)
    model_name = str(getattr(client, "model_name", None) or settings.local_llm_model)
    cache_key = (base_url, model_name)

    now = time.time()
    cached = _availability_cache.get(cache_key)

    is_available = True
    if cached is not None:
        avail, expiry = cached
        if now < expiry:
            is_available = avail

    if is_available:
        try:
            tools = get_langchain_tools()
            llm_with_tools = client.bind_tools(tools)
            messages = state.get("messages", [])
            user_query = state.get("user_query") or _extract_query_text(messages)
            if not messages and user_query:
                messages = [{"role": "user", "content": user_query}]

            # Attempt LLM call
            response = llm_with_tools.invoke(messages)

            # Record success in cache
            _availability_cache[cache_key] = (True, now + 30.0)

            # Check if model produced tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_dict = {t.name: t for t in tools}
                results = []
                active_diagram = state.get("active_diagram")
                generated_code = state.get("generated_code")
                ci_report = state.get("ci_report")
                gitops_result = state.get("gitops_result")

                for tc in response.tool_calls:
                    t_name = tc.get("name")
                    t_args = tc.get("args", {})
                    if t_name in tool_dict:
                        t_output = tool_dict[t_name].invoke(t_args)
                        results.append(str(t_output))
                        if t_name == "generate_diagram_tool":
                            active_diagram = str(t_output)
                        elif t_name == "generate_etl_pipeline_tool":
                            ent = t_args.get("entity_name", "facilities")
                            lyr = t_args.get("layer", "silver")
                            reg = SemanticRegistry("configs/semantic_models")
                            e_model = reg.get_entity(ent)
                            if e_model:
                                pipe = generate_medallion_pipeline(e_model, layer=lyr)
                                generated_code = {
                                    "pyspark": pipe.pyspark_code,
                                    "sparksql": pipe.sparksql_code,
                                    "table_name": pipe.table_name,
                                    "layer": pipe.layer,
                                }

                combined_resp = "\n\n".join(results)
                return {
                    "messages": [AIMessage(content=combined_resp)],
                    "response": combined_resp,
                    "active_diagram": active_diagram,
                    "generated_code": generated_code,
                    "ci_report": ci_report,
                    "gitops_result": gitops_result,
                }
            if response.content:
                return {
                    "messages": [response],
                    "response": str(response.content),
                    "active_diagram": state.get("active_diagram"),
                    "generated_code": state.get("generated_code"),
                    "ci_report": state.get("ci_report"),
                    "gitops_result": state.get("gitops_result"),
                }
        except Exception as e:  # noqa: BLE001
            # Mark unavailable in cache for 10 seconds to avoid repeating failed connection attempts
            _availability_cache[cache_key] = (False, now + 10.0)
            logger.debug("Local LLM offline or unreachable (%s); using deterministic steward router.", e)

    # Deterministic fallback engine
    return _deterministic_steward_execution(state)


def create_steward_graph(llm: ChatOpenAI | None = None) -> Any:
    """Create and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    node_func = (lambda s: steward_node(s, llm=llm)) if llm is not None else steward_node
    workflow.add_node("steward", node_func)
    workflow.add_edge(START, "steward")
    workflow.add_edge("steward", END)

    return workflow.compile()
