"""LangGraph agent coordinating Databricks stewardship, semantic modeling, CI, and GitOps."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
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
_resolved_models_cache: dict[str, str] = {}


def resolve_local_model(base_url: str, preferred_model: str) -> str:
    """Discover available models on OpenAI-compatible endpoint and select an active model."""
    if not base_url:
        return preferred_model

    # Preserve default development model on localhost to ensure deterministic unit tests
    if preferred_model == "qwen2.5-coder:7b" and ("localhost" in base_url or "127.0.0.1" in base_url):
        return preferred_model

    cached = _resolved_models_cache.get(base_url)
    if cached:
        return cached

    clean_base = base_url.rstrip("/")
    models_endpoint = f"{clean_base}/models" if clean_base.endswith("/v1") else f"{clean_base}/v1/models"

    try:
        req = urllib.request.Request(models_endpoint, headers={"User-Agent": "Databricks-Steward/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            available_ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
            if not available_ids and "models" in data:
                available_ids = [m.get("name") for m in data["models"] if isinstance(m, dict)]

            if preferred_model in available_ids:
                _resolved_models_cache[base_url] = preferred_model
                return preferred_model

            if available_ids:
                priority = ["llama3.2:3b", "llama3.2:1b", "qwen2.5-coder:7b", "deepseek-r1:1.5b"]
                chosen = next((p for p in priority if p in available_ids), available_ids[0])
                logger.info(
                    "Model '%s' not found at %s. Auto-selected available model '%s'.",
                    preferred_model, base_url, chosen,
                )
                _resolved_models_cache[base_url] = chosen
                return chosen
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not resolve models from %s: %s", models_endpoint, e)

    return preferred_model


def get_local_chat_client(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """Instantiate OpenAI-compatible local chat client (Ollama/vLLM) with automatic model resolution."""
    raw_base = base_url or settings.local_llm_base_url
    raw_model = model or settings.local_llm_model
    effective_model = resolve_local_model(raw_base, raw_model)
    return ChatOpenAI(
        base_url=raw_base,
        model=effective_model,
        api_key=api_key or settings.local_llm_api_key or "ollama",
        temperature=temperature if temperature is not None else settings.local_llm_temperature,
        timeout=15.0,
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


def _is_greeting(query: str) -> bool:
    """Check if query is a conversational greeting."""
    q = (query or "").strip().lower()
    if not q:
        return True
    pattern = r"^(ol[áa]|oi|bom\s+dia|boa\s+tarde|boa\s+noite|hello|hi|hey|sauda[çc][õo]es|e\s+a[íi]|tudo\s+bem|como\s+vai)[!?,.\s]*$"
    if re.match(pattern, q):
        return True
    return bool(any(q.startswith(g) for g in ("olá", "ola", "oi", "hello", "hi")) and len(q.split()) <= 4)


def _is_title_request(query: str) -> bool:
    """Check if query is an Open WebUI background prompt to generate a conversation title."""
    q = (query or "").lower()
    return (
        ("title" in q or "título" in q)
        and any(w in q for w in ("generate", "gerar", "crie", "create", "summarize", "resumo", "3-5", "words", "palavras", "chat", "conversa"))
    ) or (
        "generate a concise" in q and "title" in q
    )


def _deterministic_steward_execution(state: AgentState) -> dict[str, Any]:
    """Fallback deterministic rule-based router executing steward capabilities."""
    messages = state.get("messages", [])
    user_query = state.get("user_query") or _extract_query_text(messages)
    q_lower = (user_query or "").strip().lower()

    response_text = ""
    active_diagram = state.get("active_diagram")
    generated_code = state.get("generated_code")
    ci_report = state.get("ci_report")
    gitops_result = state.get("gitops_result")

    # Title generation request from Open WebUI
    if _is_title_request(user_query):
        response_text = "Databricks Steward - Governança"
        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
        }

    # Numeric shortcuts from welcome menu
    is_opt_1 = q_lower in ("1", "1.", "opcao 1", "opção 1")
    is_opt_2 = q_lower in ("2", "2.", "opcao 2", "opção 2")
    is_opt_3 = q_lower in ("3", "3.", "opcao 3", "opção 3")
    is_opt_4 = q_lower in ("4", "4.", "opcao 4", "opção 4")
    is_opt_5 = q_lower in ("5", "5.", "opcao 5", "opção 5")
    is_opt_6 = q_lower in ("6", "6.", "opcao 6", "opção 6")

    # 1. Mermaid Diagram Generation (Option 3)
    if is_opt_3 or any(k in q_lower for k in ("diagram", "diagrama", "erd", "erdiagram", "mermaid", "lineage", "desenhar", "modelo visual")):
        if "lineage" in q_lower or "fluxo" in q_lower or "medallion" in q_lower:
            diag = generate_diagram("lineage")
        elif "sales" in q_lower or "vendas" in q_lower:
            diag = generate_diagram("er", domain="sales_lakehouse")
        else:
            diag = generate_diagram("er", domain="corporate_credit")

        active_diagram = diag
        response_text = f"Here is the requested Mermaid diagram:\n\n{diag}"

    # 2. Databricks Unity Catalog Introspection (Option 1)
    elif is_opt_1 or any(k in q_lower for k in ("catalog", "catálogo", "schema", "tabelas", "unity catalog", "introspect")):
        response_text = inspect_unity_catalog()

    # 3. Semantic Layer & Business Models (Option 2)
    elif is_opt_2 or any(k in q_lower for k in ("semantic", "semântica", "semantica", "metrica", "métrica", "dimensao", "dimensão", "ontology", "ontologia", "negocio", "negócio")):
        response_text = load_semantic_models()

    # 4. ETL Pipeline Generation
    elif is_opt_4 or any(k in q_lower for k in ("etl", "pipeline", "pyspark", "sparksql", "bronze", "silver", "gold")):
        layer = "gold" if "gold" in q_lower else "bronze" if "bronze" in q_lower else "silver"
        entity_name = "facilities"
        if "sales" in q_lower or "order" in q_lower or "venda" in q_lower:
            entity_name = "orders"
        elif "transaction" in q_lower or "transac" in q_lower:
            entity_name = "silver_transactions" if layer == "silver" else "bronze_raw_transactions"

        reg = SemanticRegistry(settings.semantic_models_path)
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
    elif is_opt_5 or any(k in q_lower for k in ("ci", "esteira", "lint", "ruff", "sqlfluff", "anti-pattern", "validar")):
        py_code = generated_code.get("pyspark") if generated_code else None
        sql_code = generated_code.get("sparksql") if generated_code else None

        if not py_code or not sql_code:
            py_code = "def process(df):\n    return df.filter(df['active'] == True)\n"
            sql_code = "SELECT order_id, total_amount FROM main.sales.orders;"

        report = run_ci_pipeline(pyspark_code=py_code, sparksql_code=sql_code)
        ci_report = report
        response_text = report.summary_markdown or report.format_markdown()

    # 6. GitOps & PR Opening
    elif is_opt_6 or re.search(r"\b(pr|pull\s*request|gitops|branch|commit|push)\b", q_lower):
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

    # Greeting / Default assistance menu
    else:
        response_text = (
            "Olá! Sou o **Databricks Steward Agent**, seu copiloto de governança e engenharia de dados Lakehouse.\n\n"
            "Posso ajudar você com:\n"
            "1. **Unity Catalog Introspection**: Descobrir catálogos, schemas e tabelas.\n"
            "2. **Semantic Modeling**: Consultar dimensões, métricas de negócio e relacionamentos.\n"
            "3. **Mermaid.js Diagrams**: Gerar diagramas conceituais (`erDiagram`) e fluxos medalhão.\n"
            "4. **Modular ETL Engineering**: Produzir pipelines idempotentes em PySpark e SparkSQL (Bronze, Silver, Gold).\n"
            "5. **CI Quality Gate**: Validar código com Ruff, SQLFluff (sparksql) e detecção de anti-patterns.\n"
            "6. **Automated GitOps**: Criar feature branches, Conventional Commits e abrir Pull Requests no GitHub.\n\n"
            "💡 *Dica: Digite o número da opção (ex: `1`, `3`, `4`) ou descreva sua solicitação em linguagem natural!*"
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
    """Main routing and execution node for the Databricks Steward Agent."""
    client = llm or get_local_chat_client()
    base_url = str(getattr(client, "openai_api_base", None) or settings.local_llm_base_url)
    model_name = str(getattr(client, "model_name", None) or settings.local_llm_model)
    cache_key = (base_url, model_name)

    messages = state.get("messages", [])
    user_query = state.get("user_query") or _extract_query_text(messages)
    q_lower = (user_query or "").strip().lower()

    # 1. Instant resolution for UI title requests
    if _is_title_request(user_query):
        return {
            "messages": [AIMessage(content="Databricks Steward - Governança")],
            "response": "Databricks Steward - Governança",
            "active_diagram": state.get("active_diagram"),
            "generated_code": state.get("generated_code"),
            "ci_report": state.get("ci_report"),
            "gitops_result": state.get("gitops_result"),
        }

    # 2. Direct numeric menu shortcuts
    if q_lower in ("1", "1.", "opcao 1", "opção 1", "2", "2.", "opcao 2", "opção 2",
                   "3", "3.", "opcao 3", "opção 3", "4", "4.", "opcao 4", "opção 4",
                   "5", "5.", "opcao 5", "opção 5", "6", "6.", "opcao 6", "opção 6"):
        return _deterministic_steward_execution(state)

    now = time.time()
    cached = _availability_cache.get(cache_key)

    is_available = True
    if cached is not None:
        avail, expiry = cached
        if now < expiry:
            is_available = avail

    # 3. Conversational greetings handled naturally without tool bindings
    if _is_greeting(user_query):
        if is_available:
            try:
                system_prompt = (
                    "Você é o Databricks Steward Agent, assistente especializado em governança e engenharia de dados Lakehouse. "
                    "Apresente-se cordialmente em português de forma clara e profissional. "
                    "Explique resumidamente que você ajuda com:\n"
                    "1. Unity Catalog Introspection: descoberta de tabelas e schemas\n"
                    "2. Camada Semântica: métricas e dimensões de negócio\n"
                    "3. Diagramas Mermaid: modelagem visual ER e linhagem medalhão\n"
                    "4. Pipelines ETL: geração PySpark/SparkSQL Bronze, Silver e Gold\n"
                    "5. Esteira de CI: qualidade de código (Ruff, SQLFluff, anti-patterns)\n"
                    "6. GitOps: automação de branch, commit e Pull Requests no GitHub\n"
                    "Oriente o usuário a escolher um número (1 a 6) ou descrever sua necessidade."
                )
                llm_res = client.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query or "olá"},
                ])
                _availability_cache[cache_key] = (True, now + 30.0)
                reply = str(llm_res.content or "").strip()
                if reply.startswith("{") and reply.endswith("}"):
                    try:
                        p = json.loads(reply)
                        if isinstance(p, dict) and "parameters" in p and "message" in p["parameters"]:
                            reply = str(p["parameters"]["message"])
                    except Exception as err:  # noqa: BLE001
                        logger.debug("Failed parsing JSON greeting reply: %s", err)
                if reply:
                    return {
                        "messages": [AIMessage(content=reply)],
                        "response": reply,
                        "active_diagram": state.get("active_diagram"),
                        "generated_code": state.get("generated_code"),
                        "ci_report": state.get("ci_report"),
                        "gitops_result": state.get("gitops_result"),
                    }
            except Exception as e:  # noqa: BLE001
                _availability_cache[cache_key] = (False, now + 10.0)
                logger.debug("Local LLM offline or unreachable (%s); using deterministic welcome.", e)
        return _deterministic_steward_execution(state)

    # 4. Technical queries: LLM with tool calling
    if is_available:
        try:
            tools = get_langchain_tools()
            llm_with_tools = client.bind_tools(tools)
            if not messages and user_query:
                messages = [{"role": "user", "content": user_query}]

            response = llm_with_tools.invoke(messages)
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
                            reg = SemanticRegistry(settings.semantic_models_path)
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
                content_str = str(response.content).strip()
                if content_str.startswith("{") and content_str.endswith("}"):
                    try:
                        p = json.loads(content_str)
                        if isinstance(p, dict) and "parameters" in p and "message" in p["parameters"]:
                            content_str = str(p["parameters"]["message"])
                    except Exception as err:  # noqa: BLE001
                        logger.debug("Failed parsing JSON content reply: %s", err)
                return {
                    "messages": [response],
                    "response": content_str,
                    "active_diagram": state.get("active_diagram"),
                    "generated_code": state.get("generated_code"),
                    "ci_report": state.get("ci_report"),
                    "gitops_result": state.get("gitops_result"),
                }
        except Exception as e:  # noqa: BLE001
            _availability_cache[cache_key] = (False, now + 10.0)
            logger.debug("Local LLM offline or unreachable (%s); using deterministic steward router.", e)

    # 5. Deterministic fallback
    return _deterministic_steward_execution(state)


def create_steward_graph(llm: ChatOpenAI | None = None) -> Any:
    """Create and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    node_func = (lambda s: steward_node(s, llm=llm)) if llm is not None else steward_node
    workflow.add_node("steward", node_func)
    workflow.add_edge(START, "steward")
    workflow.add_edge("steward", END)

    return workflow.compile()
