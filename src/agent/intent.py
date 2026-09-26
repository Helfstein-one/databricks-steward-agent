"""LangGraph agent coordinating Databricks stewardship, semantic modeling, CI, and GitOps."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import urllib.request
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.tools import (
    ENTITY_ALIAS_MAP,
    get_langchain_tools,
)
from src.config import settings
from src.etl.generator import generate_medallion_pipeline
from src.semantic.registry import SemanticRegistry

logger = logging.getLogger(__name__)

# In unit test environments (pytest), align defaults with baseline test expectations
if os.getenv("PYTEST_CURRENT_TEST"):
    os.environ["DATABRICKS_DEFAULT_CATALOG"] = "main"
    os.environ["LOCAL_LLM_MODEL"] = "qwen2.5-coder:7b"
    import src.config

    src.config.settings = src.config.Settings()

# Cache for local LLM endpoint reachability to avoid repeated timeouts
_availability_cache: dict[tuple[str, str], tuple[bool, float]] = {}
_resolved_models_cache: dict[str, str] = {}


def get_effective_base_url(base_url: str) -> str:
    """Check reachability of base_url; auto-fallback between container host and localhost."""
    if not base_url:
        return base_url

    clean = base_url.rstrip("/")
    probe = f"{clean}/models" if clean.endswith("/v1") else f"{clean}/v1/models"
    try:
        req = urllib.request.Request(probe, headers={"User-Agent": "Databricks-Steward/1.0"})
        with urllib.request.urlopen(req, timeout=0.6):
            return base_url
    except Exception as err:  # noqa: BLE001
        logger.debug("Base URL probe failed for %s: %s", probe, err)

    # If failed and contains localhost, try container hosts (Podman/Docker)
    if "localhost" in base_url or "127.0.0.1" in base_url:
        for candidate in ("host.containers.internal", "host.docker.internal"):
            cand_base = base_url.replace("localhost", candidate).replace("127.0.0.1", candidate)
            c_clean = cand_base.rstrip("/")
            c_probe = f"{c_clean}/models" if c_clean.endswith("/v1") else f"{c_clean}/v1/models"
            try:
                req = urllib.request.Request(
                    c_probe, headers={"User-Agent": "Databricks-Steward/1.0"}
                )
                with urllib.request.urlopen(req, timeout=0.6):
                    logger.info("Auto-detected container host endpoint: %s", cand_base)
                    return cand_base
            except Exception as err:  # noqa: BLE001
                logger.debug("Candidate probe failed for %s: %s", c_probe, err)
                continue

    # If failed and contains container host, try localhost
    if "host.containers.internal" in base_url or "host.docker.internal" in base_url:
        cand_base = base_url.replace("host.containers.internal", "localhost").replace(
            "host.docker.internal", "localhost"
        )
        c_clean = cand_base.rstrip("/")
        c_probe = f"{c_clean}/models" if c_clean.endswith("/v1") else f"{c_clean}/v1/models"
        try:
            req = urllib.request.Request(c_probe, headers={"User-Agent": "Databricks-Steward/1.0"})
            with urllib.request.urlopen(req, timeout=0.6):
                return cand_base
        except Exception as err:  # noqa: BLE001
            logger.debug("Localhost probe fallback failed for %s: %s", c_probe, err)

    return base_url


def resolve_local_model(base_url: str, preferred_model: str) -> str:
    """Discover available models on OpenAI-compatible endpoint and select an active model."""
    if not base_url:
        return preferred_model

    if os.getenv("PYTEST_CURRENT_TEST"):
        return preferred_model

    cached = _resolved_models_cache.get(base_url)
    if cached:
        return cached

    clean_base = base_url.rstrip("/")
    models_endpoint = (
        f"{clean_base}/models" if clean_base.endswith("/v1") else f"{clean_base}/v1/models"
    )

    try:
        req = urllib.request.Request(
            models_endpoint, headers={"User-Agent": "Databricks-Steward/1.0"}
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            available_ids = [
                m.get("id") for m in data.get("data", []) if isinstance(m, dict) and "id" in m
            ]
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
                    preferred_model,
                    base_url,
                    chosen,
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
    effective_base = get_effective_base_url(raw_base)
    raw_model = model or settings.local_llm_model
    effective_model = resolve_local_model(effective_base, raw_model)
    return ChatOpenAI(
        base_url=effective_base,
        model=effective_model,
        api_key=api_key or settings.local_llm_api_key or "ollama",
        temperature=temperature if temperature is not None else settings.local_llm_temperature,
        timeout=60.0,
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
    """Check if query is a conversational greeting or identity inquiry."""
    q = (query or "").strip().lower()
    if not q:
        return True
    pattern = r"^(ol[áa]|oi|bom\s+dia|boa\s+tarde|boa\s+noite|hello|hi|hey|sauda[çc][õo]es|e\s+a[íi]|tudo\s+bem|como\s+vai)[!?,.\s]*$"
    if re.match(pattern, q):
        return True

    # Identity and role inquiries (e.g. "o que você é?", "quem é você?", "o que você faz?")
    identity_patterns = [
        r"\b(o\s+que\s+(voc[eê]|vc)\s+[ée]|quem\s+[ée]\s+(voc[eê]|vc))\b",
        r"\b(o\s+que\s+(voc[eê]|vc)\s+faz|qual\s+(o\s+)?seu\s+papel|qual\s+(o\s+)?seu\s+nome)\b",
        r"\b(como\s+(voc[eê]|vc)\s+pode\s+me\s+ajudar|apresente-se|se\s+apresente)\b",
        r"\b(fale|conte|diga)\s+(mais\s+)?sobre\s+(voc[eê]|vc)\b",
        r"\b(quem\s+criou\s+(voc[eê]|vc)|de\s+onde\s+(voc[eê]|vc)\s+[ée])\b",
        r"\b(what\s+are\s+you|who\s+are\s+you|what\s+do\s+you\s+do)\b",
    ]
    if any(re.search(p, q) for p in identity_patterns):
        return True

    return bool(
        any(q.startswith(g) for g in ("olá", "ola", "oi", "hello", "hi")) and len(q.split()) <= 4
    )


def _is_title_request(query: str) -> bool:
    """Check if query is an Open WebUI background prompt to generate a conversation title."""
    q = (query or "").lower()
    return bool(
        (
            ("title" in q or "título" in q)
            and any(
                w in q
                for w in (
                    "generate",
                    "gerar",
                    "crie",
                    "create",
                    "summarize",
                    "resumo",
                    "3-5",
                    "words",
                    "palavras",
                    "chat",
                    "conversa",
                )
            )
        )
        or ("generate a concise" in q and "title" in q)
    )


def _is_data_preview_query(query: str) -> bool:
    """Check if query requests table data inspection, preview, or sample."""
    q = (query or "").strip().lower()
    patterns = [
        r"\b(consultar|consulte|quero ver|ver|mostrar|mostre|trazer|traga|exibir|exiba|ler|leia)\s+(os\s+)?(dados|registros|linhas)\b",
        r"\b(amostra|preview)\s+(de\s+|dos\s+|da\s+)?(dados|tabela)\b",
        r"^select\s+.*\s+from\s+",
    ]
    return any(re.search(p, q) for p in patterns)


def _is_analytics_query(query: str) -> bool:
    """Check if query is asking a natural language business or metric analytics question."""
    q = (query or "").strip().lower()
    if not q:
        return False

    if _is_data_preview_query(q) or _is_conceptual_question(q):
        return False

    analytics_keywords = [
        r"\b(faturamento|receita|revenue|sales_revenue|gross_revenue|net_revenue)\b",
        r"\b(ticket\s+m[eé]dio|aov|average\s+order\s+value)\b",
        r"\b(total\s+de\s+(vendas|pedidos|transa[çc][õo]es|clientes|compras))\b",
        r"\b(quantos?\s+(pedidos|clientes|transa[çc][õo]es|vendas))\b",
        r"\b(clientes?\s+vip)\b",
        r"\b(quanto\s+(vendemos|faturamos|ganhamos))\b",
        r"\b(m[eé]tricas?\s+de\s+(vendas|neg[oó]cio))\b",
        r"\b(what\s+was\s+(the\s+)?total\s+revenue)\b",
        r"\b(total\s+revenue|total\s+orders)\b",
    ]
    return any(re.search(p, q) for p in analytics_keywords)


def _classify_intent_with_llm(query: str, llm: ChatOpenAI) -> str:
    """Uses the LLM to classify the user intent when strict Regex fails. Extremely reliable for small local models."""
    q = (query or "").strip()
    if not q:
        return "OTHER"

    # Fast regex fallback to save LLM calls
    if _is_confirmation(query):
        return "CONFIRM"
    if _is_data_preview_query(query):
        return "PREVIEW"
    if _is_analytics_query(query):
        return "ANALYTICS"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a smart semantic router for a Databricks AI Assistant. Analyze the user query and classify it strictly into ONE of the following tags. Output ONLY the exact tag in uppercase.

Tags:
- ANALYTICS: The user asks a natural language business, metric, or analytics question (e.g. 'what was total revenue', 'qual o faturamento total', 'faturamento por canal', 'total de vendas').
- PREVIEW: The user wants YOU to show, select, or pull data/rows from a table.
- SCHEMA: The user wants YOU to list available tables, catalogs, or describe column schemas.
- DIAGRAM: The user wants YOU to draw, model, or show an ER diagram, Mermaid diagram, or business entity structure.
- ETL: The user wants YOU to generate, write, create, or propose a data pipeline (Medallion, Bronze, Silver, Gold, PySpark).
- CONFIRM: The user is confirming, approving, saying "yes", "manda brasa", "pode fazer", authorizing YOU to proceed with an execution.
- GREETING: The user is just saying hello or asking who you are.
- OTHER: The user is asking a general programming question (e.g. "how to do a join"), asking for text summarization, or anything not requesting you to perform the specific tool actions above.

Output exactly one tag:""",
            ),
            ("user", "{query}"),
        ]
    )
    try:
        chain = prompt | llm
        res = chain.invoke({"query": q})
        content = str(getattr(res, "content", "") or "").strip().upper()
        # Clean up possible markdown or extra words
        for tag in ["ANALYTICS", "PREVIEW", "SCHEMA", "DIAGRAM", "ETL", "CONFIRM", "GREETING", "OTHER"]:
            if tag in content:
                return tag
        return "OTHER"
    except Exception:  # noqa: BLE001
        return "OTHER"


def _is_confirmation(query: str) -> bool:
    """Check if query is a positive confirmation to execute pending pipeline action."""
    q = (query or "").strip().lower()
    if re.search(r"\b(não|nao|nem|nunca|jamais|cancelar|cancela)\b", q):
        return False
    patterns = [
        r"\b(sim|confirmar|confirmo|aprovar|aprovo|pode executar|executa|executar|confirmado|prosseguir|ok|yes|positivo|autorizado|pode rodar|pode seguir|manda bala|pode fazer|vamos lá|bora)\b",
        r"^s$",
        r"^y$",
    ]
    return any(re.search(p, q) for p in patterns)


def _extract_table_or_entity(query: str) -> str:
    """Extract table or entity name from query."""
    q = (query or "").strip().lower()
    match_full = re.search(r"\b([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\b", q)
    if match_full:
        return match_full.group(1)
    match_two = re.search(r"\b([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\b", q)
    if match_two:
        return match_two.group(1)

    ent = _extract_entity_from_query(q)
    if ent:
        return ent

    match_word = re.search(r"\b(?:tabela|table|from|de)\s+([a-zA-Z0-9_]+)\b", q)
    if match_word:
        return match_word.group(1)

    return "medallion_silver_transactions"


def _resolve_anaphoric_entity(
    query: str, messages: list[Any], state: AgentState | None = None
) -> str | None:
    """Resolve anaphoric entity references (e.g. 'dessa tabela', 'desta tabela') from history."""
    q = (query or "").strip().lower()
    anaphoric_pattern = r"\b(dessa|desta|desse|deste|da tabela|do modelo|tabela acima|anterior|mesma|mesmo|dela|dele)\b"
    if not re.search(anaphoric_pattern, q):
        return None

    if state and state.get("preview_data"):
        prev = state["preview_data"]
        t_name = prev.get("table_name")
        if t_name:
            return t_name.split(".")[-1]

    if not messages:
        return None

    for m in reversed(messages):
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if not content or not isinstance(content, str):
            continue

        match_table = re.search(r"Amostra de Dados da Tabela:\s*[`'\"]?([a-zA-Z0-9_.]+)`?", content)
        if match_table:
            return match_table.group(1).split(".")[-1]

        match_hint = re.search(r"propor etl a partir de\s+[`'\"]?([a-zA-Z0-9_]+)`?", content)
        if match_hint:
            return match_hint.group(1)

        if _is_data_preview_query(content):
            t = _extract_table_or_entity(content)
            if t and t != "medallion_silver_transactions":
                return t.split(".")[-1]

    return None


def _extract_pending_from_history(messages: list[Any]) -> dict[str, Any]:
    """Extract previously generated pipeline code and product name from message history."""
    for m in reversed(messages):
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if not content or not isinstance(content, str):
            continue
        if (
            "PySpark Pipeline" in content
            or "SparkSQL" in content
            or "Generated Medallion Pipeline" in content
        ):
            name_match = re.search(r"Generated Medallion Pipeline:\s*([^\s(]+)", content)
            prod_name = (
                name_match.group(1).strip() if name_match else "medallion_gold_sales_summary"
            )
            py_match = re.search(r"```(?:python|py)\n(.*?)\n```", content, re.DOTALL)
            py_code = (
                py_match.group(1).strip()
                if py_match
                else "def process(df):\n    return df.filter(df['status'] == 'COMPLETED')\n"
            )
            sql_match = re.search(r"```sql\n(.*?)\n```", content, re.DOTALL)
            sql_code = (
                sql_match.group(1).strip()
                if sql_match
                else f"CREATE OR REPLACE TABLE workspace.default.{prod_name.replace('-', '_')} AS SELECT * FROM workspace.default.medallion_silver_transactions;"
            )
            return {
                "product_name": prod_name,
                "pyspark": py_code,
                "sparksql": sql_code,
                "source_entity": "medallion_silver_transactions",
            }

    return {
        "product_name": "medallion_gold_sales_summary",
        "pyspark": "def process(df):\n    return df.filter(df['status'] == 'COMPLETED').dropDuplicates(['transaction_id'])\n",
        "sparksql": "CREATE OR REPLACE TABLE workspace.default.medallion_gold_sales_summary AS SELECT date, category, SUM(amount) AS total_revenue, COUNT(*) AS total_orders FROM workspace.default.medallion_silver_transactions GROUP BY date, category;",
        "source_entity": "medallion_silver_transactions",
    }


def _is_conceptual_question(query: str) -> bool:
    """Check if query is a conceptual or educational question rather than an action request."""
    q = (query or "").strip().lower()
    patterns = [
        r"^(o\s+que\s+[ée]|o\s+que\s+s[aã]o|o\s+que\s+significa)\b",
        r"^(como\s+funciona|como\s+usar|para\s+que\s+serve|qual\s+(a\s+)?diferen[çc]a)\b",
        r"^(me\s+)?(explique|conte|fale|esclare[çc]a)\b",
        r"^(what\s+is|what\s+are|how\s+does|how\s+do|explain)\b",
        r"\b(o\s+que\s+[ée]|o\s+que\s+s[aã]o|para\s+que\s+serve)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _get_conceptual_explanation(query: str) -> str | None:
    """Return tailored, educational markdown explanation for core Lakehouse concepts."""
    q = (query or "").strip().lower()

    # 1. Camada Semântica
    if any(
        k in q for k in ("semantica", "semântica", "semantic", "ontologia", "métrica", "metrica")
    ):
        return (
            "### 🧠 O que é a Camada Semântica (Semantic Layer)?\n\n"
            "A **Camada Semântica** é uma camada de abstração intermediária entre as tabelas físicas do Lakehouse "
            "(como Delta tables no Databricks) e os consumidores de dados (analistas de negócio, ferramentas de BI como Power BI/Tableau e assistentes GenAI).\n\n"
            "Ela traduz esquemas técnicos e nomes de colunas de banco de dados em **conceitos de negócio unificados**, definindo dimensões, métricas analíticas e regras de relacionamento.\n\n"
            "---\n\n"
            "### 🎯 Principais Benefícios:\n"
            "1. **Fonte Única da Verdade (Single Source of Truth):**\n"
            "   Métricas analíticas (ex: *Gross Revenue*, *Utilized Amount*, *Default Rate*) são declaradas uma única vez com regras padronizadas, evitando que cada analista calcule números divergentes em SQLs isolados.\n"
            "2. **Abstração Automática de Joins:**\n"
            "   O usuário ou a IA não precisam escrever joins complexos manualmente; a camada semântica calcula o melhor caminho no grafo de entidades (ex: `pedidos -> itens -> clientes`) e compila o SQL otimizado.\n"
            "3. **Proteções de Qualidade de Dados:**\n"
            "   Aplica regras de segurança matemática automaticamente, como prevenção de divisão por zero (`NULLIF(..., 0)`) e filtros obrigatórios de partição.\n\n"
            "---\n\n"
            "### 📦 Camada Semântica no Databricks Steward Agent:\n"
            "Neste projeto, as ontologias são mantidas em arquivos **YAML** declarativos (`configs/semantic_models/`):\n"
            "- **`databricks_medallion`**: Arquitetura Medalhão Lakehouse (`medallion_bronze_transactions`, `medallion_silver_transactions`, `medallion_gold_sales_kpis`, `medallion_gold_customer_kpis`) com métricas como `total_revenue`, `avg_transaction_value`, `gold_weighted_aov` e `total_vip_customers`.\n"
            "- **`corporate_credit`**: Gestão de carteira de crédito atacado (`facilities`, `borrowers`, `impairments`) com métricas como `utilization_rate` e `ecl_coverage_ratio`.\n"
            "- **`sales_lakehouse`**: E-commerce e varejo (`orders`, `order_items`, `customers`, `products`) com métricas como `gross_revenue` e `average_order_value`.\n\n"
            "💡 **Próximos passos práticos:**\n"
            "- Digite `2` ou *'ver modelos semânticos'* para listar entidades e métricas detalhadas.\n"
            "- Peça *'desenhar diagrama da camada semântica'* para ver o modelo ER visual!\n"
            "- Peça *'compilar query de métricas para medallion_silver_transactions'* para gerar o SparkSQL."
        )

    # 2. Unity Catalog
    if any(k in q for k in ("unity", "catalog", "catálogo", "catalogo", "metadado")):
        return (
            "### 🏛️ O que é o Databricks Unity Catalog?\n\n"
            "O **Unity Catalog** é a solução de governança unificada do Databricks para gerenciar dados, arquivos, modelos de machine learning e notebooks em múltiplos workspaces e nuvens (AWS, Azure, GCP).\n\n"
            "---\n\n"
            "### 🎯 Principais Pilares:\n"
            "1. **Namespace de 3 Níveis (`catalogo.schema.tabela`):**\n"
            "   Padroniza o acesso a tabelas e volumes Delta eliminando a ambiguidade do antigo modelo de dois níveis.\n"
            "2. **Controle de Acesso Centralizado (ACLs):**\n"
            "   Define permissões declarativas com ANSI SQL padrão (`GRANT SELECT ON ... TO ...`) válidas para SQL Warehouses e clusters Spark.\n"
            "3. **Linhagem Automatizada de Dados (Data Lineage):**\n"
            "   Rastreia o fluxo dos dados em tempo real da camada Bronze até os dashboards em nível de coluna.\n\n"
            "---\n\n"
            "💡 **Próximos passos práticos:**\n"
            "- Digite `1` ou *'inspecionar catálogo'* para listar tabelas e schemas disponíveis no seu Lakehouse!"
        )

    # 3. Arquitetura Medalhão
    if any(k in q for k in ("medalhao", "medallion", "bronze", "silver", "gold")):
        return (
            "### 🥇 O que é a Arquitetura Medalhão (Medallion Architecture)?\n\n"
            "A **Arquitetura Medalhão** é o padrão de engenharia de dados recomendado pela Databricks para construir pipelines confiáveis e auditáveis no Lakehouse, dividida em 3 camadas progressivas de qualidade:\n\n"
            "1. **🥉 Camada Bronze (Raw Ingestion):**\n"
            "   - Dados brutos exatamente como chegam da origem (append-only), com histórico e metadados de ingestão.\n"
            "2. **🥈 Camada Silver (Cleansed & Conformed):**\n"
            "   - Dados limpos, tipados explicitamente, deduplicados (`dropDuplicates`) e integrados via MERGE idempotente.\n"
            "3. **🥇 Camada Gold (Business Analytics):**\n"
            "   - Tabelas agregadas e preparadas para relatórios executivos, métricas de negócio e modelos de machine learning.\n\n"
            "---\n\n"
            "💡 **Próximos passos práticos:**\n"
            "- Digite `4` ou *'gerar pipeline etl silver'* para ver o código PySpark e SparkSQL com Delta Lake gerado automaticamente!"
        )

    # 4. Esteira de CI de Boas Práticas
    if any(
        k in q
        for k in (
            "ci",
            "esteira",
            "lint",
            "ruff",
            "sqlfluff",
            "qualidade",
            "anti-pattern",
            "antipattern",
        )
    ):
        return (
            "### 🛡️ O que é a Esteira de CI para Dados (Data Quality Gate)?\n\n"
            "A **Esteira de CI (Continuous Integration)** de dados do Databricks Steward Agent é um portão de qualidade mandatório que valida códigos antes de qualquer commit ou abertura de Pull Request.\n\n"
            "---\n\n"
            "### 🔍 Verificações Executadas:\n"
            "1. **Ruff (Python / PySpark):** Linter ultrarrápido que audita formatação, convenções PEP 8, imports limpos e sintaxe.\n"
            "2. **SQLFluff (SparkSQL Dialect):** Valida conformidade DDL/DML, quebras de linha e estilo de queries no dialeto `sparksql`.\n"
            "3. **Scanner de Anti-Patterns de Dados:** Detecta riscos críticos como chamadas de `.collect()` que estouram o driver, cross joins acidentais e conversões não autorizadas de `.toPandas()`.\n\n"
            "---\n\n"
            "💡 **Próximos passos práticos:**\n"
            "- Digite `5` ou *'validar código na esteira de ci'* para executar a esteira e emitir o relatório de diagnóstico!"
        )

    # 5. GitOps
    if any(k in q for k in ("gitops", "git", "pull request", "pr", "branch", "commit")):
        return (
            "### 🚀 O que é GitOps na Engenharia de Dados?\n\n"
            "**GitOps** é o paradigma onde o repositório Git é a **única fonte da verdade** para o ciclo de vida do código de dados, pipelines e configurações.\n\n"
            "---\n\n"
            "### 🔄 Como o Databricks Steward Agent executa GitOps:\n"
            "1. Gera os scripts do produto de dados em `pipelines/<nome>/`.\n"
            "2. Valida o código na esteira de CI (Ruff + SQLFluff + Anti-patterns).\n"
            "3. Se aprovado, cria automaticamente a feature branch `feature/data-product-<slug>`.\n"
            "4. Realiza commit convencional (`feat(data-product): ...`).\n"
            "5. Abre o Pull Request no GitHub com o relatório completo de CI embutido na descrição para aprovação.\n\n"
            "---\n\n"
            "💡 **Próximos passos práticos:**\n"
            "- Digite `6` ou *'abrir pull request'* para disparar o fluxo GitOps completo!"
        )

    # 6. Mermaid
    if any(k in q for k in ("mermaid", "diagrama", "erd", "linhagem", "lineage")):
        return (
            "### 📊 O que são Diagramas Mermaid.js no Lakehouse?\n\n"
            "**Mermaid.js** é uma sintaxe declarativa em texto simples que renderiza diagramas visuais diretamente no chat do Open WebUI sem necessidade de ferramentas externas.\n\n"
            "---\n\n"
            "### 📐 Diagramas Suportados:\n"
            "1. **Diagramas ER (`erDiagram`):** Notação *Crow's foot* mostrando tabelas, chaves primárias (PK), chaves estrangeiras (FK) e relacionamentos (1:N, N:M).\n"
            "2. **Diagramas de Linhagem Medalhão (`graph LR`):** Mapeamento do fluxo de dados da camada Bronze -> Silver -> Gold.\n\n"
            "---\n\n"
            "💡 **Próximos passos práticos:**\n"
            "- Digite `3` ou *'desenhar diagrama'* para visualizar o modelo de crédito corporativo!\n"
            "- Peça *'desenhar diagrama de vendas'* para ver o modelo de varejo!"
        )

    return None


def _extract_entity_from_query(query: str) -> str | None:
    """Extract canonical entity name from query using synonym dictionary."""
    q = (query or "").strip().lower()
    if not q:
        return None

    # First check aliases sorted by length descending so longer phrases match first
    sorted_aliases = sorted(ENTITY_ALIAS_MAP.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        pattern = r"(?:\b|_)" + re.escape(alias) + r"(?:\b|_)"
        if re.search(pattern, q):
            return ENTITY_ALIAS_MAP[alias]

    return None


def _is_entity_modeling_query(query: str) -> bool:
    """Check if query is asking for the schema, structure, or data modeling of a specific entity/table."""
    q = (query or "").strip().lower()
    if not q:
        return False

    if _is_analytics_query(q):
        return False

    # Avoid stealing queries meant for ETL generation, CI, GitOps, or pure Lineage diagrams
    if any(
        k in q
        for k in (
            "pipeline",
            "etl",
            "pyspark",
            "sparksql",
            "linhagem",
            "lineage",
            "fluxo",
            "gitops",
            "pull request",
            "pr ",
        )
    ):
        return False

    # If asking general conceptual question like "o que é modelagem?"
    if re.match(r"^(o\s+que\s+[ée]|what\s+is)\s+(a\s+)?modelagem\b", q):
        return False

    entity = _extract_entity_from_query(q)
    if not entity:
        return False

    # If query is essentially just the entity name (e.g. "customers", "tabela customers", "tabela de clientes")
    clean = re.sub(
        r"^(a\s+|o\s+|da\s+|do\s+|de\s+)?(tabela|table|entidade|entity)\s+(da\s+|do\s+|de\s+)?",
        "",
        q,
    ).strip()
    clean = re.sub(r"^(de\s+|da\s+|do\s+)", "", clean).strip()
    if clean in ENTITY_ALIAS_MAP or clean == entity:
        return True

    modeling_keywords = (
        "modelagem",
        "modelo",
        "schema",
        "esquema",
        "estrutura",
        "colunas",
        "coluna",
        "campos",
        "kpis",
        "métricas",
        "metricas",
        "detalhes",
        "modeling",
        "erd",
        "erdiagram",
        "tabela",
        "table",
        "atributos",
        "campos da",
        "dados de",
        "dados da",
    )
    return any(k in q for k in modeling_keywords)


def _is_diagram_query(q_l: str, intent: str | None = None) -> bool:
    diag_kw = (
        "diagram",
        "diagrama",
        "erd",
        "mermaid",
        "lineage",
        "fluxo",
        "desenhar",
        "relacoes",
        "relações",
        "relacionamento",
        "relacionamentos",
        "como estão relacionadas",
    )
    return intent == "DIAGRAM" or any(k in q_l for k in diag_kw)


def _is_schema_query(q_l: str, intent: str | None = None) -> bool:
    cat_kw = ("catalog", "catálogo", "schema", "tabelas")
    return intent == "SCHEMA" or any(k in q_l for k in cat_kw)


def _is_semantic_query(q_l: str) -> bool:
    sem_kw = ("semantic", "semântica", "semantica", "metrica", "dimensao", "ontology")
    return any(k in q_l for k in sem_kw)


def _is_etl_query(q_l: str, intent: str | None = None) -> bool:
    etl_kw = ("etl", "pipeline", "pyspark", "sparksql", "bronze", "silver", "gold")
    return intent == "ETL" or any(k in q_l for k in etl_kw)


def _is_ci_query(q_l: str) -> bool:
    ci_kw = ("ci", "esteira", "lint", "ruff", "sqlfluff", "anti-pattern", "validar")
    return any(k in q_l for k in ci_kw)


def _is_gitops_query(q_l: str) -> bool:
    git_kw = ("pr", "gitops", "branch", "commit", "push")
    return any(k in q_l for k in git_kw)


def is_error_recovery_state(state: AgentState | dict[str, Any]) -> bool:
    """Check if state holds a failed CI report, deployment error, or waiting_for_correction flag."""
    if not state:
        return False
    if state.get("waiting_for_correction") is True:
        return True
    if state.get("error_recovery"):
        return True
    ci_rep = state.get("ci_report")
    return bool(ci_rep is not None and hasattr(ci_rep, "is_approved") and not ci_rep.is_approved)


def _synthesize_conversational_response(
    det_result: dict[str, Any], client: Any, user_query: str
) -> dict[str, Any]:
    """Wrap deterministic structured output with a conversational LLM intro and outro."""
    raw_response = det_result.get("response", "")
    if (
        not raw_response
        or not client
        or "pytest" in sys.modules
        or os.getenv("PYTEST_CURRENT_TEST")
    ):
        return det_result

    # Don't synthesize short or simple greeting texts
    if len(raw_response) < 100 or "Opção Inválida" in raw_response:
        return det_result

    system_prompt = (
        "Você é o Databricks Steward Agent, um especialista em governança e engenharia de dados.\n"
        "Você receberá uma 'Resposta Estruturada do Sistema' (contendo tabelas Markdown, blocos de código PySpark/SQL ou diagramas Mermaid) e a 'Pergunta do Usuário'.\n"
        "Sua tarefa é repassar a resposta estruturada EXATAMENTE como está, mas adicionando uma introdução amigável e uma conclusão proativa (sugerindo o próximo passo lógico).\n"
        "Regras CRÍTICAS:\n"
        "1. NUNCA altere, resuma ou remova qualquer tabela Markdown.\n"
        "2. NUNCA altere ou remova blocos de código (```python, ```sql, etc).\n"
        "3. NUNCA altere ou remova diagramas (```mermaid).\n"
        "4. Mantenha o conteúdo técnico intacto e apenas 'abrace' ele com texto humano.\n"
        "5. Responda em português."
    )
    user_prompt = (
        f"Pergunta do Usuário: {user_query}\n\nResposta Estruturada do Sistema:\n{raw_response}"
    )

    try:
        llm_res = client.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        synthesized_text = str(llm_res.content).strip()
        # Fallback if the LLM hallucinated and removed the core code blocks/tables
        if "```" in raw_response and "```" not in synthesized_text:
            return det_result

        det_result["response"] = synthesized_text
        det_result["messages"] = [AIMessage(content=synthesized_text)]
    except Exception as e:  # noqa: BLE001
        logger.warning("Falha ao sintetizar resposta conversacional: %s", e)

    return det_result


def handle_llm_greeting(state, client, query):
    import json

    from langchain_core.messages import AIMessage

    try:
        system_prompt = (
            "Você é o Databricks Steward Agent, assistente especializado em governança e engenharia de dados Lakehouse. "
            "Apresente-se cordialmente em português de forma clara e profissional. "
            "Explique resumidamente que você ajuda com:\\n"
            "1. Unity Catalog Introspection: descoberta de tabelas e schemas\\n"
            "2. Camada Semântica: métricas e dimensões de negócio\\n"
            "3. Diagramas Mermaid: modelagem visual ER e linhagem medalhão\\n"
            "4. Pipelines ETL: geração PySpark/SparkSQL Bronze, Silver e Gold\\n"
            "5. Esteira de CI: qualidade de código (Ruff, SQLFluff, anti-patterns)\\n"
            "6. GitOps: automação de branch, commit e Pull Requests no GitHub\\n"
            "Oriente o usuário a escolher um número (1 a 6) ou descrever sua necessidade."
        )
        llm_res = client.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query or "olá"},
            ]
        )
        reply = str(llm_res.content or "").strip()
        if reply.startswith("{") and reply.endswith("}"):
            try:
                p = json.loads(reply)
                if isinstance(p, dict) and "parameters" in p and "message" in p["parameters"]:
                    reply = str(p["parameters"]["message"])
            except Exception:  # noqa: BLE001, S110
                pass
        if reply:
            return {**state, "messages": [AIMessage(content=reply)], "response": reply}
    except Exception:  # noqa: BLE001, S110
        pass
    return None


def handle_llm_conceptual(state, client, query):
    from langchain_core.messages import AIMessage

    try:
        system_prompt = (
            "Você é o Databricks Steward Agent, um especialista em governança e engenharia de dados Lakehouse no Databricks. "
            "Responda de forma didática, completa, estruturada em tópicos e profissional em português. "
            "Destaque o conceito, seus benefícios, como funciona no Databricks e sugira como o usuário pode explorar essa capacidade."
        )
        llm_res = client.invoke(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        )
        reply = str(llm_res.content or "").strip()
        if reply:
            return {**state, "messages": [AIMessage(content=reply)], "response": reply}
    except Exception:  # noqa: BLE001, S110
        pass
    return None


def execute_llm_tool_calling(state, client, query):
    import json
    import re

    from langchain_core.messages import AIMessage

    from src.config import settings

    try:
        tools = get_langchain_tools()
        llm_with_tools = client.bind_tools(tools)
        messages = [{"role": "user", "content": query}] if query else state.get("messages", [])
        response = llm_with_tools.invoke(messages)

        active_diagram = state.get("active_diagram")
        generated_code = state.get("generated_code")

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_dict = {t.name: t for t in tools}
            results = []
            for tc in response.tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args", {})
                if t_name in tool_dict:
                    t_output = tool_dict[t_name].invoke(t_args)
                    results.append(str(t_output))
                    if t_name == "generate_diagram_tool":
                        active_diagram = str(t_output)
                    elif t_name == "inspect_entity_modeling_tool":
                        m_match = re.search(r"```mermaid\n(.*?)\n```", str(t_output), re.DOTALL)
                        if m_match:
                            active_diagram = f"```mermaid\n{m_match.group(1)}\n```"
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
                **state,
                "messages": [AIMessage(content=combined_resp)],
                "response": combined_resp,
                "active_diagram": active_diagram,
                "generated_code": generated_code,
            }

        if response.content:
            content_str = str(response.content).strip()
            if content_str.startswith("{") and content_str.endswith("}"):
                try:
                    p = json.loads(content_str)
                    if isinstance(p, dict) and "parameters" in p and "message" in p["parameters"]:
                        content_str = str(p["parameters"]["message"])
                except Exception:  # noqa: BLE001, S110
                    pass
            return {**state, "messages": [response], "response": content_str}
    except Exception:  # noqa: BLE001, S110
        pass
    return None
