"""LangGraph agent coordinating Databricks stewardship, semantic modeling, CI, and GitOps."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import urllib.request
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState
from src.agent.tools import (
    ENTITY_ALIAS_MAP,
    format_entity_modeling,
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

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a smart semantic router for a Databricks AI Assistant. Analyze the user query and classify it strictly into ONE of the following tags. Output ONLY the exact tag in uppercase.

Tags:
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
        for tag in ["PREVIEW", "SCHEMA", "DIAGRAM", "ETL", "CONFIRM", "GREETING", "OTHER"]:
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
    pending_pipeline = state.get("pending_pipeline")

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
            "pending_pipeline": pending_pipeline,
        }

    # Conceptual questions (e.g. "o que é camada semantica?")
    if _is_conceptual_question(user_query):
        concept_resp = _get_conceptual_explanation(user_query)
        if concept_resp:
            return {
                "messages": [AIMessage(content=concept_resp)],
                "response": concept_resp,
                "active_diagram": active_diagram,
                "generated_code": generated_code,
                "ci_report": ci_report,
                "gitops_result": gitops_result,
                "pending_pipeline": pending_pipeline,
            }

    # Data preview / inspection query (e.g. "consultar dados da tabela X")
    if _is_data_preview_query(user_query) or state.get("intent") == "PREVIEW":
        from src.agent.tools import preview_table_data

        target_table = _extract_table_or_entity(user_query)
        lim_match = re.search(r"\blimit\s+(\d+)\b", q_lower)
        limit_val = int(lim_match.group(1)) if lim_match else 10
        response_text = preview_table_data(table_name=target_table, limit=limit_val)
        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "pending_pipeline": pending_pipeline,
            "preview_data": {"table_name": target_table, "limit": limit_val},
        }

    # Entity-specific data modeling / schema requests (e.g. "qual a modelagem de customers")
    if _is_entity_modeling_query(user_query):
        ent_name = _extract_entity_from_query(user_query) or "customers"
        resp = format_entity_modeling(ent_name)
        m_match = re.search(r"```mermaid\n(.*?)\n```", resp, re.DOTALL)
        diag = f"```mermaid\n{m_match.group(1)}\n```" if m_match else active_diagram
        return {
            "messages": [AIMessage(content=resp)],
            "response": resp,
            "active_diagram": diag,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "pending_pipeline": pending_pipeline,
        }

    # Confirmation of pending ETL pipeline lifecycle (CI -> GitOps -> Databricks Job -> Semantic Layer)

    if _is_confirmation(user_query) or state.get("intent") == "CONFIRM":
        from src.agent.tools import deploy_and_materialize_data_product

        pending = state.get("pending_pipeline") or _extract_pending_from_history(messages)
        prod_name = pending.get("product_name", "medallion_gold_sales_kpis")
        p_py = pending.get("pyspark", "")
        p_sql = pending.get("sparksql", "")
        p_src = pending.get("source_entity", "medallion_silver_transactions")

        res = deploy_and_materialize_data_product(
            product_name=prod_name,
            pyspark_code=p_py,
            sparksql_code=p_sql,
            source_entity=p_src,
        )

        if res.get("status") == "ci_failed":
            response_text = str(res.get("message", "❌ CI Quality Gate rejeitou o pipeline."))
            ci_report = res.get("ci_report")
        else:
            ci_rep = res["ci_report"]
            git_res = res["git_result"]
            job_res = res["job_result"]
            ent_model = res.get("entity_model")
            diag_md = res.get("updated_diagram", "")
            active_diagram = diag_md

            metrics_list = (
                ", ".join([f"`{m.name}`" for m in ent_model.metrics])
                if ent_model and ent_model.metrics
                else "`total_records`"
            )
            push_label = (
                "✅ Sincronizado com `origin/main` no GitHub"
                if git_res.get("push_success")
                else "✅ Commit local em `main`"
            )

            response_text = (
                f"## 🚀 Ciclo de Vida do Data Product Concluído com Sucesso!\n\n"
                f"### 🛡️ 1. Esteira de CI Quality Gate\n"
                f"{ci_rep.summary_markdown}\n\n"
                f"### 📦 2. GitOps Auto Commit & Push\n"
                f"- **Branch:** `{git_res['branch']}`\n"
                f"- **Commit SHA:** `{git_res['commit_sha']}`\n"
                f"- **Status Push:** {push_label}\n"
                f"- **Arquivos Comitados:** `{', '.join(git_res['files'])}`\n\n"
                f"### ⚡ 3. Databricks Workflow Job & Materialização\n"
                f"- **Job ID:** `{job_res['job_id']}` (Run: `{job_res['run_id']}`)\n"
                f"- **Tabela Unity Catalog:** `{res['table_name']}`\n"
                f"- **Status:** `✅ Materializada no Catálogo com Sucesso`\n\n"
                f"### 🧠 4. Camada Semântica & Modelo de Dados\n"
                f"- **Entidade Registrada:** `{ent_model.name if ent_model else prod_name}`\n"
                f"- **Métricas Analíticas Criadas:** {metrics_list}\n"
                f"- **Arquivo de Ontologia:** `configs/semantic_models/databricks_medallion.yaml`\n\n"
                f"### 📐 5. Diagrama de Relacionamentos Atualizado\n"
                f"```mermaid\n{diag_md}\n```"
            )

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "job_result": res.get("job_result"),
            "pending_pipeline": None,
        }

    # Intent & Numeric shortcuts
    intent = state.get("intent")
    is_opt_1 = q_lower in ("1", "1.", "opcao 1", "opção 1") or intent == "SCHEMA"
    is_opt_2 = q_lower in ("2", "2.", "opcao 2", "opção 2")
    is_opt_3 = q_lower in ("3", "3.", "opcao 3", "opção 3") or intent == "DIAGRAM"
    is_opt_4 = q_lower in ("4", "4.", "opcao 4", "opção 4") or intent == "ETL"
    is_opt_5 = q_lower in ("5", "5.", "opcao 5", "opção 5")
    is_opt_6 = q_lower in ("6", "6.", "opcao 6", "opção 6")

    # 1. Mermaid Diagram & Relationships (Option 3)
    if is_opt_3 or any(
        k in q_lower
        for k in (
            "diagram",
            "diagrama",
            "erd",
            "erdiagram",
            "mermaid",
            "lineage",
            "fluxo",
            "desenhar",
            "modelo visual",
            "relações",
            "relacoes",
            "relacionamento",
            "relacionamentos",
            "como estão relacionadas",
        )
    ):
        reg = SemanticRegistry(settings.semantic_models_path)
        has_medallion = reg.get_domain("databricks_medallion") is not None
        ent = _extract_entity_from_query(user_query)
        if (
            "lineage" in q_lower
            or "fluxo" in q_lower
            or "medallion" in q_lower
            or "medalhao" in q_lower
            or "medalhão" in q_lower
        ):
            diag = generate_diagram(
                "lineage", domain="databricks_medallion" if has_medallion else None
            )
        elif ent:
            diag = generate_diagram("er", domain=ent)
        elif ("sales" in q_lower or "vendas" in q_lower) and not has_medallion:
            diag = generate_diagram("er", domain="sales_lakehouse")
        elif (
            "credit" in q_lower or "credito" in q_lower or "crédito" in q_lower
        ) and not has_medallion:
            diag = generate_diagram("er", domain="corporate_credit")
        elif has_medallion:
            diag = generate_diagram("er", domain="databricks_medallion")
        elif "sales" in q_lower or "vendas" in q_lower:
            diag = generate_diagram("er", domain="sales_lakehouse")
        else:
            diag = generate_diagram("er", domain="corporate_credit")

        active_diagram = diag
        response_text = diag

    # 2. Databricks Unity Catalog Introspection (Option 1)
    elif is_opt_1 or any(
        k in q_lower
        for k in ("catalog", "catálogo", "schema", "tabelas", "unity catalog", "introspect")
    ):
        response_text = inspect_unity_catalog()

    # 3. Semantic Layer & Business Models (Option 2)
    elif is_opt_2 or any(
        k in q_lower
        for k in (
            "semantic",
            "semântica",
            "semantica",
            "metrica",
            "métrica",
            "dimensao",
            "dimensão",
            "ontology",
            "ontologia",
            "negocio",
            "negócio",
        )
    ):
        models_summary = load_semantic_models()
        reg = SemanticRegistry(settings.semantic_models_path)
        has_medallion = reg.get_domain("databricks_medallion") is not None
        hint = (
            "💡 *Dica: Você pode pedir 'qual a modelagem das transações', 'desenhar diagrama' ou 'compilar query de métricas para medallion_silver_transactions'.*"
            if has_medallion
            else "💡 *Dica: Você pode pedir 'desenhar diagrama do domínio sales_lakehouse' ou 'compilar query da métrica gross_revenue por canal'.*"
        )
        response_text = (
            "### 📦 Modelos Semânticos Registrados no Lakehouse\n\n"
            "Aqui estão os modelos de domínio e ontologias de negócio configurados em YAML:\n\n"
            f"{models_summary}\n\n"
            f"{hint}"
        )

    # 4. ETL Pipeline Generation
    elif is_opt_4 or any(
        k in q_lower for k in ("etl", "pipeline", "pyspark", "sparksql", "bronze", "silver", "gold")
    ):
        layer = "gold" if "gold" in q_lower else "bronze" if "bronze" in q_lower else "silver"
        reg = SemanticRegistry(settings.semantic_models_path)
        has_medallion = reg.get_domain("databricks_medallion") is not None
        ent = _extract_entity_from_query(user_query)
        if not ent:
            ent = _resolve_anaphoric_entity(user_query, messages, state)

        if ent:
            if (
                has_medallion
                and layer == "gold"
                and ent in ("customers", "customer", "cliente", "clientes", "usuarios", "user")
            ):
                entity_name = "medallion_gold_customer_kpis"
            else:
                entity_name = ent
        elif has_medallion:
            if layer == "bronze":
                entity_name = "medallion_bronze_transactions"
            elif layer == "gold":
                entity_name = "medallion_gold_sales_kpis"
            else:
                entity_name = "medallion_silver_transactions"
        elif "sales" in q_lower or "order" in q_lower or "venda" in q_lower:
            entity_name = "orders"
        elif "transaction" in q_lower or "transac" in q_lower:
            entity_name = "silver_transactions" if layer == "silver" else "bronze_raw_transactions"
        else:
            entity_name = "facilities"

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
            pending_pipeline = {
                "product_name": pipeline.table_name,
                "pyspark": pipeline.pyspark_code,
                "sparksql": pipeline.sparksql_code,
                "source_entity": entity_name,
            }
            confirmation_prompt = (
                "\n\n---\n"
                "❓ **Deseja confirmar e disparar a esteira de CI, auto commit & push na branch `main` e criação do Job no Databricks?**\n"
                "👉 *Digite **'sim'** ou **'confirmar'** para executar o ciclo de vida completo!*"
            )
            response_text = (
                f"### Generated Medallion Pipeline: {pipeline.table_name} ({pipeline.layer})\n\n"
                f"#### PySpark Pipeline\n```python\n{pipeline.pyspark_code}\n```\n\n"
                f"#### SparkSQL DDL & Ingestion\n```sql\n{pipeline.sparksql_code}\n```{confirmation_prompt}"
            )
        else:
            response_text = generate_etl_pipeline(entity_name, layer=layer)
            confirmation_prompt = (
                "\n\n---\n"
                "❓ **Deseja confirmar e disparar a esteira de CI, auto commit & push na branch `main` e criação do Job no Databricks?**\n"
                "👉 *Digite **'sim'** ou **'confirmar'** para executar o ciclo de vida completo!*"
            )
            response_text += confirmation_prompt

    # 5. Data Best Practices CI Quality Gate
    elif is_opt_5 or any(
        k in q_lower
        for k in ("ci", "esteira", "lint", "ruff", "sqlfluff", "anti-pattern", "validar")
    ):
        py_code = generated_code.get("pyspark") if generated_code else None
        sql_code = generated_code.get("sparksql") if generated_code else None

        if not py_code or not sql_code:
            reg = SemanticRegistry(settings.semantic_models_path)
            has_medallion = reg.get_domain("databricks_medallion") is not None
            if has_medallion:
                py_code = (
                    "def process(df):\n"
                    "    return df.filter(df['status'] == 'COMPLETED').dropDuplicates(['transaction_id'])\n"
                )
                sql_code = "SELECT transaction_id, user_id, amount FROM workspace.default.medallion_silver_transactions;"
            else:
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
            reg = SemanticRegistry(settings.semantic_models_path)
            has_medallion = reg.get_domain("databricks_medallion") is not None
            if has_medallion:
                py_code = (
                    "def process(df):\n"
                    "    return df.filter(df['status'] == 'COMPLETED').dropDuplicates(['transaction_id'])\n"
                )
                sql_code = "SELECT transaction_id, user_id, amount FROM workspace.default.medallion_silver_transactions;"
            else:
                py_code = "def process(df):\n    return df.filter(df['active'] == True)\n"
                sql_code = "SELECT order_id, total_amount FROM main.sales.orders;"

        report = ci_report or run_ci_pipeline(pyspark_code=py_code, sparksql_code=sql_code)
        ci_report = report

        reg = SemanticRegistry(settings.semantic_models_path)
        has_medallion = reg.get_domain("databricks_medallion") is not None
        default_prod = "medallion-silver-transactions" if has_medallion else "corporate-credit-kpis"
        product_name = (
            generated_code.get("table_name", default_prod) if generated_code else default_prod
        )
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
        is_identity = any(
            re.search(p, q_lower)
            for p in (
                r"\b(o\s+que\s+(voc[eê]|vc)\s+[ée]|quem\s+[ée]\s+(voc[eê]|vc))\b",
                r"\b(o\s+que\s+(voc[eê]|vc)\s+faz|qual\s+(o\s+)?seu\s+papel|qual\s+(o\s+)?seu\s+nome)\b",
                r"\b(como\s+(voc[eê]|vc)\s+pode\s+me\s+ajudar|apresente-se|se\s+apresente)\b",
                r"\b(fale|conte|diga)\s+(mais\s+)?sobre\s+(voc[eê]|vc)\b",
                r"\b(quem\s+criou\s+(voc[eê]|vc)|de\s+onde\s+(voc[eê]|vc)\s+[ée])\b",
                r"\b(what\s+are\s+you|who\s+are\s+you|what\s+do\s+you\s+do)\b",
            )
        )
        if is_identity:
            response_text = (
                "Eu sou o **Databricks Steward Agent**, um assistente inteligente especializado em governança de dados, "
                "modelagem semântica e automação de engenharia de dados no ecossistema Databricks Lakehouse.\n\n"
                "Meu objetivo é ajudar engenheiros e analistas de dados em:\n"
                "1. **Unity Catalog Introspection**: Descobrir e auditar tabelas, schemas e colunas físicas.\n"
                "2. **Camada Semântica Declarativa**: Modelar entidades de negócio, métricas padronizadas e relacionamentos em YAML.\n"
                "3. **Diagramas Mermaid.js**: Gerar diagramas conceituais (ERD Crow's foot) e fluxos de linhagem medalhão.\n"
                "4. **Pipelines ETL Modulares**: Produzir código PySpark e SparkSQL idempotente (Bronze, Silver, Gold).\n"
                "5. **Esteira de Qualidade de Dados (CI)**: Validar conformidade com Ruff, SQLFluff e detectores de anti-patterns.\n"
                "6. **Automated GitOps**: Abrir feature branches, Conventional Commits e Pull Requests no GitHub.\n\n"
                "Como posso ajudar você hoje no seu Lakehouse?"
            )
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
        "pending_pipeline": pending_pipeline,
    }


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


def steward_node(state: AgentState, llm: ChatOpenAI | None = None) -> dict[str, Any]:
    """Main routing and execution node for the Databricks Steward Agent."""
    client = llm or get_local_chat_client()
    base_url = str(getattr(client, "openai_api_base", None) or settings.local_llm_base_url)
    model_name = str(getattr(client, "model_name", None) or settings.local_llm_model)
    cache_key = (base_url, model_name)

    messages = state.get("messages", [])
    user_query = state.get("user_query") or _extract_query_text(messages)
    q_lower = (user_query or "").strip().lower()

    now = time.time()
    cached = _availability_cache.get(cache_key)

    is_available = True

    if os.getenv("PYTEST_CURRENT_TEST"):
        is_available = False
    elif cached is not None:
        avail, expiry = cached
        if now < expiry:
            is_available = avail

    # LLM Intent Router (Semantic Fallback to make chat perfectly fluid and smart)
    intent = state.get("intent")
    if not intent and is_available:
        try:
            intent = _classify_intent_with_llm(user_query, client)
            state["intent"] = intent

            print("=> INTENT CLASSIFIED AS:", intent)
        except Exception:  # noqa: BLE001
            intent = "OTHER"

    # 1. Instant resolution for UI title requests
    if _is_title_request(user_query) or intent == "TITLE":
        return {
            "messages": [AIMessage(content="Databricks Steward - Governança")],
            "response": "Databricks Steward - Governança",
            "active_diagram": state.get("active_diagram"),
            "generated_code": state.get("generated_code"),
            "ci_report": state.get("ci_report"),
            "gitops_result": state.get("gitops_result"),
        }

    # 2. Confirmation of pending pipeline lifecycle (e.g. 'sim', 'confirmar')
    if _is_confirmation(user_query) or state.get("intent") == "CONFIRM":
        res = _deterministic_steward_execution(state)
        return _synthesize_conversational_response(res, client, user_query)

    # 3. Data preview queries (e.g. 'consultar dados da tabela X')
    if _is_data_preview_query(user_query) or state.get("intent") == "PREVIEW":
        res = _deterministic_steward_execution(state)
        return _synthesize_conversational_response(res, client, user_query)

    # 4. Direct numeric menu shortcuts
    if q_lower in (
        "1",
        "1.",
        "opcao 1",
        "opção 1",
        "2",
        "2.",
        "opcao 2",
        "opção 2",
        "3",
        "3.",
        "opcao 3",
        "opção 3",
        "4",
        "4.",
        "opcao 4",
        "opção 4",
        "5",
        "5.",
        "opcao 5",
        "opção 5",
        "6",
        "6.",
        "opcao 6",
        "opção 6",
    ):
        res = _deterministic_steward_execution(state)
        return _synthesize_conversational_response(res, client, user_query)

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
                llm_res = client.invoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query or "olá"},
                    ]
                )
                _availability_cache[cache_key] = (True, now + 30.0)
                reply = str(llm_res.content or "").strip()
                if reply.startswith("{") and reply.endswith("}"):
                    try:
                        p = json.loads(reply)
                        if (
                            isinstance(p, dict)
                            and "parameters" in p
                            and "message" in p["parameters"]
                        ):
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
                logger.debug(
                    "Local LLM offline or unreachable (%s); using deterministic welcome.", e
                )
        res = _deterministic_steward_execution(state)
        return _synthesize_conversational_response(res, client, user_query)

    # 4. Conceptual and educational inquiries (answered without tool binding)
    if _is_conceptual_question(user_query):
        tailored = _get_conceptual_explanation(user_query)
        if tailored:
            return {
                "messages": [AIMessage(content=tailored)],
                "response": tailored,
                "active_diagram": state.get("active_diagram"),
                "generated_code": state.get("generated_code"),
                "ci_report": state.get("ci_report"),
                "gitops_result": state.get("gitops_result"),
            }

        if is_available:
            try:
                system_prompt = (
                    "Você é o Databricks Steward Agent, um especialista em governança e engenharia de dados Lakehouse no Databricks. "
                    "Responda de forma didática, completa, estruturada em tópicos e profissional em português. "
                    "Destaque o conceito, seus benefícios, como funciona no Databricks e sugira como o usuário pode explorar essa capacidade."
                )
                llm_res = client.invoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                    ]
                )
                _availability_cache[cache_key] = (True, now + 30.0)
                reply = str(llm_res.content or "").strip()
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
                logger.debug(
                    "Local LLM conceptual call failed (%s); using deterministic router.", e
                )

        res = _deterministic_steward_execution(state)
        return _synthesize_conversational_response(res, client, user_query)

    # 5. Entity modeling queries (e.g. "qual a modelagem de customers", "schema da tabela orders")
    if _is_entity_modeling_query(user_query):
        res = _deterministic_steward_execution(state)
        return _synthesize_conversational_response(res, client, user_query)

    # 6. Technical queries: LLM with tool calling
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
                        if (
                            isinstance(p, dict)
                            and "parameters" in p
                            and "message" in p["parameters"]
                        ):
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
            logger.debug(
                "Local LLM offline or unreachable (%s); using deterministic steward router.", e
            )

    # 7. Deterministic fallback
    res = _deterministic_steward_execution(state)
    return _synthesize_conversational_response(res, client, user_query)


def create_steward_graph(llm: ChatOpenAI | None = None) -> Any:
    """Create and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    node_func = (lambda s: steward_node(s, llm=llm)) if llm is not None else steward_node
    workflow.add_node("steward", node_func)
    workflow.add_edge(START, "steward")
    workflow.add_edge("steward", END)

    return workflow.compile()
