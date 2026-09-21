import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def _extract_query_text(messages: list[Any]) -> str:
    if not messages:
        return ""
    last_msg = messages[-1]
    if isinstance(last_msg, dict):
        return last_msg.get("content", "") or ""
    if hasattr(last_msg, "content"):
        return str(last_msg.content or "")
    return str(last_msg or "")


def _is_greeting(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"^(ol[áa]|hello|hi|oi|oie|bom dia|boa tarde|boa noite|opa|e a[íi])\b",
        r"\b(quem\s+[ée]\s+voc[eê]|o\s+que\s+voc[eê]\s+faz)\b",
        r"\b(ajuda|help|socorro|menu)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _is_title_request(query: str) -> bool:
    q = (query or "").strip().lower()
    return bool(re.search(r"^(title|título|gerar t[ií]tulo)", q))


def _is_data_preview_query(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"\b(consultar|consulte|quero ver|ver|mostrar|mostre|trazer|traga|exibir|exiba|ler|leia)\s+(os\s+)?(dados|registros|linhas)\b",
        r"\b(amostra|preview)\s+(de\s+|dos\s+|da\s+)?(dados|tabela)\b",
        r"\b(select\s+\*\s+from)\b",
        r"\b(puxa|puxe)\s+(uns\s+)?(\d+\s+)?(registros|linhas)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _is_confirmation(query: str) -> bool:
    q = (query or "").strip().lower()
    if re.search(r"\b(não|nao|nem|nunca|jamais|cancelar|cancela)\b", q):
        return False
    patterns = [
        r"\b(sim|confirmar|confirmo|aprovar|aprovo|pode executar|executa|executar|confirmado|prosseguir|ok|yes|positivo|autorizado|pode rodar|pode seguir|manda bala|pode fazer|vamos lá|bora)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _classify_intent_with_llm(query: str, llm: ChatOpenAI) -> str:
    q = (query or "").strip()
    if not q:
        return "OTHER"
    if _is_confirmation(query):
        return "CONFIRM"
    if _is_data_preview_query(query):
        return "PREVIEW"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a smart semantic router for a Databricks AI Assistant...
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
        for tag in ["PREVIEW", "SCHEMA", "DIAGRAM", "ETL", "CONFIRM", "GREETING", "OTHER"]:
            if tag in content:
                return tag
        return "OTHER"
    except Exception:  # noqa: BLE001
        return "OTHER"


def _extract_table_or_entity(query: str) -> str:
    q = (query or "").strip().lower()
    match = re.search(r"\b(?:tabela|dados de|dados da|de|da|tabela de)\s+([a-zA-Z0-9_.]+)\b", q)
    if match:
        return match.group(1)
    words = q.split()
    for i, w in enumerate(words):
        if w in ("select", "from") and i + 1 < len(words):
            return words[i + 1]
    return ""


def _resolve_anaphoric_entity(query: str, messages: list[Any], state: dict[str, Any]) -> str | None:
    q = (query or "").strip().lower()
    anaphoric_refs = [
        "dessa tabela",
        "da tabela",
        "dela",
        "dessa",
        "desse",
        "dele",
        "disso",
        "essa tabela",
    ]
    is_anaphoric = any(ref in q for ref in anaphoric_refs) or "tabela" in q

    if is_anaphoric:
        if state.get("pending_pipeline"):
            return state["pending_pipeline"].get("source_entity")
        for msg in reversed(messages[:-1]):
            content = str(getattr(msg, "content", "") or "").lower()
            if getattr(msg, "type", "") != "human":
                match_hint = re.search(r"tabela\s+lakehouse:\s*`([^`]+)`", content)
                if match_hint:
                    return match_hint.group(1).split(".")[-1]
                match_hint = re.search(
                    r"generated medallion pipeline:\s*([^\s\(]+)", content, re.IGNORECASE
                )
                if match_hint:
                    return match_hint.group(1)
            if getattr(msg, "type", "") == "human" and _is_data_preview_query(content):
                t = _extract_table_or_entity(content)
                if t and t != "medallion_silver_transactions":
                    return t.split(".")[-1]
    return None


def _extract_pending_from_history(messages: list[Any]) -> dict[str, Any]:
    pending = {}
    for msg in reversed(messages):
        content = str(getattr(msg, "content", "") or "")
        if "Generated Medallion Pipeline" in content and "Deseja confirmar e disparar" in content:
            m_prod = re.search(r"Generated Medallion Pipeline:\s*([^\s\(]+)", content)
            if m_prod:
                pending["product_name"] = m_prod.group(1).strip()
            m_py = re.search(r"#### PySpark Pipeline\s*```python\n(.*?)```", content, re.DOTALL)
            if m_py:
                pending["pyspark"] = m_py.group(1).strip()
            m_sql = re.search(
                r"#### SparkSQL DDL & Ingestion\s*```sql\n(.*?)```", content, re.DOTALL
            )
            if m_sql:
                pending["sparksql"] = m_sql.group(1).strip()
            break
    return pending


def _is_conceptual_question(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"^(o\s+que\s+[ée]|qual\s+o\s+conceito|o\s+que\s+significa|me\s+explica)\b",
        r"\b(como\s+funciona|pra\s+que\s+serve|por\s+que\s+usar)\b",
        r"\b(conceito\s+de|vantagens\s+de|diferen[çc]a\s+entre)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _get_conceptual_explanation(query: str) -> str | None:
    q = (query or "").strip().lower()
    if "unity catalog" in q:
        return "### 📘 Conceito: Unity Catalog\n..."
    if "camada sem" in q or "semantic" in q:
        return "### 📘 Conceito: Camada Semântica (Semantic Layer)\n..."
    if "medallion" in q or "medalha" in q:
        return "### 📘 Conceito: Arquitetura Medallion\n..."
    if "gitops" in q or "ci/cd" in q or "ci cd" in q:
        return "### 📘 Conceito: GitOps e CI no Lakehouse\n..."
    return None


def _extract_entity_from_query(query: str) -> str | None:
    q = (query or "").strip().lower()
    patterns = [
        r"\b(?:tabela|entidade|camada|sobre|para|da|de|das|dos)\s+([a-zA-Z0-9_]+)\b",
        r"\b([a-zA-Z0-9_]+)\b",
    ]
    ignore_words = {
        "tabela",
        "entidade",
        "dados",
        "schema",
        "modelagem",
        "diagrama",
        "mermaid",
        "pipeline",
        "etl",
        "bronze",
        "silver",
        "gold",
        "uma",
        "um",
        "as",
        "os",
        "de",
        "da",
        "para",
        "sobre",
        "qual",
    }
    for p in patterns:
        matches = re.finditer(p, q)
        for m in matches:
            word = m.group(1)
            if word not in ignore_words and len(word) > 2:
                return word
    return None


def _is_entity_modeling_query(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"\b(qual( \w+)* modelagem|modelagem de)\b",
        r"\b(schema da tabela|colunas da tabela|estrutura da tabela)\b",
        r"\b(mostre( \w+)* schema|mostrar( \w+)* schema)\b",
        r"\b(quais( \w+)* colunas|quais( \w+)* campos)\b",
    ]
    return any(re.search(p, q) for p in patterns)
