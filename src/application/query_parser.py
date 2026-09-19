import re
from typing import Any


def extract_query_text(messages: list[Any]) -> str:
    if not messages:
        return ""
    last_msg = messages[-1]
    return str(getattr(last_msg, "content", "") or "").strip()


def extract_table_or_entity(query: str) -> str:
    q = (query or "").strip().lower()
    match = re.search(r"\b(?:tabela|dados de|dados da|de|da|tabela de)\s+([a-zA-Z0-9_.]+)\b", q)
    if match:
        return match.group(1)
    words = q.split()
    for i, w in enumerate(words):
        if w in ("select", "from") and i + 1 < len(words):
            return words[i + 1]
    return ""


def extract_entity_from_query(query: str) -> str | None:
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


def resolve_anaphoric_entity(query: str, messages: list[Any], state: dict[str, Any]) -> str | None:
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

            if getattr(msg, "type", "") == "human":
                from src.application.intent_router import is_data_preview_query

                if is_data_preview_query(content):
                    t = extract_table_or_entity(content)
                    if t and t != "medallion_silver_transactions":
                        return t.split(".")[-1]
    return None


def extract_pending_from_history(messages: list[Any]) -> dict[str, Any]:
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
