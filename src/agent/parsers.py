import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PORTUGUESE_STOP_WORDS: set[str] = {
    "de",
    "da",
    "do",
    "das",
    "dos",
    "a",
    "o",
    "as",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "em",
    "na",
    "no",
    "nas",
    "nos",
    "para",
    "por",
    "com",
    "e",
    "ou",
    "se",
    "que",
    "como",
    "dessa",
    "desse",
    "dessas",
    "desses",
    "desta",
    "deste",
    "destas",
    "destos",
    "dela",
    "dele",
    "delas",
    "deles",
    "disso",
    "disto",
    "esta",
    "este",
    "essa",
    "esse",
    "estas",
    "estes",
    "essas",
    "esses",
    "mes",
    "mês",
    "passado",
    "passada",
    "me",
    "gera",
    "gerar",
    "propor",
    "puxe",
    "puxa",
    "sobre",
    "qual",
    "quais",
    "partir",
    "acima",
    "anterior",
    "mesma",
    "mesmo",
    "aqui",
    "ali",
    "crie",
    "cria",
    "criar",
    "criacao",
    "criação",
}

ENGLISH_STOP_WORDS: set[str] = {
    "the",
    "a",
    "an",
    "for",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "with",
    "by",
    "show",
    "get",
    "pull",
    "display",
    "view",
    "fetch",
    "select",
    "me",
    "please",
    "want",
    "would",
    "like",
    "create",
    "generate",
    "propose",
    "draw",
    "above",
    "same",
    "this",
    "that",
}

ALL_STOP_WORDS: set[str] = PORTUGUESE_STOP_WORDS | ENGLISH_STOP_WORDS

FILLER_KEYWORDS: set[str] = {
    "tabela",
    "table",
    "tables",
    "entidade",
    "entity",
    "entities",
    "dados",
    "data",
    "schema",
    "schemas",
    "modelagem",
    "modeling",
    "model",
    "models",
    "diagrama",
    "diagram",
    "mermaid",
    "pipeline",
    "pipelines",
    "etl",
    "bronze",
    "silver",
    "gold",
    "sobre",
    "para",
    "quero",
    "ver",
    "mostre",
    "mostrar",
    "consultar",
    "consulte",
    "exibir",
    "exiba",
    "trazer",
    "traga",
    "ler",
    "leia",
    "amostra",
    "sample",
    "preview",
    "registros",
    "records",
    "linhas",
    "rows",
    "select",
    "from",
    "limit",
    "structure",
    "estrutura",
    "colunas",
    "columns",
    "campos",
    "fields",
    "desenhar",
    "desenha",
    "desenho",
    "draw",
    "drawing",
}

ENTITY_SYNONYMS: dict[str, str] = {
    # Customers / Clientes
    "clientes": "customers",
    "cliente": "customers",
    "compradores": "customers",
    "buyers": "customers",
    "customer": "customers",
    "customers": "customers",
    # Transactions / Transações
    "transações": "transactions",
    "transacoes": "transactions",
    "transação": "transactions",
    "transacao": "transactions",
    "transactions": "transactions",
    "transaction": "transactions",
    # Orders / Pedidos
    "pedidos": "orders",
    "pedido": "orders",
    "sales_orders": "orders",
    "order": "orders",
    "orders": "orders",
    # Products / Produtos
    "produtos": "products",
    "produto": "products",
    "product": "products",
    "products": "products",
    # Facilities / Credit
    "linha de credito": "facilities",
    "linhas de credito": "facilities",
    "linha de crédito": "facilities",
    "linhas de crédito": "facilities",
    "facilities": "facilities",
    "facility": "facilities",
    # Medallion layer aliases
    "silver": "medallion_silver_transactions",
    "bronze": "medallion_bronze_transactions",
    "transacoes_silver": "medallion_silver_transactions",
    "transações_silver": "medallion_silver_transactions",
    "transacoes_bronze": "medallion_bronze_transactions",
    "transações_bronze": "medallion_bronze_transactions",
}


def resolve_entity_synonym(entity: str) -> str:
    """Resolve an entity name or alias to its canonical synonym."""
    if not entity:
        return ""
    clean = entity.strip().lower()
    return ENTITY_SYNONYMS.get(clean, entity)


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


def _extract_table_or_entity(query: str, resolve_synonyms: bool = False) -> str:
    """Extract table or entity name from query, skipping stop-words and filler words."""
    q = (query or "").strip().lower()
    if not q:
        return ""

    # Check fully qualified table paths first (e.g. workspace.default.customers or default.customers)
    match_full = re.search(r"\b([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\b", q)
    if match_full:
        res = match_full.group(1)
        return resolve_entity_synonym(res) if resolve_synonyms else res

    match_two = re.search(r"\b([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\b", q)
    if match_two:
        res = match_two.group(1)
        return resolve_entity_synonym(res) if resolve_synonyms else res

    raw_tokens = [re.sub(r"[^\w.]", "", tok) for tok in q.split()]
    tokens = [tok for tok in raw_tokens if tok]

    # Check SQL select ... from clause
    for i, tok in enumerate(tokens):
        if tok in ("select", "from") and i + 1 < len(tokens):
            candidate = tokens[i + 1]
            if candidate not in ALL_STOP_WORDS and candidate not in FILLER_KEYWORDS:
                return resolve_entity_synonym(candidate) if resolve_synonyms else candidate

    # Look for marker words
    markers = {
        "tabela",
        "table",
        "dados",
        "pipeline",
        "amostra",
        "preview",
        "for",
        "from",
        "of",
        "about",
        "modelagem",
        "schema",
    }
    for i, tok in enumerate(tokens):
        if tok in markers:
            for candidate in tokens[i + 1 :]:
                if candidate not in ALL_STOP_WORDS and candidate not in FILLER_KEYWORDS:
                    return resolve_entity_synonym(candidate) if resolve_synonyms else candidate

    # Fallback token extraction
    for tok in tokens:
        if tok not in ALL_STOP_WORDS and tok not in FILLER_KEYWORDS and len(tok) > 1:
            return resolve_entity_synonym(tok) if resolve_synonyms else tok

    return ""


def _extract_entity_from_query(query: str) -> str | None:
    """Extract canonical entity name from query using synonym dictionary and stop-word skipping."""
    q = (query or "").strip().lower()
    if not q:
        return None

    candidate = _extract_table_or_entity(q, resolve_synonyms=False)
    if candidate:
        syn = resolve_entity_synonym(candidate)
        if syn and syn != candidate:
            return syn
        if candidate in ENTITY_SYNONYMS:
            return ENTITY_SYNONYMS[candidate]

    # Check multi-word or layer aliases in ENTITY_SYNONYMS sorted by length descending
    sorted_synonyms = sorted(ENTITY_SYNONYMS.keys(), key=len, reverse=True)
    for alias in sorted_synonyms:
        pattern = r"(?:\b|_)" + re.escape(alias) + r"(?:\b|_)"
        if re.search(pattern, q):
            return ENTITY_SYNONYMS[alias]

    if candidate and candidate not in ALL_STOP_WORDS and candidate not in FILLER_KEYWORDS:
        return candidate

    return None


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
        "desta tabela",
        "a partir dessa tabela",
        "tabela acima",
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
                if _is_data_preview_query(content):
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


def _is_entity_modeling_query(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"\b(qual( \w+)* modelagem|modelagem de)\b",
        r"\b(schema da tabela|colunas da tabela|estrutura da tabela)\b",
        r"\b(mostre( \w+)* schema|mostrar( \w+)* schema)\b",
        r"\b(quais( \w+)* colunas|quais( \w+)* campos)\b",
    ]
    return any(re.search(p, q) for p in patterns)
