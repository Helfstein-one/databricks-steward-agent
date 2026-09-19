import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def is_confirmation(query: str) -> bool:
    q = (query or "").strip().lower()
    if re.search(r"\b(não|nao|nem|nunca|jamais|cancelar|cancela)\b", q):
        return False
    patterns = [
        r"\b(sim|confirmar|confirmo|aprovar|aprovo|pode executar|executa|executar|confirmado|prosseguir|ok|yes|positivo|autorizado|pode rodar|pode seguir|manda bala|pode fazer|vamos lá|bora)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def is_data_preview_query(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"\b(consultar|consulte|quero ver|ver|mostrar|mostre|trazer|traga|exibir|exiba|ler|leia)\s+(os\s+)?(dados|registros|linhas)\b",
        r"\b(amostra|preview)\s+(de\s+|dos\s+|da\s+)?(dados|tabela)\b",
        r"\b(select\s+\*\s+from)\b",
        r"\b(puxa|puxe)\s+(uns\s+)?(\d+\s+)?(registros|linhas)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def classify_intent_with_llm(query: str, llm: ChatOpenAI) -> str:
    q = (query or "").strip()
    if not q:
        return "OTHER"
    if is_confirmation(query):
        return "CONFIRM"
    if is_data_preview_query(query):
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
        for tag in ["PREVIEW", "SCHEMA", "DIAGRAM", "ETL", "CONFIRM", "GREETING", "OTHER"]:
            if tag in content:
                return tag
        return "OTHER"
    except Exception:  # noqa: BLE001
        return "OTHER"


def is_greeting(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"^(ol[áa]|hello|hi|oi|oie|bom dia|boa tarde|boa noite|opa|e a[íi])\b",
        r"\b(quem\s+[ée]\s+voc[eê]|o\s+que\s+voc[eê]\s+faz)\b",
        r"\b(ajuda|help|socorro|menu)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def is_title_request(query: str) -> bool:
    q = (query or "").strip().lower()
    return bool(re.search(r"^(title|título|gerar t[ií]tulo)", q))


def is_conceptual_question(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"^(o\s+que\s+[ée]|qual\s+o\s+conceito|o\s+que\s+significa|me\s+explica)\b",
        r"\b(como\s+funciona|pra\s+que\s+serve|por\s+que\s+usar)\b",
        r"\b(conceito\s+de|vantagens\s+de|diferen[çc]a\s+entre)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def is_entity_modeling_query(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"\b(qual( \w+)* modelagem|modelagem de)\b",
        r"\b(schema da tabela|colunas da tabela|estrutura da tabela)\b",
        r"\b(mostre( \w+)* schema|mostrar( \w+)* schema)\b",
        r"\b(quais( \w+)* colunas|quais( \w+)* campos)\b",
    ]
    return any(re.search(p, q) for p in patterns)
