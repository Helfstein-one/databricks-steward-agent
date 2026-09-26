from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.application.intent_router import (
    classify_intent_with_llm,
    is_conceptual_question,
    is_confirmation,
    is_data_preview_query,
    is_entity_modeling_query,
    is_greeting,
    is_title_request,
)
from src.application.orchestrator import StewardAppService
from src.application.query_parser import (
    extract_entity_from_query,
    extract_pending_from_history,
    extract_query_text,
    extract_table_or_entity,
    resolve_anaphoric_entity,
)
from src.application.use_cases.data_preview import DataPreviewUseCase
from src.application.use_cases.schema_introspection import SchemaIntrospectionUseCase
from src.domain.ports.ci_port import ICiAdapter
from src.domain.ports.databricks_port import IDatabricksAdapter
from src.domain.ports.gitops_port import IGitOpsAdapter


# --- Ports Base Class Tests ---
def test_ports_abstract_methods():
    class DummyCiPort(ICiAdapter):
        def run_pipeline(self, pyspark_code: str, sparksql_code: str):
            return super().run_pipeline(pyspark_code, sparksql_code)

    class DummyDbPort(IDatabricksAdapter):
        def execute_query(self, query: str):
            return super().execute_query(query)

        def preview_table(self, table_name: str, limit: int = 10):
            return super().preview_table(table_name, limit)

        def inspect_schema(self):
            return super().inspect_schema()

        def introspect_catalog(self, catalog: str | None = None, schema: str | None = None):
            return super().introspect_catalog(catalog, schema)

        def deploy_job(self, product_name: str, pyspark_code: str, sparksql_code: str):
            return super().deploy_job(product_name, pyspark_code, sparksql_code)

    class DummyGitOpsPort(IGitOpsAdapter):
        def commit_and_push(
            self,
            product_name: str,
            pyspark_code: str,
            sparksql_code: str,
            ci_report: str,
            active_diagram: str,
        ):
            return super().commit_and_push(
                product_name, pyspark_code, sparksql_code, ci_report, active_diagram
            )

    ci = DummyCiPort()
    assert ci.run_pipeline("", "") is None

    db = DummyDbPort()
    assert db.execute_query("") is None
    assert db.preview_table("") is None
    assert db.inspect_schema() is None
    assert db.deploy_job("", "", "") is None

    git = DummyGitOpsPort()
    assert git.commit_and_push("", "", "", "", "") is None


# --- Use Cases Tests ---
def test_data_preview_use_case():
    mock_db = MagicMock()
    mock_db.preview_table.return_value = "markdown preview"
    use_case = DataPreviewUseCase(mock_db)

    # Empty table name
    res_empty = use_case.execute("")
    assert "❌ Nenhuma tabela identificada" in res_empty

    # Valid table name
    res_valid = use_case.execute("my_catalog.my_schema.my_table", limit=15)
    mock_db.preview_table.assert_called_once_with("my_catalog.my_schema.my_table", 15)
    assert res_valid == "markdown preview"


def test_schema_introspection_use_case():
    mock_db = MagicMock()
    mock_db.inspect_schema.return_value = "schema info"
    use_case = SchemaIntrospectionUseCase(mock_db)

    res = use_case.execute()
    mock_db.inspect_schema.assert_called_once()
    assert res == "schema info"


# --- Orchestrator Tests ---
def test_steward_app_service_preview_intent():
    mock_db = MagicMock()
    mock_db.preview_table.return_value = "| col1 |\n| val1 |"
    service = StewardAppService(mock_db, MagicMock(), MagicMock())

    # Case 1: Direct table preview with limit
    state = {"messages": []}
    res = service.execute(state, "PREVIEW", "consultar dados de sales limit 5")
    mock_db.preview_table.assert_called_once_with("sales", 5)
    assert res["response"] == "| col1 |\n| val1 |"
    assert isinstance(res["messages"][0], AIMessage)

    # Case 2: Preview without table name and no context
    res_no_table = service.execute(state, "PREVIEW", "ver dados")
    assert "❌ Nenhuma tabela identificada" in res_no_table["response"]


def test_steward_app_service_confirm_intent():
    mock_db = MagicMock()
    mock_db.deploy_job.return_value = {"status": "deployed", "job_id": "123"}
    service = StewardAppService(mock_db, MagicMock(), MagicMock())

    # With pending_pipeline in state
    state = {
        "pending_pipeline": {
            "product_name": "gold_sales",
            "pyspark": "print('py')",
            "sparksql": "SELECT 1",
            "source_entity": "silver_trans",
        }
    }
    res = service.execute(state, "CONFIRM", "sim, pode executar")
    mock_db.deploy_job.assert_called_once_with(
        product_name="gold_sales", pyspark_code="print('py')", sparksql_code="SELECT 1"
    )
    assert res == {"status": "deployed", "job_id": "123"}


def test_steward_app_service_other_intent():
    service = StewardAppService(MagicMock(), MagicMock(), MagicMock())
    res = service.execute({}, "OTHER", "qualquer coisa")
    assert res == {"response": "Not Implemented Yet"}


# --- Intent Router Tests ---
def test_intent_router_matchers():
    assert is_confirmation("sim, pode rodar")
    assert is_confirmation("confirmar")
    assert not is_confirmation("não, cancelar")

    assert is_data_preview_query("quero ver os dados da tabela")
    assert is_data_preview_query("amostra de dados")
    assert is_data_preview_query("SELECT * FROM table")
    assert is_data_preview_query("puxa uns 10 registros")
    assert not is_data_preview_query("olá tudo bem")

    assert is_greeting("olá, bom dia")
    assert is_greeting("quem é você")
    assert is_greeting("ajuda")
    assert not is_greeting("criar pipeline")

    assert is_title_request("title do chat")
    assert is_title_request("gerar título para conversa")
    assert not is_title_request("outra coisa")

    assert is_conceptual_question("o que é medallion?")
    assert is_conceptual_question("como funciona delta lake?")
    assert is_conceptual_question("diferença entre bronze e silver")
    assert not is_conceptual_question("puxe 10 linhas")

    assert is_entity_modeling_query("modelagem de vendas")
    assert is_entity_modeling_query("schema da tabela clientes")
    assert is_entity_modeling_query("quais colunas existem?")
    assert not is_entity_modeling_query("sim")


def test_classify_intent_with_llm():
    # Empty query
    assert classify_intent_with_llm("", None) == "OTHER"

    # Confirmation query short-circuit
    assert classify_intent_with_llm("sim, confirmo", None) == "CONFIRM"

    # Data preview query short-circuit
    assert classify_intent_with_llm("ver dados da tabela", None) == "PREVIEW"

    # Mock LLM chain for SCHEMA classification
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "  TAG: SCHEMA  "
    mock_llm.invoke.return_value = mock_response

    # Prompt | llm chain mock
    with patch("src.application.intent_router.ChatPromptTemplate") as mock_prompt_cls:
        mock_prompt = MagicMock()
        mock_prompt_cls.from_messages.return_value = mock_prompt
        mock_chain = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mock_chain.invoke.return_value = mock_response

        tag = classify_intent_with_llm("quais tabelas existem?", mock_llm)
        assert tag == "SCHEMA"

        # LLM exception fallback
        mock_chain.invoke.side_effect = Exception("LLM Error")
        tag_err = classify_intent_with_llm("descreva o schema", mock_llm)
        assert tag_err == "OTHER"

        # LLM response with no known tag
        mock_chain.invoke.side_effect = None
        mock_response.content = "UNKNOWN_TAG"
        tag_unk = classify_intent_with_llm("alguma query", mock_llm)
        assert tag_unk == "OTHER"


# --- Query Parser Tests ---
def test_query_parser_extract_query_text():
    assert extract_query_text([]) == ""
    msg = HumanMessage(content="  hello world  ")
    assert extract_query_text([msg]) == "hello world"


def test_query_parser_extract_table_or_entity():
    assert extract_table_or_entity("dados de sales_data") == "sales_data"
    assert extract_table_or_entity("from my_table") == "my_table"
    assert extract_table_or_entity("qual o conceito?") == ""


def test_query_parser_extract_entity_from_query():
    assert extract_entity_from_query("schema da tabela clientes") == "clientes"
    assert extract_entity_from_query("uma um de para") is None


def test_query_parser_resolve_anaphoric_entity():
    # Anaphoric reference with state pending_pipeline
    state = {"pending_pipeline": {"source_entity": "silver_orders"}}
    res = resolve_anaphoric_entity("preview dessa tabela", [], state)
    assert res == "silver_orders"

    # Anaphoric reference from message history (AI message with lakehouse table hint)
    state_empty = {}
    msgs_lakehouse = [
        AIMessage(content="Tabela Lakehouse: `main.default.gold_kpis`"),
        HumanMessage(content="ver dados dela"),
    ]
    res_lake = resolve_anaphoric_entity("dela", msgs_lakehouse, state_empty)
    assert res_lake == "gold_kpis"

    # Anaphoric reference from message history (AI message with generated pipeline hint)
    msgs_pipeline = [
        AIMessage(content="Generated Medallion Pipeline: silver_transactions (Silver Layer)"),
        HumanMessage(content="amostra dessa tabela"),
    ]
    res_pipe = resolve_anaphoric_entity("amostra dessa tabela", msgs_pipeline, state_empty)
    assert res_pipe == "silver_transactions"

    # Anaphoric reference from previous human preview message
    msgs_preview = [
        HumanMessage(content="consultar dados de catalog.schema.customers"),
        AIMessage(content="aqui estao os dados"),
        HumanMessage(content="mostre mais dados da tabela"),
    ]
    res_prev = resolve_anaphoric_entity("mostre mais dados da tabela", msgs_preview, state_empty)
    assert res_prev == "customers"

    # Non anaphoric query
    assert resolve_anaphoric_entity("olá tudo bem", [], state_empty) is None


def test_query_parser_extract_pending_from_history():
    content = """
    Generated Medallion Pipeline: gold_sales_kpis (Gold Layer)
    Deseja confirmar e disparar o deploy?

    #### PySpark Pipeline
    ```python
    df = spark.read.table("silver")
    ```

    #### SparkSQL DDL & Ingestion
    ```sql
    CREATE TABLE gold_sales_kpis;
    ```
    """
    messages = [HumanMessage(content="faca o etl"), AIMessage(content=content)]
    pending = extract_pending_from_history(messages)
    assert pending.get("product_name") == "gold_sales_kpis"
    assert pending.get("pyspark") == 'df = spark.read.table("silver")'
    assert pending.get("sparksql") == "CREATE TABLE gold_sales_kpis;"

    # History with no matching pending message
    assert extract_pending_from_history([HumanMessage(content="oi")]) == {}
