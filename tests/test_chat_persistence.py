"""Unit and integration tests for stateful SQLite chat history persistence and ETL version checkpoints."""

from src.agent import checkpoint as chk_mod
from src.agent.checkpoint import ConversationalCheckpointManager
from src.agent.tools import (
    get_etl_checkpoint,
    list_etl_checkpoints,
    restore_etl_checkpoint,
)
from src.application.use_cases.etl_generation import ETLGenerationUseCase


def test_checkpoint_manager_save_and_list(tmp_path):
    db_path = str(tmp_path / "test_chat_history.db")
    mgr = ConversationalCheckpointManager(db_path=db_path)

    chk1 = mgr.save_checkpoint(
        thread_id="thread-101",
        entity_name="facilities",
        pyspark_code="print('v1')",
        sparksql_code="SELECT 1;",
    )

    assert chk1["version"] == 1
    assert chk1["entity_name"] == "facilities"

    chk2 = mgr.save_checkpoint(
        thread_id="thread-101",
        entity_name="facilities",
        pyspark_code="print('v2')",
        sparksql_code="SELECT 2;",
    )

    assert chk2["version"] == 2

    checkpoints = mgr.list_checkpoints(thread_id="thread-101", entity_name="facilities")
    assert len(checkpoints) == 2
    assert checkpoints[0]["version"] == 2
    assert checkpoints[1]["version"] == 1


def test_checkpoint_manager_get_and_restore(tmp_path):
    db_path = str(tmp_path / "test_chat_history.db")
    mgr = ConversationalCheckpointManager(db_path=db_path)

    mgr.save_checkpoint(
        thread_id="thread-202",
        entity_name="orders",
        pyspark_code="df = spark.read.table('orders')",
        sparksql_code="SELECT * FROM orders;",
    )

    v1 = mgr.get_checkpoint(version_or_id=1, thread_id="thread-202", entity_name="orders")
    assert v1 is not None
    assert v1["version"] == 1
    assert "df = spark" in v1["pyspark_code"]

    restored = mgr.restore_checkpoint(version_or_id=1, thread_id="thread-202", entity_name="orders")
    assert restored is not None
    assert restored["version"] == 1


def test_etl_generation_use_case_persists_checkpoint(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test_usecase.db")
    test_mgr = ConversationalCheckpointManager(db_path=db_path)
    monkeypatch.setattr(chk_mod, "_default_checkpoint_manager", test_mgr)

    use_case = ETLGenerationUseCase()
    state = {
        "user_query": "gerar pipeline etl para facilities",
        "messages": [],
        "thread_id": "thread-test-usecase",
    }

    res = use_case.execute(state)

    assert "response" in res
    assert "Versão v1 do ETL salva no banco de histórico conversacional" in res["response"]

    saved = test_mgr.list_checkpoints(thread_id="thread-test-usecase")
    assert len(saved) >= 1
    assert saved[0]["version"] == 1


def test_checkpoint_tools(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test_tools.db")
    test_mgr = ConversationalCheckpointManager(db_path=db_path)
    monkeypatch.setattr(chk_mod, "_default_checkpoint_manager", test_mgr)

    test_mgr.save_checkpoint(
        thread_id="thread-tools",
        entity_name="customers",
        pyspark_code="def process_customers(): pass",
        sparksql_code="SELECT customer_id FROM customers;",
    )

    list_res = list_etl_checkpoints(thread_id="thread-tools")
    assert "Histórico de Versões e Checkpoints de ETL" in list_res
    assert "`v1`" in list_res
    assert "`customers`" in list_res

    get_res = get_etl_checkpoint(version_or_id=1, thread_id="thread-tools")
    assert "ETL Checkpoint" in get_res
    assert "SELECT customer_id FROM customers;" in get_res

    restore_res = restore_etl_checkpoint(version_or_id=1, thread_id="thread-tools")
    assert "Versão v1 do ETL restaurada com sucesso!" in restore_res
