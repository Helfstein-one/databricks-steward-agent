import pytest

from src.agent.parsers import (
    PORTUGUESE_STOP_WORDS,
    _extract_entity_from_query,
    _extract_table_or_entity,
    resolve_entity_synonym,
)


def test_acceptance_criterion_1() -> None:
    """_extract_table_or_entity('quero ver os dados da tabela de clientes') returns 'clientes'."""
    result = _extract_table_or_entity("quero ver os dados da tabela de clientes")
    assert result == "clientes"


def test_acceptance_criterion_2() -> None:
    """_extract_table_or_entity('dados das transacoes do mes passado') returns 'transacoes'."""
    result = _extract_table_or_entity("dados das transacoes do mes passado")
    assert result == "transacoes"


@pytest.mark.parametrize(
    "query,expected_table",
    [
        # Portuguese scenarios
        ("quero ver os dados da tabela de clientes", "clientes"),
        ("dados das transacoes do mes passado", "transacoes"),
        ("me gera um pipeline silver das transações", "transações"),
        ("preview da tabela de pedidos", "pedidos"),
        ("mostrar dados de produtos", "produtos"),
        ("consulte os dados da tabela workspace.default.customers", "workspace.default.customers"),
        ("amostra de dados da tabela default.customers", "default.customers"),
        ("select * from orders limit 10", "orders"),
        # English scenarios
        ("show data for customers table", "customers"),
        ("pull records from transactions", "transactions"),
        ("display sales_orders data", "sales_orders"),
        ("get preview of products", "products"),
    ],
)
def test_extract_table_or_entity_scenarios(query: str, expected_table: str) -> None:
    """Test table and entity extraction from free-text queries in Portuguese and English."""
    assert _extract_table_or_entity(query) == expected_table


@pytest.mark.parametrize(
    "query,expected_canonical",
    [
        # Priority synonyms resolution
        ("quero ver os dados da tabela de clientes", "customers"),
        ("me gera um pipeline silver das transações", "transactions"),
        ("dados das transacoes do mes passado", "transactions"),
        ("propor pipeline para pedidos", "orders"),
        ("qual a modelagem de produtos", "products"),
        ("mostrar schema da tabela de compradores", "customers"),
        # English queries
        ("show schema for customers table", "customers"),
        ("generate silver pipeline for transactions", "transactions"),
        ("model for sales_orders", "orders"),
        ("display products structure", "products"),
    ],
)
def test_extract_entity_from_query_scenarios(query: str, expected_canonical: str) -> None:
    """Test NLP entity extraction with synonym resolution."""
    assert _extract_entity_from_query(query) == expected_canonical


def test_resolve_entity_synonym() -> None:
    """Test explicit synonym resolution for priority terms."""
    assert resolve_entity_synonym("clientes") == "customers"
    assert resolve_entity_synonym("cliente") == "customers"
    assert resolve_entity_synonym("transações") == "transactions"
    assert resolve_entity_synonym("transacoes") == "transactions"
    assert resolve_entity_synonym("pedidos") == "orders"
    assert resolve_entity_synonym("pedido") == "orders"
    assert resolve_entity_synonym("produtos") == "products"
    assert resolve_entity_synonym("unknown_entity") == "unknown_entity"


def test_stop_words_filtering() -> None:
    """Verify Portuguese stop-words are recognized and ignored."""
    for stop_word in ["de", "da", "dos", "as", "um", "uma", "o", "a"]:
        assert stop_word in PORTUGUESE_STOP_WORDS
