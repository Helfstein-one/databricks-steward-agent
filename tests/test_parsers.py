import pytest
from src.agent.parsers import _extract_table_or_entity

def test_extract_table_or_entity():
    scenarios = [
        ("tabela de clientes", "customers"),
        ("dados de transações", "transactions"),
        ("dados da tabela pedidos", "orders"),
        ("quero os dados de um cliente", "customers"),
        ("quero os dados da clientes", "customers"),
        ("select * from usuarios", "users"),
        ("quero ver as vendas", "sales"),
        ("me mostre a tabela de produtos", "products"),
        ("consultar os dados sobre vendas", "sales"),
        ("qual a tabela de usuarios", "users"),
        ("dados de sales", "sales"),
        ("tabela de users", "users"),
        ("show me the orders table", "orders"),
        ("select from products", "products"),
        ("tabela de", ""),
        ("", "")
    ]
    
    for query, expected in scenarios:
        assert _extract_table_or_entity(query) == expected
