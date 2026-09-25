import re
from typing import Any

from langchain_core.messages import AIMessage

from src.agent.intent import (
    _extract_entity_from_query,
    _extract_table_or_entity,
    _get_conceptual_explanation,
)
from src.agent.state import AgentState
from src.agent.tools import (
    format_entity_modeling,
    inspect_unity_catalog,
    load_semantic_models,
    preview_table_data,
)
from src.config import settings
from src.semantic.registry import SemanticRegistry


class TitleUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        resp = "Databricks Steward - Governança"
        return {**state, "messages": [AIMessage(content=resp)], "response": resp}


class ConceptualExplanationUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        user_query = state.get("user_query") or ""
        resp = _get_conceptual_explanation(user_query) or "Conceito não encontrado."
        return {**state, "messages": [AIMessage(content=resp)], "response": resp}


class DataPreviewUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        user_query = state.get("user_query") or ""
        target_table = _extract_table_or_entity(user_query)
        lim_match = re.search(r"\blimit\s+(\d+)\b", user_query.lower())
        limit_val = int(lim_match.group(1)) if lim_match else 10
        response_text = preview_table_data(table_name=target_table, limit=limit_val)
        return {
            **state,
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "preview_data": {"table_name": target_table, "limit": limit_val},
        }


class EntityModelingUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        user_query = state.get("user_query") or ""
        ent_name = _extract_entity_from_query(user_query) or "customers"
        resp = format_entity_modeling(ent_name)
        m_match = re.search(r"```mermaid\n(.*?)\n```", resp, re.DOTALL)
        diag = f"```mermaid\n{m_match.group(1)}\n```" if m_match else state.get("active_diagram")
        return {
            **state,
            "messages": [AIMessage(content=resp)],
            "response": resp,
            "active_diagram": diag,
        }


class UnityCatalogUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        response_text = inspect_unity_catalog()
        return {**state, "messages": [AIMessage(content=response_text)], "response": response_text}


class SemanticLayerUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
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
        return {**state, "messages": [AIMessage(content=response_text)], "response": response_text}


class GreetingUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
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
        return {**state, "messages": [AIMessage(content=response_text)], "response": response_text}
