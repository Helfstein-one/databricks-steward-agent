import re
from typing import Any

from langchain_core.messages import AIMessage

from src.adapters.databricks_adapter import DatabricksAdapter
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
from src.semantic.compiler import SemanticQueryCompiler
from src.semantic.registry import SemanticRegistry


class AnalyticsUseCase:
    """Use case to compile natural language analytics requests into SQL via semantic layer and auto-run against Databricks."""

    def _parse_nl_query(
        self, user_query: str, registry: SemanticRegistry
    ) -> tuple[str, list[str], list[str], list[str]]:
        q_lower = user_query.lower()

        # 1. Metric Resolution
        selected_metrics: list[str] = []
        # Search all metrics in registry by name or synonym
        for m_name, metric in registry.metrics.items():
            names_to_check = [m_name] + [s.lower() for s in metric.synonyms]
            if any(name in q_lower for name in names_to_check):
                selected_metrics.append(metric.name)

        if not selected_metrics:
            if any(k in q_lower for k in ("receita", "faturamento", "revenue")):
                selected_metrics.append("total_revenue")
            elif any(k in q_lower for k in ("pedido", "pedidos", "orders")):
                selected_metrics.append("total_orders")
            elif any(k in q_lower for k in ("ticket", "aov", "medio", "médio")):
                selected_metrics.append("avg_transaction_value")
            elif "vip" in q_lower:
                selected_metrics.append("total_vip_customers")
            elif any(k in q_lower for k in ("cliente", "clientes", "customers")):
                selected_metrics.append("total_customers")
            else:
                selected_metrics.append("total_revenue")

        selected_metrics = list(dict.fromkeys(selected_metrics))

        # 2. Entity Resolution
        target_entity: str | None = None
        for ent_name, ent in registry.entities.items():
            ent_names = [ent_name, ent.name.lower()] + [s.lower() for s in ent.synonyms]
            if any(name in q_lower for name in ent_names):
                target_entity = ent.name
                break

        if not target_entity:
            owner_info = registry.find_metric_owner(selected_metrics[0])
            if owner_info and owner_info[1]:
                target_entity = owner_info[1]
            else:
                target_entity = "medallion_silver_transactions"

        # 3. Group By Dimensions Resolution
        group_by_dims: list[str] = []
        dim_keywords = {
            "category": ["categoria", "category"],
            "channel": ["canal", "channel"],
            "status": ["status", "estado"],
            "date": ["data", "date", "dia"],
            "segment": ["segmento", "segment"],
            "country": ["pais", "país", "country"],
            "user_id": ["usuario", "usuário", "cliente_id", "user"],
            "is_vip": ["vip"],
        }
        for dim_name, keywords in dim_keywords.items():
            if any(
                f"por {kw}" in q_lower or f"by {kw}" in q_lower or f"per {kw}" in q_lower
                for kw in keywords
            ):
                dim_info = registry.find_dimension_owner(dim_name, preferred_entity=target_entity)
                if dim_info:
                    group_by_dims.append(dim_info[0].name)
                else:
                    group_by_dims.append(dim_name)

        group_by_dims = list(dict.fromkeys(group_by_dims))

        # 4. Filters Resolution
        filters: list[str] = []
        if any(w in q_lower for w in ("ontem", "yesterday")):
            filters.append(f"{target_entity}.date = date_sub(current_date(), 1)")
        elif any(w in q_lower for w in ("hoje", "today")):
            filters.append(f"{target_entity}.date = current_date()")

        if any(w in q_lower for w in ("concluida", "concluido", "completed")):
            filters.append(f"{target_entity}.status = 'COMPLETED'")

        return target_entity, selected_metrics, group_by_dims, filters

    def _format_results_to_markdown(
        self, results: Any, expected_cols: list[str] | None = None
    ) -> str:
        if not results:
            return "*Nenhum registro retornado.*"

        if isinstance(results, list) and len(results) > 0:
            first = results[0]
            if isinstance(first, dict):
                if "error" in first:
                    return f"❌ **Erro na execução Databricks:** {first['error']}"
                cols = list(first.keys())
                header = "| " + " | ".join(cols) + " |"
                sep = "| " + " | ".join(["---"] * len(cols)) + " |"
                rows = [
                    "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |" for row in results
                ]
                return "\n".join([header, sep, *rows])
            elif isinstance(first, (list, tuple)):
                cols = (
                    expected_cols
                    if expected_cols and len(expected_cols) == len(first)
                    else [f"col_{i + 1}" for i in range(len(first))]
                )
                header = "| " + " | ".join(cols) + " |"
                sep = "| " + " | ".join(["---"] * len(cols)) + " |"
                rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in results]
                return "\n".join([header, sep, *rows])

        return f"```\n{results}\n```"

    def execute(self, state: AgentState) -> dict[str, Any]:
        user_query = state.get("user_query") or ""
        registry = SemanticRegistry(settings.semantic_models_path)
        compiler = SemanticQueryCompiler(registry)

        entity_name, metric_names, group_by_dims, filters = self._parse_nl_query(
            user_query, registry
        )

        sql = compiler.compile_query(
            entity_name=entity_name,
            metric_names=metric_names,
            group_by_dims=group_by_dims,
            filters=filters,
        )

        db_adapter = DatabricksAdapter()
        raw_results = db_adapter.execute_query(sql)

        expected_cols = group_by_dims + metric_names
        md_table = self._format_results_to_markdown(raw_results, expected_cols)

        response_text = (
            "### 📊 Consulta Analítica Semântica (NL2SQL)\n\n"
            "**Query SparkSQL Compilada via Camada Semântica:**\n"
            f"```sql\n{sql}\n```\n\n"
            "**Resultado Executado no Lakehouse:**\n\n"
            f"{md_table}"
        )

        return {
            **state,
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "compiled_sql": sql,
            "analytics_result": raw_results,
        }


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
