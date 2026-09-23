import logging
import re
from typing import Any

from src.agent.tools import deploy_and_materialize_data_product, inspect_unity_catalog
from src.databricks.client import DatabricksCEClient
from src.databricks.introspector import introspect_catalog
from src.domain.ports.databricks_port import IDatabricksAdapter

_client_instance = DatabricksCEClient()


def execute_query(query: str) -> list[dict[str, Any]]:
    try:
        res = _client_instance.execute_query(query)
        if isinstance(res, dict) and "error" in res:
            return [res]
        raw_res = res.get("result") if isinstance(res, dict) else None
        return getattr(raw_res, "data_array", []) if raw_res else []
    except Exception as e:
        return [{"error": f"❌ Erro ao consultar Databricks: {e}"}]


def preview_table_data(table_name: str, limit: int = 10) -> str:
    try:
        res = _client_instance.preview_table_data(table_name, limit)
        if isinstance(res, dict) and "error" in res:
            return str(res["error"])
        return str(res.get("markdown_table", "")) if isinstance(res, dict) else str(res)
    except Exception as e:
        return f"❌ Erro ao consultar Databricks: {e}"


def inspect_unity_catalog_local(catalog: str | None = None, schema: str | None = None) -> str:
    from src.config import settings
    client = DatabricksCEClient()
    cat_to_use = catalog or settings.databricks_default_catalog or "workspace"
    sch_to_use = schema or settings.databricks_default_schema or "default"
    mode_str = (
        "🟢 Conectado ao Databricks Real via SDK"
        if client.is_configured()
        else "🟡 Modo Demonstração Offline (defina DATABRICKS_HOST e DATABRICKS_TOKEN no .env)"
    )

    entities = introspect_catalog(catalog=cat_to_use, schema=sch_to_use, client=client)
    actual_cat = entities[0].catalog if entities else cat_to_use
    actual_sch = entities[0].schema_name if entities else sch_to_use
    lines = [
        f"Discovered {len(entities)} entities in {actual_cat}.{actual_sch} (*{mode_str}*):",
    ]
    for ent in entities:
        col_summary = ", ".join([f"`{c.name}` ({c.type})" for c in ent.columns[:6]])
        lines.append(f"- **`{ent.name}`** (Camada: `{ent.layer or 'unassigned'}`): {col_summary}")

    lines.append(
        "\n💡 *Nota: Estas são as tabelas físicas inspecionadas no Unity Catalog. "
        "Para ver os modelos de dados e Data Products da Camada Semântica, digite `2`.*"
    )
    return "\n".join(lines)


def deploy_and_materialize_data_product_local(
    product_name: str,
    pyspark_code: str,
    sparksql_code: str,
    source_entity: str | None = None,
) -> dict[str, Any]:
    from src.ci.runner import run_ci_pipeline
    from src.config import settings
    from src.gitops.git_client import GitClient
    from src.semantic.registry import SemanticRegistry
    from src.visualizer.mermaid import generate_er_diagram

    logger = logging.getLogger(__name__)

    # 1. CI Quality Gate
    ci_rep = run_ci_pipeline(pyspark_code=pyspark_code, sparksql_code=sparksql_code)
    if not ci_rep.is_approved:
        return {
            "status": "ci_failed",
            "ci_report": ci_rep,
            "message": f"❌ CI Quality Gate rejeitou o pipeline:\n\n{ci_rep.summary_markdown}",
        }

    product_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", product_name.replace(".", "-")).lower().strip("-")
    table_name = f"{settings.databricks_default_catalog}.{settings.databricks_default_schema}.{product_slug.replace('-', '_')}"

    # 2. Git commit & push to main
    files = {
        f"pipelines/{product_slug}/etl.py": pyspark_code,
        f"pipelines/{product_slug}/schema.sql": sparksql_code,
    }
    git = GitClient()
    commit_msg = f"feat(pipeline): add data product {product_slug}"
    git_res = git.commit_and_push_to_main(files=files, message=commit_msg)

    # 3. Databricks Job Creation & Execution
    db_client = DatabricksCEClient()
    job_res = db_client.create_or_update_pipeline_job(
        job_name=f"DataProduct_{product_slug.replace('-', '_')}",
        product_slug=product_slug,
        sql_statement=sparksql_code,
        pyspark_code=pyspark_code,
    )

    # 4. Inspecionar colunas e registrar na Camada Semântica
    inferred_cols: list[dict[str, Any]] = []
    try:
        preview = db_client.preview_table_data(table_name=table_name, limit=1)
        for c in preview.get("columns", []):
            inferred_cols.append({"name": c, "type": "string"})
    except Exception as err:
        logger.debug("Preview inference notice: %s", err)

    if not inferred_cols:
        col_matches = re.findall(r"`?([a-zA-Z0-9_]+)`?\s+([A-Z]+)", sparksql_code)
        for c_name, c_type in col_matches:
            if c_name.upper() not in ("CREATE", "TABLE", "IF", "NOT", "EXISTS", "AS", "SELECT"):
                inferred_cols.append({"name": c_name, "type": c_type.lower()})

    reg = SemanticRegistry(settings.semantic_models_path)
    ent_model = reg.register_data_product_entity(
        entity_name=product_slug.replace("-", "_"),
        table_name=table_name,
        columns=inferred_cols or [{"name": "id", "type": "string"}, {"name": "total_amount", "type": "double"}],
        source_entity=source_entity,
        models_dir=settings.semantic_models_path,
    )

    updated_diagram = generate_er_diagram(list(reg.entities.values()), reg.relationships)

    return {
        "status": "success",
        "product_name": product_name,
        "product_slug": product_slug,
        "table_name": table_name,
        "ci_report": ci_rep,
        "git_result": git_res,
        "job_result": job_res,
        "entity_model": ent_model,
        "updated_diagram": updated_diagram,
    }


class DatabricksAdapter(IDatabricksAdapter):
    def execute_query(self, query: str) -> list[dict[str, Any]]:
        return execute_query(query)

    def preview_table(self, table_name: str, limit: int = 10) -> str:
        return preview_table_data(table_name, limit)

    def preview_table_formatted(self, table_name: str, limit: int = 10) -> str:
        try:
            client = DatabricksCEClient()
            res = client.preview_table_data(table_name=table_name, limit=limit)
            md_table = res.get("markdown_table", "*Nenhum dado encontrado.*")
            row_count = res.get("row_count", 0)
            full_table = res.get("table_name", table_name)
            return (
                f"### 📊 Amostra de Dados da Tabela: `{full_table}` (Top {row_count} registros)\n\n"
                f"{md_table}\n\n"
                f"💡 *Dica: Para gerar um pipeline ETL a partir desta tabela, peça: 'propor etl a partir de {table_name}'.*"
            )
        except Exception as e:
            return f"❌ Erro ao consultar Databricks: {e}"

    def inspect_schema(self, catalog: str | None = None, schema: str | None = None) -> str:
        try:
            return inspect_unity_catalog_local(catalog=catalog, schema=schema)
        except Exception as e:
            return f"❌ Erro ao consultar Databricks: {e}"

    def deploy_job(
        self,
        product_name: str,
        pyspark_code: str,
        sparksql_code: str,
        source_entity: str | None = None,
    ) -> dict[str, Any]:
        try:
            return deploy_and_materialize_data_product_local(
                product_name,
                pyspark_code,
                sparksql_code,
                source_entity=source_entity,
            )
        except Exception as e:
            return {"status": "error", "error": f"❌ Erro ao consultar Databricks: {e}"}
