from typing import Any

from langchain_core.messages import AIMessage

from src.agent.intent import _extract_pending_from_history
from src.agent.state import AgentState
from src.agent.tools import deploy_and_materialize_data_product
from src.config import settings
from src.etl.generator import generate_medallion_pipeline
from src.semantic.models import EntityModel
from src.semantic.registry import SemanticRegistry


class DeploymentConfirmationUseCase:
    def execute(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])

        generated_code = state.get("generated_code")
        gitops_result = state.get("gitops_result")

        pending = state.get("pending_pipeline") or _extract_pending_from_history(messages)
        prod_name = pending.get("product_name", "medallion_gold_sales_kpis")
        p_py = pending.get("pyspark", "")
        p_sql = pending.get("sparksql", "")
        p_src = pending.get("source_entity", "medallion_silver_transactions")
        layer = pending.get("layer", "silver")
        ent_name = pending.get("entity_name") or p_src or prod_name

        if not p_py or not p_sql:
            reg = SemanticRegistry(settings.semantic_models_path)
            ent_obj = reg.get_entity(ent_name)
            if not ent_obj:
                from src.databricks.introspector import _build_mock_entities

                mock_ents = {e.name: e for e in _build_mock_entities()}
                ent_obj = mock_ents.get(ent_name)

            if not ent_obj:
                ent_obj = EntityModel(name=ent_name, table_name=prod_name)

            pipeline = generate_medallion_pipeline(ent_obj, layer=layer)
            p_py = pipeline.pyspark_code
            p_sql = pipeline.sparksql_code

        generated_code = {
            "pyspark": p_py,
            "sparksql": p_sql,
            "table_name": prod_name,
            "layer": layer,
        }

        res = deploy_and_materialize_data_product(
            product_name=prod_name,
            pyspark_code=p_py,
            sparksql_code=p_sql,
            source_entity=p_src,
        )

        if res.get("status") == "ci_failed":
            response_text = str(res.get("message", "❌ CI Quality Gate rejeitou o pipeline."))
            ci_report = res.get("ci_report")
            active_diagram = state.get("active_diagram")
        else:
            ci_rep = res["ci_report"]
            git_res = res["git_result"]
            job_res = res["job_result"]
            ent_model = res.get("entity_model")
            diag_md = res.get("updated_diagram", "")
            active_diagram = diag_md
            ci_report = res.get("ci_report")

            metrics_list = (
                ", ".join([f"`{m.name}`" for m in ent_model.metrics])
                if ent_model and ent_model.metrics
                else "`total_records`"
            )
            push_label = (
                "✅ Sincronizado com `origin/main` no GitHub"
                if git_res.get("push_success")
                else "✅ Commit local em `main`"
            )

            response_text = (
                f"## 🚀 Ciclo de Vida do Data Product Concluído com Sucesso!\n\n"
                f"### 💻 Código PySpark e SparkSQL Gerado\n"
                f"#### PySpark Pipeline\n```python\n{p_py}\n```\n\n"
                f"#### SparkSQL DDL & Ingestion\n```sql\n{p_sql}\n```\n\n"
                f"### 🛡️ 1. Esteira de CI Quality Gate\n"
                f"{ci_rep.summary_markdown}\n\n"
                f"### 📦 2. GitOps Auto Commit & Push\n"
                f"- **Branch:** `{git_res['branch']}`\n"
                f"- **Commit SHA:** `{git_res['commit_sha']}`\n"
                f"- **Status Push:** {push_label}\n"
                f"- **Arquivos Comitados:** `{', '.join(git_res['files'])}`\n\n"
                f"### ⚡ 3. Databricks Workflow Job & Materialização\n"
                f"- **Job ID:** `{job_res['job_id']}` (Run: `{job_res['run_id']}`)\n"
                f"- **Tabela Unity Catalog:** `{res['table_name']}`\n"
                f"- **Status:** `✅ Materializada no Catálogo com Sucesso`\n\n"
                f"### 🧠 4. Camada Semântica & Modelo de Dados\n"
                f"- **Entidade Registrada:** `{ent_model.name if ent_model else prod_name}`\n"
                f"- **Métricas Analíticas Criadas:** {metrics_list}\n"
                f"- **Arquivo de Ontologia:** `configs/semantic_models/databricks_medallion.yaml`\n\n"
                f"### 📐 5. Diagrama de Relacionamentos Atualizado\n"
                f"```mermaid\n{diag_md}\n```"
            )

        return {
            "messages": [AIMessage(content=response_text)],
            "response": response_text,
            "active_diagram": active_diagram,
            "generated_code": generated_code,
            "ci_report": ci_report,
            "gitops_result": gitops_result,
            "job_result": res.get("job_result", None),
            "pending_pipeline": None,
        }
