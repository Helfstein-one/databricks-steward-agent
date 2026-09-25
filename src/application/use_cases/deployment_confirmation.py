from typing import Any

from langchain_core.messages import AIMessage

from src.agent.intent import _extract_pending_from_history
from src.agent.state import AgentState
from src.agent.tools import deploy_and_materialize_data_product


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
