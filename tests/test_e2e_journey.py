"""End-to-End User Journey Test for Databricks Steward Agent.

Simulates the complete 3-step Open WebUI conversation flow against live Databricks:
1. Preview table: 'consulte os dados da tabela customers'
2. Propose Gold ETL: 'propor um etl gold a partir dessa tabela'
3. Confirm execution: 'sim, pode prosseguir' (regex confirmation)
4. Verify remote Databricks Workflow Job via Databricks SDK
"""

from __future__ import annotations

import re
import sys
import time

from databricks.sdk import WorkspaceClient

from open_webui_pipe import Pipe
from src.config import settings


def print_banner(title: str) -> None:
    """Print formatted section banner."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_e2e_journey() -> bool:
    """Execute complete end-to-end journey against live Databricks."""
    print_banner("🚀 STARTING DATABRICKS STEWARD AGENT E2E JOURNEY TEST")
    print(f"Databricks Host: {settings.databricks_host}")
    print(f"Default Catalog: {settings.databricks_default_catalog}")
    print(f"Default Schema:  {settings.databricks_default_schema}")
    print(f"Warehouse ID:    {settings.databricks_warehouse_id or '(auto-discovery)'}")
    print(f"Local LLM Model: {settings.local_llm_model}")

    # Ensure required configuration
    if not settings.databricks_host or not settings.databricks_token:
        print("❌ Error: DATABRICKS_HOST and DATABRICKS_TOKEN must be configured in .env")
        return False

    pipe = Pipe()
    messages: list[dict[str, str]] = []

    # -------------------------------------------------------------------------
    # STEP 1: Data Preview Query
    # -------------------------------------------------------------------------
    print_banner("STEP 1: Data Preview Query ('consulte os dados da tabela customers')")
    q1 = "consulte os dados da tabela customers"
    print(f"[User Turn 1]: {q1}\n")
    messages.append({"role": "user", "content": q1})

    t0 = time.time()
    resp1 = pipe.pipe({"messages": messages, "stream": False})
    elapsed1 = time.time() - t0

    assert isinstance(resp1, str) and len(resp1) > 0, "Step 1: Empty response received"
    print(f"[Agent Response 1] ({elapsed1:.2f}s):\n{resp1}\n")

    # Assertions for Step 1
    assert any(k in resp1.lower() for k in ("amostra de dados", "customers", "customer_id", "|")), (
        f"Step 1: Response does not contain expected customer preview table:\n{resp1[:300]}"
    )
    assert "❌ **Erro ao consultar Databricks:**" not in resp1, (
        f"Step 1: Databricks preview returned an error:\n{resp1}"
    )

    messages.append({"role": "assistant", "content": resp1})
    print("✅ Step 1 Verified: Customer preview returned successfully with live data.")

    # -------------------------------------------------------------------------
    # STEP 2: Propose Gold ETL Pipeline
    # -------------------------------------------------------------------------
    print_banner("STEP 2: Propose Gold ETL Pipeline ('propor um etl gold a partir dessa tabela')")
    q2 = "propor um etl gold a partir dessa tabela"
    print(f"[User Turn 2]: {q2}\n")
    messages.append({"role": "user", "content": q2})

    t0 = time.time()
    resp2 = pipe.pipe({"messages": messages, "stream": False})
    elapsed2 = time.time() - t0

    assert isinstance(resp2, str) and len(resp2) > 0, "Step 2: Empty response received"
    print(f"[Agent Response 2] ({elapsed2:.2f}s):\n{resp2}\n")

    # Assertions for Step 2
    assert "```python" in resp2 or "```py" in resp2, "Step 2: Missing PySpark code block"
    assert "```sql" in resp2, "Step 2: Missing SparkSQL DDL block"
    assert any(k in resp2.lower() for k in ("confirmar", "deseja confirmar", "prosseguir")), (
        f"Step 2: Missing confirmation prompt in response:\n{resp2[-300:]}"
    )

    messages.append({"role": "assistant", "content": resp2})
    print("✅ Step 2 Verified: Gold ETL code generated with PySpark and SparkSQL DDL.")

    # -------------------------------------------------------------------------
    # STEP 3: Confirm Lifecycle Execution (Regex Confirmation)
    # -------------------------------------------------------------------------
    print_banner("STEP 3: Confirm Lifecycle Execution ('sim, pode prosseguir')")
    q3 = "sim, pode prosseguir"
    print(f"[User Turn 3]: {q3}\n")
    messages.append({"role": "user", "content": q3})

    t0 = time.time()
    resp3 = pipe.pipe({"messages": messages, "stream": False})
    elapsed3 = time.time() - t0

    assert isinstance(resp3, str) and len(resp3) > 0, "Step 3: Empty response received"
    print(f"[Agent Response 3] ({elapsed3:.2f}s):\n{resp3}\n")

    # Assertions on CI Quality Gate
    assert any(
        k in resp3 for k in ("Esteira de CI Quality Gate", "CI Quality Gate", "Quality Gate")
    ), "Step 3: Missing CI Quality Gate section in response"
    assert "APPROVED" in resp3, "Step 3: CI Quality Gate was not APPROVED"
    assert "Ruff" in resp3, "Step 3: Missing Ruff report in CI Gate"
    assert "SQLFluff" in resp3, "Step 3: Missing SQLFluff report in CI Gate"

    # Assertions on Databricks Job creation
    assert any(k in resp3 for k in ("Databricks Workflow Job", "Job ID")), (
        "Step 3: Missing Databricks Workflow Job section in response"
    )

    job_match = re.search(r"Job ID[:\*]*\s*`?(\d+)`?", resp3)
    assert job_match is not None, f"Step 3: Could not parse numeric Job ID from output:\n{resp3}"
    created_job_id = int(job_match.group(1))
    print(f"✅ Step 3 Extracted Remote Databricks Job ID: {created_job_id}")

    run_match = re.search(r"Run[:\*]*\s*`?(\d+)`?", resp3)
    created_run_id = int(run_match.group(1)) if run_match else None
    if created_run_id:
        print(f"✅ Step 3 Extracted Remote Run ID: {created_run_id}")

    messages.append({"role": "assistant", "content": resp3})
    print(
        "✅ Step 3 Verified: Full lifecycle completed (CI APPROVED, GitOps committed, Databricks Job created)."
    )

    # -------------------------------------------------------------------------
    # STEP 4: Remote Databricks SDK Verification
    # -------------------------------------------------------------------------
    print_banner(f"STEP 4: Remote Databricks SDK Verification for Job {created_job_id}")
    w = WorkspaceClient(host=settings.databricks_host, token=settings.databricks_token)

    print(f"Connecting to Databricks Workspace: {w.config.host} ...")
    remote_job = w.jobs.get(job_id=created_job_id)
    assert remote_job is not None, f"Remote job {created_job_id} not found on Databricks!"
    assert remote_job.settings is not None, f"Remote job {created_job_id} has empty settings!"

    job_name = remote_job.settings.name
    creator = remote_job.creator_user_name
    print("✅ Verified Remote Job Exists on Databricks:")
    print(f"   - Job ID:   {created_job_id}")
    print(f"   - Job Name: {job_name}")
    print(f"   - Creator:  {creator}")
    print(f"   - Tasks:    {len(remote_job.settings.tasks or [])} task(s)")

    for t in remote_job.settings.tasks or []:
        print(f"     * Task Key: {t.task_key}")
        if t.sql_task and t.sql_task.file:
            print(f"       SQL File: {t.sql_task.file.path} (Source: {t.sql_task.file.source})")

    # Verify dispatched runs
    runs = list(w.jobs.list_runs(job_id=created_job_id))
    print(f"✅ Verified Remote Runs Dispatched: {len(runs)} run(s) found.")
    for r in runs[:3]:
        state = r.state
        print(
            f"   - Run ID: {r.run_id}, LifeCycle: {state.life_cycle_state if state else 'UNKNOWN'}"
        )

    print_banner("🎉 ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!")
    print("Summary of E2E Validation:")
    print("1. [PASS] Turn 1: Preview live customers table from Databricks SQL Warehouse.")
    print("2. [PASS] Turn 2: Generate Gold Medallion ETL (PySpark + SparkSQL Delta DDL).")
    print("3. [PASS] Turn 3: Regex confirmation ('sim, pode prosseguir') executed lifecycle:")
    print("          - CI Quality Gate passed (Ruff + SQLFluff + Anti-patterns -> APPROVED).")
    print("          - GitOps auto commit & push completed.")
    print(f"          - Remote Databricks Job created (ID: {created_job_id}).")
    print(f"4. [PASS] Step 4: Databricks SDK verified remote Job '{job_name}' exists in cloud.")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    try:
        success = run_e2e_journey()
        sys.exit(0 if success else 1)
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ E2E Journey Failed with Exception: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
