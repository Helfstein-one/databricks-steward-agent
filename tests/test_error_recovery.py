import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.agent.graph import create_steward_graph, steward_node
from src.agent.intent import is_error_recovery_state
from src.agent.state import AgentState
from src.application.use_cases.ci_quality_gate import CIQualityGateUseCase
from src.application.use_cases.deployment_confirmation import DeploymentConfirmationUseCase
from src.application.use_cases.error_correction import ErrorCorrectionUseCase
from src.ci.report import CIReport, Violation


class TestErrorRecovery(unittest.TestCase):
    def test_agent_state_fields(self) -> None:
        state: AgentState = {
            "messages": [],
            "waiting_for_correction": True,
            "error_recovery": {"source": "ci", "logs": "Error log details"},
        }
        self.assertTrue(state.get("waiting_for_correction"))
        self.assertEqual(state.get("error_recovery", {}).get("source"), "ci")

    def test_router_detects_error_recovery_state(self) -> None:
        state_clean: AgentState = {"messages": [], "waiting_for_correction": False}
        self.assertFalse(is_error_recovery_state(state_clean))

        state_flag: AgentState = {"messages": [], "waiting_for_correction": True}
        self.assertTrue(is_error_recovery_state(state_flag))

        state_rec: AgentState = {"messages": [], "error_recovery": {"logs": "failed"}}
        self.assertTrue(is_error_recovery_state(state_rec))

        failed_report = CIReport(
            is_approved=False,
            ruff_status="FAILED",
            sqlfluff_status="PASSED",
            anti_patterns=[Violation(rule="DP001", message="Data Anti-Pattern")],
        )
        state_rep: AgentState = {"messages": [], "ci_report": failed_report}
        self.assertTrue(is_error_recovery_state(state_rep))

    def test_ci_failure_sets_waiting_for_correction(self) -> None:
        use_case = CIQualityGateUseCase()
        failing_code = {
            "pyspark": "def process(df):\n    df.collect()\n    return df\n",
            "sparksql": "SELECT * FROM workspace.default.medallion_silver_transactions;",
        }
        state: AgentState = {"messages": [], "generated_code": failing_code}
        res = use_case.execute(state)

        self.assertTrue(res.get("waiting_for_correction"))
        self.assertIsNotNone(res.get("error_recovery"))
        self.assertEqual(res["error_recovery"].get("source"), "ci_quality_gate")
        self.assertIn("CI Quality Gate Report", res["response"])

    @patch("src.application.use_cases.deployment_confirmation.deploy_and_materialize_data_product")
    def test_deployment_failure_sets_waiting_for_correction(self, mock_deploy: MagicMock) -> None:
        mock_deploy.return_value = {
            "status": "ci_failed",
            "message": "❌ CI Quality Gate rejeitou o pipeline por anti-patterns.",
            "ci_report": None,
        }
        use_case = DeploymentConfirmationUseCase()
        state: AgentState = {
            "messages": [HumanMessage(content="confirmar")],
            "pending_pipeline": {
                "product_name": "medallion_gold_sales_kpis",
                "pyspark": "df.collect()",
                "sparksql": "SELECT * FROM main.sales.orders",
            },
        }
        res = use_case.execute(state)

        self.assertTrue(res.get("waiting_for_correction"))
        self.assertIsNotNone(res.get("error_recovery"))
        self.assertEqual(res["error_recovery"].get("source"), "deployment")
        self.assertIn("CI Quality Gate rejeitou", res["response"])

    def test_error_correction_use_case_heuristic_fix(self) -> None:
        use_case = ErrorCorrectionUseCase()
        state: AgentState = {
            "messages": [HumanMessage(content="Fix the join e remova select *")],
            "user_query": "Fix the join e remova select *",
            "waiting_for_correction": True,
            "error_recovery": {
                "pyspark": "def process(df):\n    return df\n",
                "sparksql": "SELECT * FROM workspace.default.medallion_silver_transactions;",
            },
            "generated_code": {
                "pyspark": "def process(df):\n    return df\n",
                "sparksql": "SELECT * FROM workspace.default.medallion_silver_transactions;",
            },
        }
        res = use_case.execute(state)

        self.assertFalse(res.get("waiting_for_correction"))
        self.assertIsNone(res.get("error_recovery"))
        self.assertIn("SELECT transaction_id, user_id, amount, status", res["generated_code"]["sparksql"])
        self.assertIn("Correção Aplicada com Sucesso", res["response"])

    def test_error_correction_use_case_delegation_to_jules(self) -> None:
        mock_jules_adapter = MagicMock()
        mock_jules_adapter.ask_jules.return_value = (
            "I analyzed the pipeline. Replace `SELECT *` with explicit column selection `SELECT id, amount`."
        )
        use_case = ErrorCorrectionUseCase(jules_adapter=mock_jules_adapter)

        state: AgentState = {
            "messages": [HumanMessage(content="Ask Jules to fix it")],
            "user_query": "Ask Jules to fix it",
            "waiting_for_correction": True,
            "error_recovery": {
                "source": "ci_quality_gate",
                "logs": "SQLFluff Violation: Avoid SELECT *",
            },
        }
        res = use_case.execute(state)

        mock_jules_adapter.ask_jules.assert_called_once()
        self.assertFalse(res.get("waiting_for_correction"))
        self.assertIn("Jules MCP Server Response", res["response"])
        self.assertIn("Replace `SELECT *`", res["response"])

    def test_steward_node_routes_in_error_recovery_state(self) -> None:
        state: AgentState = {
            "messages": [HumanMessage(content="Fix the join e remova select *")],
            "user_query": "Fix the join e remova select *",
            "waiting_for_correction": True,
            "generated_code": {
                "pyspark": "def process(df):\n    return df\n",
                "sparksql": "SELECT * FROM workspace.default.medallion_silver_transactions;",
            },
        }
        res = steward_node(state)

        self.assertFalse(res.get("waiting_for_correction"))
        self.assertIn("Correção Aplicada com Sucesso", res["response"])

    def test_graph_e2e_error_recovery_loop(self) -> None:
        graph = create_steward_graph()

        # 1. Trigger CI check with failing code
        state_1 = graph.invoke(
            {
                "messages": [HumanMessage(content="validar código na esteira de ci")],
                "user_query": "validar código na esteira de ci",
                "generated_code": {
                    "pyspark": "def process(df):\n    df.collect()\n    return df\n",
                    "sparksql": "SELECT * FROM workspace.default.medallion_silver_transactions;",
                },
            }
        )
        self.assertTrue(state_1.get("waiting_for_correction"))

        # 2. Next message in error state: user sends correction request
        state_2 = graph.invoke(
            {
                **state_1,
                "messages": [
                    HumanMessage(content="validar código na esteira de ci"),
                    HumanMessage(content="Fix the join and remove select * and collect"),
                ],
                "user_query": "Fix the join and remove select * and collect",
            }
        )
        self.assertFalse(state_2.get("waiting_for_correction"))
        self.assertIn("Correção Aplicada com Sucesso", state_2["response"])


if __name__ == "__main__":
    unittest.main()
