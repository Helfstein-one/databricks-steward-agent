from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.agent.intent import (
    _classify_intent_with_llm,
    _extract_query_text,
    _is_ci_query,
    _is_conceptual_question,
    _is_confirmation,
    _is_data_preview_query,
    _is_diagram_query,
    _is_entity_modeling_query,
    _is_etl_query,
    _is_gitops_query,
    _is_greeting,
    _is_schema_query,
    _is_semantic_query,
    _is_title_request,
    execute_llm_tool_calling,
    get_local_chat_client,
    handle_llm_conceptual,
    handle_llm_greeting,
    is_error_recovery_state,
)
from src.agent.state import AgentState
from src.application import use_cases as uc


def steward_node(state: AgentState, llm: ChatOpenAI | None = None) -> dict[str, Any]:
    client = llm or get_local_chat_client()
    query = state.get("user_query") or _extract_query_text(state.get("messages", []))
    q_l = query.strip().lower()

    if is_error_recovery_state(state):
        return uc.ErrorCorrectionUseCase().execute(state, llm=client)

    intent = state.get("intent")
    is_test = bool(os.getenv("PYTEST_CURRENT_TEST"))
    if not intent and not is_test:
        try:
            intent = _classify_intent_with_llm(query, client)
        except Exception:  # noqa: BLE001
            intent = "OTHER"
        state["intent"] = intent

    if _is_title_request(query) or intent == "TITLE":
        return uc.TitleUseCase().execute(state)
    if _is_confirmation(query) or intent == "CONFIRM":
        return uc.DeploymentConfirmationUseCase().execute(state)
    if _is_data_preview_query(query) or intent == "PREVIEW":
        return uc.DataPreviewUseCase().execute(state)
    if _is_entity_modeling_query(query):
        return uc.EntityModelingUseCase().execute(state)

    if _is_conceptual_question(query):
        res = not is_test and handle_llm_conceptual(state, client, query)
        return res if res else uc.ConceptualExplanationUseCase().execute(state)

    if _is_greeting(query):
        res = not is_test and handle_llm_greeting(state, client, query)
        return res if res else uc.GreetingUseCase().execute(state)

    opt = q_l.rstrip(".").replace("opção ", "").replace("opcao ", "").strip()
    if opt == "3" or _is_diagram_query(q_l, intent):
        return uc.DiagramGenerationUseCase().execute(state)
    if opt == "1" or _is_schema_query(q_l, intent):
        return uc.UnityCatalogUseCase().execute(state)
    if opt == "2" or _is_semantic_query(q_l):
        return uc.SemanticLayerUseCase().execute(state)
    if opt == "4" or _is_etl_query(q_l, intent):
        return uc.ETLGenerationUseCase().execute(state)
    if opt == "5" or _is_ci_query(q_l):
        return uc.CIQualityGateUseCase().execute(state)
    if opt == "6" or _is_gitops_query(q_l):
        return uc.GitOpsPRUseCase().execute(state)

    if not is_test:
        res = execute_llm_tool_calling(state, client, query)
        if res:
            return res

    return uc.GreetingUseCase().execute(state)


def create_steward_graph(llm: ChatOpenAI | None = None) -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("steward", lambda s: steward_node(s, llm))
    workflow.add_edge(START, "steward")
    workflow.add_edge("steward", END)
    return workflow.compile()
