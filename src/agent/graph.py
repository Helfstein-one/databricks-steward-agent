from __future__ import annotations
from typing import Any
import os

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from src.agent.state import AgentState
from src.agent.intent import (
    _extract_query_text, _is_title_request, _is_confirmation, _is_data_preview_query, 
    _is_greeting, _is_conceptual_question, _is_entity_modeling_query, _classify_intent_with_llm,
    get_local_chat_client, execute_llm_tool_calling, handle_llm_conceptual, handle_llm_greeting
)
from src.application.use_cases.etl_generation import ETLGenerationUseCase
from src.application.use_cases.diagram_generation import DiagramGenerationUseCase
from src.application.use_cases.ci_quality_gate import CIQualityGateUseCase
from src.application.use_cases.gitops_pr import GitOpsPRUseCase
from src.application.use_cases.deployment_confirmation import DeploymentConfirmationUseCase
from src.application.use_cases.others import (
    TitleUseCase, ConceptualExplanationUseCase, DataPreviewUseCase, EntityModelingUseCase,
    UnityCatalogUseCase, SemanticLayerUseCase, GreetingUseCase
)

def steward_node(state: AgentState, llm: ChatOpenAI | None = None) -> dict[str, Any]:
    client = llm or get_local_chat_client()
    query = state.get("user_query") or _extract_query_text(state.get("messages", []))
    q_l = query.strip().lower()
    
    intent = state.get("intent")
    is_test = bool(os.getenv("PYTEST_CURRENT_TEST"))
    if not intent and not is_test:
        try:
            intent = _classify_intent_with_llm(query, client)
        except Exception:
            intent = "OTHER"
        state["intent"] = intent
        
    if _is_title_request(query) or intent == "TITLE": return TitleUseCase().execute(state)
    if _is_confirmation(query) or intent == "CONFIRM": return DeploymentConfirmationUseCase().execute(state)
    if _is_data_preview_query(query) or intent == "PREVIEW": return DataPreviewUseCase().execute(state)
    if _is_entity_modeling_query(query): return EntityModelingUseCase().execute(state)
    
    if _is_conceptual_question(query):
        if not is_test:
            res = handle_llm_conceptual(state, client, query)
            if res: return res
        return ConceptualExplanationUseCase().execute(state)
        
    if _is_greeting(query):
        if not is_test:
            res = handle_llm_greeting(state, client, query)
            if res: return res
        return GreetingUseCase().execute(state)
        
    is_1, is_2, is_3 = q_l in ("1", "1.", "opcao 1", "opção 1"), q_l in ("2", "2.", "opcao 2", "opção 2"), q_l in ("3", "3.", "opcao 3", "opção 3")
    is_4, is_5, is_6 = q_l in ("4", "4.", "opcao 4", "opção 4"), q_l in ("5", "5.", "opcao 5", "opção 5"), q_l in ("6", "6.", "opcao 6", "opção 6")
    
    if is_3 or intent == "DIAGRAM" or any(k in q_l for k in ("diagram", "diagrama", "erd", "mermaid", "lineage", "fluxo", "desenhar", "relacoes", "relações", "relacionamento", "relacionamentos", "como estão relacionadas")): 
        return DiagramGenerationUseCase().execute(state)
    if is_1 or intent == "SCHEMA" or any(k in q_l for k in ("catalog", "catálogo", "schema", "tabelas")): 
        return UnityCatalogUseCase().execute(state)
    if is_2 or any(k in q_l for k in ("semantic", "semântica", "semantica", "metrica", "dimensao", "ontology")): 
        return SemanticLayerUseCase().execute(state)
    if is_4 or intent == "ETL" or any(k in q_l for k in ("etl", "pipeline", "pyspark", "sparksql", "bronze", "silver", "gold")): 
        return ETLGenerationUseCase().execute(state)
    if is_5 or any(k in q_l for k in ("ci", "esteira", "lint", "ruff", "sqlfluff", "anti-pattern", "validar")): 
        return CIQualityGateUseCase().execute(state)
    if is_6 or "pr" in q_l or "gitops" in q_l or "branch" in q_l or "commit" in q_l or "push" in q_l: 
        return GitOpsPRUseCase().execute(state)
    
        
    
    if not is_test:
        res = execute_llm_tool_calling(state, client, query)
        if res: return res
        
    return GreetingUseCase().execute(state)

def create_steward_graph(llm: ChatOpenAI | None = None) -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("steward", lambda s: steward_node(s, llm))
    workflow.add_edge(START, "steward")
    workflow.add_edge("steward", END)
    return workflow.compile()
