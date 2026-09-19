"""FastAPI OpenAI-compatible REST server for Databricks Steward Agent."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.graph import create_steward_graph, get_local_chat_client
from src.ci.runner import run_ci_pipeline
from src.config import settings
from src.semantic.compiler import SemanticQueryCompiler
from src.semantic.registry import SemanticRegistry

app = FastAPI(
    title="Databricks Steward Agent API",
    description="GenAI Data Steward & Semantic Engine for Databricks and Open WebUI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_text(content: Any) -> str:
    """Safely extract plain text from string or multimodal list content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content or "")


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = "databricks-steward"
    messages: list[ChatMessage]
    temperature: float | None = 0.1
    stream: bool | None = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]


class SemanticCompileRequest(BaseModel):
    entity_name: str
    metric_names: list[str] = Field(default_factory=list)
    group_by_dims: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    limit: int | None = 50


class CiValidateRequest(BaseModel):
    pyspark_code: str | None = None
    sparksql_code: str | None = None


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "databricks-steward-agent",
        "version": "0.1.0",
        "databricks_host": settings.databricks_host or "mock_mode",
        "local_llm_url": settings.local_llm_base_url,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """List models supported by the service (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "databricks-steward",
                "object": "model",
                "created": 1700000000,
                "owned_by": "databricks-steward",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Any:
    """Process OpenAI-compatible chat completion request using LangGraph with streaming support."""
    try:
        llm = get_local_chat_client(
            base_url=settings.local_llm_base_url,
            model=settings.local_llm_model,
            api_key=settings.local_llm_api_key,
            temperature=request.temperature or settings.local_llm_temperature,
        )
        graph = create_steward_graph(llm=llm)

        # Get last user prompt, or fallback to last message
        prompt = ""
        for m in reversed(request.messages):
            if m.role == "user":
                prompt = _extract_text(m.content)
                if prompt.strip():
                    break

        if not prompt and request.messages:
            prompt = _extract_text(request.messages[-1].content)

        if not prompt:
            prompt = "Olá! Como posso ajudar na governança ou engenharia de dados do Databricks?"

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": prompt}],
                "user_query": prompt,
            }
        )

        reply_text = str(result.get("response") or "")
        if not reply_text:
            messages = result.get("messages", [])
            for m in reversed(messages):
                if (
                    hasattr(m, "content")
                    and m.content
                    and getattr(m, "type", "") in ("ai", "AIMessage")
                ):
                    reply_text = str(m.content)
                    break
                elif isinstance(m, dict) and m.get("role") == "assistant":
                    reply_text = str(m.get("content", ""))
                    break

        if not reply_text:
            reply_text = "Solicitação processada pelo Databricks Steward Agent."

        model_name = request.model or "databricks-steward"
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        # Support streaming SSE if requested by client (e.g. Open WebUI)
        if request.stream:

            def _sse_generator():
                first_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first_chunk)}\n\n"

                chunk_size = 16
                for i in range(0, len(reply_text), chunk_size):
                    delta_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": reply_text[i : i + chunk_size]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(delta_chunk)}\n\n"

                stop_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return ChatCompletionResponse(
            id=chunk_id,
            created=created_ts,
            model=model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_text),
                    finish_reason="stop",
                )
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/semantic/models")
def get_semantic_models() -> dict[str, Any]:
    """Retrieve all loaded semantic domains and entities."""
    registry = SemanticRegistry(settings.semantic_models_path)
    domains = [
        getattr(d, "domain", getattr(d, "name", "unknown")) for d in registry.domains.values()
    ]
    entities = [getattr(e, "name", "unknown") for e in registry.entities.values()]
    return {"domains": domains, "entities": entities}


@app.post("/api/semantic/compile")
def compile_semantic_query(req: SemanticCompileRequest) -> dict[str, Any]:
    """Compile semantic terms to SparkSQL."""
    registry = SemanticRegistry(settings.semantic_models_path)
    compiler = SemanticQueryCompiler(registry)
    sql = compiler.compile_query(
        entity_name=req.entity_name,
        metric_names=req.metric_names,
        group_by_dims=req.group_by_dims,
        filters=req.filters,
        limit=req.limit,
    )
    return {"sql": sql}


@app.post("/api/ci/validate")
def validate_code_ci(req: CiValidateRequest) -> dict[str, Any]:
    """Run data engineering CI quality gate on PySpark / SparkSQL."""
    report = run_ci_pipeline(
        pyspark_code=req.pyspark_code,
        sparksql_code=req.sparksql_code,
    )
    return {
        "is_approved": report.is_approved,
        "ruff_status": report.ruff_status,
        "sqlfluff_status": report.sqlfluff_status,
        "summary": report.summary_markdown,
        "violations": [
            {
                "rule": v.rule,
                "line": v.line,
                "message": v.message,
                "severity": v.severity,
            }
            for v in (report.violations + report.anti_patterns)
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
