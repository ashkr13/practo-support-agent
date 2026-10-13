"""FastAPI deployment (Task 11) with structured JSON-Lines logging (Task 12).

  GET  /health          liveness + index sizes
  POST /ask             one patient turn (AskRequest -> AskResponse)
  POST /add-document    add a policy document to the knowledge base (AddDocumentRequest -> AddDocumentResponse)
  WS   /ws/chat         real-time multi-turn chat; survives a client disconnecting mid-conversation

Run:  uvicorn app.api:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from app import config, rag
from app.guardrails import detect_injection, mask_pii
from app.kb import Document, split_sentences
from app.logging_utils import log_event, new_trace_id
from app.pipeline import ask
from app.schemas import AgentResponse

MAX_QUESTION_CHARS = 8_000  # hard input cap; the token budget (Task 15) rejects well below this
SESSION_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


# ---------------------------------------------------------------- request / response models

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str = Field("default", pattern=SESSION_PATTERN)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class AskResponse(AgentResponse):
    trace_id: str


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., pattern=r"^[a-z0-9_]{3,64}$", examples=["13_parking_policy"])
    title: str = Field(..., min_length=3, max_length=120)
    text: str = Field(..., min_length=20, max_length=2_000)

    @field_validator("text")
    @classmethod
    def two_to_five_sentences(cls, v: str) -> str:
        n = len(split_sentences(v))
        if not 2 <= n <= 5:
            raise ValueError(f"text must contain 2-5 sentences (got {n})")
        return v


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_in_collection: dict[str, int]
    cache_cleared: bool
    trace_id: str


class HealthResponse(BaseModel):
    status: str
    mock_llm: bool
    collections: dict[str, int]


# ---------------------------------------------------------------- app + logging middleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(rag.build_index, False)  # index kb/ without pruning API-added documents
    await run_in_threadpool(rag.get_embedder)
    yield


app = FastAPI(title="Practo Domain Support Agent", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def jsonl_request_log(request: Request, call_next):
    """Exactly one JSON-Lines entry per HTTP request, with trace ID and latency."""
    trace_id = new_trace_id()
    request.state.trace_id = trace_id
    request.state.log = {}
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Trace-Id"] = trace_id
        return response
    except Exception as exc:  # logged here, then turned into a 500 by the exception handler
        request.state.log["error"] = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        log_event("http_request", trace_id, method=request.method, path=request.url.path,
                  status_code=status, latency_ms=round((time.perf_counter() - start) * 1000, 2),
                  level="ERROR" if status >= 500 else "INFO", **request.state.log)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={
        "detail": "Internal error", "trace_id": getattr(request.state, "trace_id", None)})


def _summary(r: AgentResponse) -> dict:
    """Log fields for one answered turn (request text is already masked)."""
    return {"session_id": r.session_id, "request_text": r.question_masked, "blocked": r.blocked,
            "guardrail_events": r.guardrail_events, "intents": r.response.intents,
            "grounded": r.response.grounded, "sources": r.response.sources,
            "tools_called": r.response.tools_called,
            "review_approved": r.review.approved if r.review else None, "cache_hit": r.cache_hit,
            "estimated_tokens": r.estimated_tokens, "actual_tokens": r.actual_tokens,
            "llm_calls": r.llm_calls}


# ---------------------------------------------------------------- endpoints

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", mock_llm=config.MOCK_LLM,
                          collections={s: rag.get_collection(s).count() for s in rag.STRATEGIES})


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    # Sync endpoint -> FastAPI runs it in a worker thread, so the event loop stays free.
    result = ask(body.question, body.session_id)
    request.state.log.update(_summary(result))
    return AskResponse(trace_id=request.state.trace_id, **result.model_dump())


@app.post("/add-document", response_model=AddDocumentResponse, status_code=201)
def add_document_endpoint(body: AddDocumentRequest, request: Request,
                          x_admin_key: str | None = Header(default=None)) -> AddDocumentResponse:
    request.state.log.update({"doc_id": body.doc_id, "title": body.title, "request_text": body.text[:300]})
    admin_key = os.getenv("ADMIN_API_KEY")
    if admin_key and x_admin_key != admin_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Key")
    if not admin_key:
        request.state.log["warning"] = "ADMIN_API_KEY not set: /add-document is open (local development)"
    # Knowledge-base poisoning defence: a document must not carry instructions to the model.
    if hits := detect_injection(f"{body.title}. {body.text}"):
        request.state.log["rejected"] = f"injection patterns {hits}"
        raise HTTPException(status_code=422, detail=f"Document rejected: contains injection patterns {hits}")
    text, _ = mask_pii(body.text)  # never store contact numbers in the KB
    counts = rag.add_document(Document(body.doc_id, body.title, text))  # also clears the answer cache
    return AddDocumentResponse(doc_id=body.doc_id, chunks_in_collection=counts, cache_cleared=True,
                               trace_id=request.state.trace_id)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """Real-time multi-turn chat. Client sends {"question": "..."}; server replies with AskResponse JSON.

    One JSONL entry per message. A client disconnect at any point (idle or while the answer is being
    computed) is caught, logged and ends only THIS connection; the server keeps serving others.
    """
    session_id = websocket.query_params.get("session_id") or f"ws-{uuid.uuid4().hex[:12]}"
    await websocket.accept()
    log_event("ws_connect", new_trace_id(), session_id=session_id)
    turns = 0
    try:
        while True:
            raw = await websocket.receive_text()
            trace_id, start = new_trace_id(), time.perf_counter()
            try:
                payload = json.loads(raw)
                body = AskRequest(question=payload.get("question", ""), session_id=session_id)
            except (ValueError, ValidationError, AttributeError) as e:
                await websocket.send_json({"error": "invalid message; expected {\"question\": \"...\"}",
                                           "detail": str(e)[:200], "trace_id": trace_id})
                log_event("ws_message", trace_id, session_id=session_id, level="WARNING",
                          status="invalid_message", latency_ms=round((time.perf_counter() - start) * 1000, 2))
                continue
            result = await run_in_threadpool(ask, body.question, session_id)
            turns += 1
            log_event("ws_message", trace_id, turn=turns, status="ok",
                      latency_ms=round((time.perf_counter() - start) * 1000, 2), **_summary(result))
            await websocket.send_json(AskResponse(trace_id=trace_id, **result.model_dump()).model_dump())
    except WebSocketDisconnect as e:
        log_event("ws_disconnect", new_trace_id(), session_id=session_id, turns_completed=turns,
                  close_code=e.code, detail="client disconnected; server continues")
    except RuntimeError as e:  # e.g. sending after the client vanished mid-answer
        log_event("ws_disconnect", new_trace_id(), session_id=session_id, turns_completed=turns,
                  detail=f"connection lost while sending: {e}"[:200], level="WARNING")
    except asyncio.CancelledError:  # connection task cancelled (client gone / server shutting down)
        log_event("ws_disconnect", new_trace_id(), session_id=session_id, turns_completed=turns,
                  detail="connection task cancelled while answering; server continues", level="WARNING")
        raise  # cancellation must propagate
