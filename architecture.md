# Architecture

**Status:** final — reflects the implemented system (2026-10-13).

## 1. Request flow
```
client ──► FastAPI  (POST /ask · POST /add-document · WS /ws/chat)
             │  trace_id + timer start
             ▼
      [Runtime governance] token-budget cap ── reject if over budget
             ▼
      [Input guardrails] mask phone numbers ─► detect prompt injection ── block if injection
             ▼
      [Session memory] LangChain InMemoryChatMessageHistory per session_id
             ▼
      ┌──────────── CrewAI crew (MockLLM ⊂ crewai BaseLLM) ────────────┐
      │ Retrieval Agent ── rag_lookup ──► cache ──► ChromaDB (chosen)  │
      │ Lookup Agent ── check_appointment_status ──► dataset.py        │  only holder of tool
      │ Response Composer ──► draft (Pydantic)                         │
      └────────────────────────────────────────────────────────────────┘
             ▼
      [Output guardrail] groundedness: similarity ≥ calibrated threshold AND cross-encoder answerability ≥ -1.04
      [Red-flag guardrail] emergency symptoms → KB emergency notice (call 108) first
             ▼
      Autogen RoundRobinGroupChat (max_turns=2, deterministic mock client)
        Policy_Compliance_Reviewer ─► Final_Editor ─► Verdict{approved, final_answer, reason}
             ▼
      validated AgentResponse ──► client      +  one JSONL log line (masked text, trace_id, latency)
```

## 2. Components
| Component | File | Responsibility |
|---|---|---|
| Dataset | `dataset.py` | Seeded generator + validation report |
| Knowledge base | `kb/*.md` | 12+ policy documents |
| RAG core | `app/rag.py` | Two chunkers, embedding, ChromaDB indexing, retrieval, grounded answer, cache |
| Tools | `app/tools.py` | `rag_lookup`, `check_appointment_status` + escalation score |
| Mock LLMs | `app/mock_llm.py` | CrewAI `BaseLLM` subclass; Autogen `ChatCompletionClient` mock |
| Crew | `app/crew.py` | Agents, tasks, kickoff, least-autonomy guard |
| Memory | `app/memory.py` | Per-session history |
| Schemas | `app/schemas.py` | Pydantic models (API, crew response, review verdict) |
| Guardrails | `app/guardrails.py` | PII masking, injection detection, red flags, groundedness (similarity + cross-encoder) |
| Governance | `app/governance.py` | Token budget, risk classification (tool allow-list lives in `app/crew.py`) |
| Review | `app/review.py` | Autogen two-agent review team |
| API & logs | `app/api.py`, `app/logging_utils.py` | FastAPI endpoints, JSONL logger |
| Evidence | `scripts/`, `transcripts/`, `eval/`, `tests/` | Demos, outputs, evaluation, automated tests |

## 3. Technology choices
| Choice | Why | Alternative rejected |
|---|---|---|
| SentenceTransformers `all-MiniLM-L6-v2` | Free, local, small (384-dim), strong for short sentences | OpenAI embeddings — paid key, network |
| ChromaDB `PersistentClient` | Embedded, file-based, no server | Pinecone/Weaviate — accounts, infra |
| CrewAI + custom `BaseLLM` | Required by brief; official extension point | Patching litellm — fragile |
| Autogen `RoundRobinGroupChat` | Required by brief; independent second team | Extra CrewAI agent — not independent |
| LangChain in-memory history | Required by brief; sufficient for one process | Redis/LangGraph checkpointing — unnecessary |
| FastAPI + Pydantic | Typed contracts, native WebSocket support | Flask — no built-in validation |
| Rule-based deterministic mocks | Reproducible, offline, auditable | Local LLM (Ollama) — non-deterministic, heavy |
| Cross-encoder `ms-marco-MiniLM-L-6-v2` for answerability | Reads question+passage together; separates near-domain questions the bi-encoder cannot | KB-vocabulary rule (v1) — overfitted, 5/12 false refusals held-out |

## 4. Security & privacy
Phone numbers masked before the model, memory, and logs; injection patterns blocked; no secrets
in code (`.env` git-ignored); fabricated data only; telemetry disabled.

## 5. Testing approach
`pytest` unit tests per module, integration tests via FastAPI `TestClient` (HTTP + WebSocket),
and end-to-end demo scripts writing transcripts to `transcripts/`.

## 6. Crew internals (implemented, Milestone 4)
Sequential process, 3 tasks; every agent shares one `MockCrewLLM` (subclass of `crewai.llms.base_llm.BaseLLM`).
```
Retrieval task  → mock decides policy intent → Action: rag_lookup {"query"}  → Observation (JSON) → Final Answer
Lookup task     → mock finds APT-#### ID      → Action: check_appointment_status {"record_id"} → Final Answer
Compose task    → context = both outputs      → mock merges JSON (patient text stripped) → CrewResponse JSON
```
* **ReAct mode:** `BaseLLM` has no `supports_function_calling`, so CrewAI uses its text loop. After a tool runs, CrewAI
  appends `Observation: <result>` to the model's **own assistant message**; the mock reads observations only there
  (the system prompt's template text `Observation: the result of the action` is never parsed).
* **Dispatch:** by each tool's `args_schema` fields (`{"query"}` vs `{"record_id"}`), never by tool name.
* **Least autonomy:** `build_agent()` rejects any tool outside `TOOL_ALLOWLIST[role]`.
* **CrewAI tool cache disabled** (`cache=False`) so the project's own response cache (Task 16) is the only cache.

## 7. Escalation score (Task 6)
`score = 0.6·follow_up_required + 0.4·(days_since_created/30)`; escalate if `score > 0.72`
(= 0.6 + 0.4·P40(days)/30, P40 = 9.2 days) → 5/50 records (10%).

## 8. Pipeline, memory and guardrails (implemented, Milestone 5) — `app/pipeline.py`
```
raw text ─► mask_pii (phone → [PHONE_REDACTED]) ─► detect_injection ──blocked──► refusal (NOT stored in memory)
                                                        │ clean
               RunnableWithMessageHistory(session_id) ─► _respond(masked question, history)
                 ├─ recall question? → answer from history (no crew call)
                 ├─ contextualize: "that appointment"/"it" → last APT-#### in history
                 ├─ run_crew(resolved) → CrewResponse (output_pydantic + validate_crew_output)
                 ├─ check_groundedness (v2): similarity ≥ threshold AND cross-encoder answerability ≥ -1.04
                 │    (appointment IDs stripped first) → else fallback / partial-policy fallback
                 ├─ red flags (chest pain, breathing, bleeding, unconsciousness, stroke) → KB emergency notice first
                 └─ re-validate CrewResponse → AIMessage (stored in history)
             ─► AgentResponse {question_masked, resolved_question, response, guardrail_events, ...}
```
Error handling: schema-invalid crew output → `InvalidCrewOutput` → safe fallback + `schema_validation_failed` event.

## 9. Review, governance, cache (implemented, Milestone 6)
* **Review (`app/review.py`):** `RoundRobinGroupChat([Policy_Compliance_Reviewer, Final_Editor], max_turns=2,
  custom_message_types=[StructuredMessage[ReviewVerdict]])`; both agents use `MockReviewClient`
  (a deterministic `ChatCompletionClient`). Reviewer: a draft sentence is supported if it is a fixed template,
  appears verbatim (words + numbers) in the context, or all its numbers and ≥80% of its content words are in the
  context. Editor: approve unchanged, or drop unsupported sentences (fallback if none remain).
* **Governance (`app/governance.py`):** risk = High (medical data); budget pre-flight
  `6000 + 12 × ceil(chars/4)` ≤ 10,000 tokens, else rejected before any model/tool call; actual usage counted
  by the mock clients and reported per response.
* **Cache (`app/rag.py`):** dict keyed by `(normalize_query(q), strategy, k, threshold)` in front of
  retrieval + generation; cleared on `build_index`/`add_document` (and the groundedness vocabulary is refreshed).

## 10. API contracts (implemented, Milestone 7) — `app/api.py`
| Endpoint | Request | Response | Errors |
|---|---|---|---|
| `GET /health` | – | `HealthResponse{status, mock_llm, collections}` | – |
| `POST /ask` | `AskRequest{question (1–8000 chars, not blank), session_id ^[A-Za-z0-9_-]{1,64}$}` | `AskResponse` = `AgentResponse` + `trace_id` | 422 validation |
| `POST /add-document` | `AddDocumentRequest{doc_id ^[a-z0-9_]{3,64}$, title, text (2–5 sentences)}`; header `X-Admin-Key` if `ADMIN_API_KEY` set | 201 `AddDocumentResponse{doc_id, chunks_in_collection, cache_cleared, trace_id}` | 401 key, 422 validation/injection |
| `WS /ws/chat?session_id=` | text frames `{"question": "..."}` | `AskResponse` JSON per message; `{"error": ...}` for malformed frames | disconnect → logged, connection ends, server continues |

**Logging (`app/logging_utils.py`):** one JSON line per HTTP request (middleware) and per WebSocket message/connect/
disconnect: `timestamp, level, service, event, trace_id, latency_ms, path/status_code or session_id, request_text
(masked), guardrail_events, intents, grounded, tools_called, review_approved, cache_hit, tokens`. Every string is
re-masked by the logger. `X-Trace-Id` response header matches the log entry.

**Concurrency:** sync endpoints run in FastAPI's thread pool; crew kickoffs are serialised by a lock.
