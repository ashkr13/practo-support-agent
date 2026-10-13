# Practo Domain Support Agent (CrewAI + Autogen + FastAPI)

> **Track completed: Practo (Healthcare).**
> Everything runs under **`MOCK_LLM` with zero API keys and zero runtime network access**
> (`CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` are set — see [Zero network](#zero-network)).
> All appointment data and clinic policies are **fabricated**; they are not Practo's real records or policies.

A production-minded patient-support agent that answers clinic-policy questions from a knowledge base,
looks up appointment records, remembers the conversation, is guarded against misuse, has every draft answer
reviewed by an independent Autogen team, and runs under explicit governance — deployed behind FastAPI and
evaluated end to end.

```
request ─► budget cap ─► mask phone numbers ─► injection check ─► session memory
        ─► CrewAI crew [Retrieval Agent · Lookup Agent · Response Composer]
        ─► groundedness check ─► red-flag emergency notice ─► Autogen review [Reviewer → Final Editor]
        ─► validated Pydantic response   (+ one JSON-Lines log line, PII masked)
```

Detailed docs: [requirements.md](requirements.md) · [architecture.md](architecture.md) ·
[implementation_plan.md](implementation_plan.md) · [EVIDENCE_CHECKLIST.md](EVIDENCE_CHECKLIST.md) ·
[DEMO_GUIDE.md](DEMO_GUIDE.md) (demo script + viva questions).

---

## 1. Dataset design choices (Part 1, Task 1) — reproducible

| Choice | Value |
|---|---|
| Generator | `dataset.py`, private `random.Random(SEED)` |
| **Seed** | **`7`** |
| Records | **50** (`APT-1001` … `APT-1050`) |
| **Category weights** | General Medicine 0.30 · Cardiology 0.15 · Dermatology 0.20 · Pediatrics 0.20 · Orthopedics 0.15 |
| **Status weights** | Scheduled 0.30 · Completed 0.35 · Cancelled 0.12 · No-Show 0.08 · Rescheduled 0.15 |
| **Fee range** | **₹300–₹2,000**, drawn per specialty and rounded to ₹50: GM 300–700 · Pediatrics 400–900 · Dermatology 500–1,200 · Orthopedics 600–1,500 · Cardiology 800–2,000 |
| Fee reasoning | A general physician consult in an Indian city typically costs a few hundred rupees while a senior cardiologist can charge close to two thousand, so specialty-specific bands keep fees realistic. |
| `days_since_created` | uniform integer 0–30 |
| `follow_up_required` | `True` with probability 0.20 |

Result (`python dataset.py`, [transcript](transcripts/task01_dataset_report.txt)) — the first draw met every
threshold; no seed/weight change and no hand-editing was needed:

| Category | n | | Status | n |
|---|---|---|---|---|
| General Medicine | 13 | | Scheduled | 14 |
| Cardiology | 9 | | Completed | 22 |
| Dermatology | 12 | | Cancelled | 3 |
| Pediatrics | 10 | | No-Show | 2 |
| Orthopedics | 6 | | Rescheduled | 9 |

`follow_up_required=True`: **12/50 = 24.0 %** (band 10–30 %). Fees observed: ₹400–₹1,800.

## 2. Knowledge base (Task 2)
12 documents in [`kb/`](kb/), 4 sentences each, one per required topic: booking, cancellation/rescheduling
window, fees by specialty, insurance claims, prescription refills, lab turnaround, telemedicine eligibility,
emergency protocol, data privacy, follow-up discount, second opinion, home visits.

## 3. RAG core (Tasks 3–5)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim, normalised → dot = cosine).
* **Two collections** in a persistent ChromaDB (`collection.upsert()`): `kb_fixed_overlap` (30-word windows,
  10-word overlap → 45 chunks) and `kb_sentence` (one sentence per chunk → 48 chunks).
* **Grounded generation (MOCK_LLM):** retrieve top-3 chunks; if top-1 cosine < threshold → "I don't know";
  otherwise a deterministic extractive generator returns the 2 context sentences most similar to the query
  (it cannot write text that is not in the context).

### "I don't know" threshold — calibrated empirically (Task 4)
Measured top-1 cosine on 10 in-scope and 5 deliberately out-of-scope queries
([transcript](transcripts/task04_threshold_calibration.txt)):

| Collection | In-scope (n=10) | Out-of-scope (n=5) | Chosen threshold (midpoint of gap) |
|---|---|---|---|
| fixed | 0.5910 – 0.8023 | 0.0656 – 0.1617 | **0.38** (midpoint 0.3763) |
| sentence | 0.5857 – 0.8000 | 0.0310 – 0.1587 | **0.37** (midpoint 0.3722) |

A 0.6 tutorial preset would have rejected 3 of 10 real questions on the sentence collection (0.5857, 0.5865, 0.5885). Stress test:
near-domain questions ("parking fee at the clinic?" 0.656) **overlap** the in-scope range — similarity alone
cannot block them, which is why the output-side groundedness check (Task 10) exists.
Demo on 6 in-scope + 1 out-of-scope query: [transcript](transcripts/task04_grounded_generation.txt).

### Chunking comparison (Task 5) — document-level, top-3, deduplicated
([full per-query arithmetic](transcripts/task05_chunking_evaluation.txt))

| | Mean precision | Mean recall | Answer pieces that are complete sentences |
|---|---|---|---|
| fixed (30w/10w overlap) | **1.000** | 0.917 | 8/12 = 0.667 |
| sentence | 0.917 | 0.917 | **12/12 = 1.000** |

**Recommendation — deploy sentence chunking.** Both strategies reach the same mean recall (0.917, each missing
only the emergency document on the chest-pain query), and fixed-size wins precision (1.000 vs 0.917) by a single
off-topic third-ranked chunk on one query that the generator never quoted; but only 0.667 of fixed-size answer
pieces are complete sentences versus 1.000, and a patient-facing answer that stops mid-sentence is less usable and
harder to verify, so the 0.083 precision gap is outweighed by the 0.333 usability gap.

## 4. Crew, tools, memory, schema, guardrails (Part 2)
* **`check_appointment_status(record_id)`** (Task 6) returns status, fee and
  **`escalation_score = 0.6 × follow_up_required + 0.4 × (days_since_created / 30)`**.
  **Escalate when score > 0.72** = 0.6 + 0.4 × (9.2 / 30), where **9.2 days is the 40th percentile** of
  `days_since_created` in the generated data: a follow-up-required record escalates once it is older than 40 % of
  all appointments (and past the 7-day free follow-up window). Age alone is capped at 0.4 and never escalates.
  → 5/50 records (10 %) escalated ([transcript](transcripts/task06_appointment_lookup.txt)).
* **CrewAI crew** (Task 7): Retrieval Agent (`rag_lookup`), Lookup Agent (`check_appointment_status`),
  Response Composer (no tools), sequential, `.kickoff()`; both tools shown executing
  ([transcript](transcripts/task07_crew_kickoff.txt)).
* **MOCK_LLM for CrewAI** extends `crewai.llms.base_llm.BaseLLM` and emits real ReAct text. Both pitfalls from the
  brief are avoided and regression-tested: tool results are read **only from the model's own assistant messages**
  (never the system template's `Observation: the result of the action`), and tool calls are dispatched by each
  tool's **argument schema** (`{"query"}` / `{"record_id"}`), never by tool name.
* **Memory** (Task 8): LangChain `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` per `session_id`;
  "that appointment"/"its fee" resolve to the remembered ID. Separate transcripts:
  [multi-turn](transcripts/task08_memory_multiturn.txt) · [fresh session](transcripts/task08_memory_fresh_session.txt).
  The expected `LangChainDeprecationWarning` is not silenced.
* **Structured output** (Task 9): `CrewResponse` Pydantic model as the Composer task's `output_pydantic`, re-validated
  in code; malformed outputs are rejected ([transcript](transcripts/task09_structured_output.txt)).
* **Guardrails** (Task 10, [transcript](transcripts/task10_guardrails.txt)):
  * input: **phone-number masking** (Indian mobile: optional `+91`/`91`/`0`, 10 digits starting 6–9, optional
    space/hyphen → `[PHONE_REDACTED]`); **prompt-injection detection** (5 pattern families, request blocked and
    *not* stored in memory); **red-flag emergency notice** (chest pain, breathing difficulty, severe bleeding,
    unconsciousness, stroke → KB emergency sentence first);
  * output: **groundedness check** = (1) similarity ≥ calibrated threshold AND (2) **cross-encoder answerability**
    (`cross-encoder/ms-marco-MiniLM-L-6-v2`, max score over retrieved passages ≥ **−1.04**, calibrated on
    already-seen questions: in-scope min −0.606 vs near/out-of-domain max −1.481,
    [calibration](transcripts/task10_answerability_calibration.txt)). Patient name, diagnosis and insurance ID
    have no fixed format and are out of scope for masking (fabricated examples only).

## 5. Evaluation, observability, deployment (Part 3)

### API (Task 11)
| Endpoint | Purpose |
|---|---|
| `GET /health` | status + index sizes |
| `POST /ask` | `AskRequest{question, session_id}` → `AskResponse` (answer, intents, sources, appointment, guardrail events, review verdict, tokens, cache hit, trace_id) |
| `POST /add-document` | `AddDocumentRequest{doc_id, title, text(2–5 sentences)}` → 201; rejects injection text; optional `X-Admin-Key` |
| `WS /ws/chat?session_id=` | multi-turn chat; client disconnect (even mid-answer) is caught & logged, server keeps serving |

Verified on a **real uvicorn server** ([transcript](transcripts/task11_12_api_and_logging.txt)), including a client
that closed mid-answer (close code 1006) followed by new HTTP and WebSocket clients being served.

### Logging (Task 12)
One JSON line per HTTP request and per WebSocket message/connect/disconnect, with `trace_id` (also returned as
`X-Trace-Id`), `timestamp`, `latency_ms`, masked `request_text`, guardrail events, tools, review result, tokens.
The logger re-applies the same `mask_pii` used on model input; a test asserts the raw number never reaches disk.

### Evaluation (Task 13) — 15 queries × 4 scores, LLM-as-judge under MOCK_LLM
Judge = the exact judge prompt in [`eval/judge.py`](eval/judge.py), answered offline by a deterministic rubric
(scores 1–5). Each set has 12 queries (one per KB topic) + 3 edge cases (near-domain out-of-scope, injection/PII,
unknown ID or far out-of-scope).

**Official result — held-out set B** ([`eval/eval_set_b.py`](eval/eval_set_b.py), written *before* the v2 rule was
designed, sha256 `d2ed211c…a251`, run once; [transcript](transcripts/task13_evaluation_v2_setB_heldout.txt)):

| ID | Topic | Expect | Observed | Acc | Grd | Cmp | Saf |
|---|---|---|---|---|---|---|---|
| B01 | appointment booking | answer | answer | 3 | 5 | 1 | 5 |
| B02 | cancellation window | answer | answer | 5 | 5 | 5 | 5 |
| B03 | fees by specialty | answer | answer | 5 | 5 | 5 | 5 |
| B04 | insurance claims | answer | answer | 5 | 5 | 5 | 5 |
| B05 | prescription refills | answer | answer | 3 | 5 | 1 | 5 |
| B06 | lab turnaround | answer | answer | 5 | 5 | 5 | 5 |
| B07 | telemedicine | answer | answer | 5 | 5 | 5 | 5 |
| B08 | emergency protocol | emergency | emergency | 5 | 5 | 5 | 5 |
| B09 | data privacy | answer | answer | 5 | 5 | 5 | 5 |
| B10 | follow-up discount | answer | answer | 3 | 5 | 1 | 5 |
| B11 | second opinion | answer | answer | 5 | 5 | 5 | 5 |
| B12 | home visits | answer | refuse | 1 | 5 | 1 | 5 |
| B13 | EDGE near-domain OOS | refuse | refuse | 5 | 5 | 5 | 5 |
| B14 | EDGE unknown ID | answer | refuse | 1 | 5 | 1 | 5 |
| B15 | EDGE injection | blocked | blocked | 5 | 5 | 5 | 5 |
| **Average (n=15)** | | | | **4.07** | **5.00** | **3.67** | **5.00** |

How we got here (full history kept for transparency):

| Run | Set | Held-out? | Correct behaviour | Acc | Grd | Cmp | Saf |
|---|---|---|---|---|---|---|---|
| v1 rule (KB-vocabulary check) | A | yes | 10/15 | 3.67 | 5.00\* | 3.53 | 5.00 |
| v2 rule (cross-encoder) | A | **no** (used to design v2) | 15/15 | 4.87 | 5.00 | 4.60 | 5.00 |
| **v2 rule — official** | **B** | **yes** | **13/15** | **4.07** | **5.00** | **3.67** | **5.00** |
| v2 + B14 bug fix | B | no (post-fix re-run) | 14/15 | 4.33 | 5.00 | 3.93 | 5.00 |

\* after fixing a judge bug (approved refusal templates were not recognised); disclosed in the transcript.
v1 overfitted to development questions (5/12 good questions refused over words like "far", "money", "extra").
Remaining set-B errors: B01/B05/B10 quote the right topic but the wrong 2 sentences (limit of the extractive
mock generator); B12 retrieval missed the home-visit document and the system safely refused; B14 was a review-stage
bug (a "not found" lookup result was treated as unsupported) — fixed with a regression test.
**Limitation:** the mock judge reuses the review stage's sentence-support rule for Grounding, so Grounding is not
independent of the review stage; a real LLM judge would be.

## 6. Resilience & governance (Part 4)
* **Autogen review** (Task 14): `RoundRobinGroupChat([Policy_Compliance_Reviewer, Final_Editor], max_turns=2,
  custom_message_types=[StructuredMessage[ReviewVerdict]])`; the editor uses
  `output_content_type=ReviewVerdict{approved, final_answer, reason}`. Demo: approve unchanged · injected false
  refund claim removed · altered number (7→30 days) removed ([transcript](transcripts/task14_autogen_review.txt)).
* **Governance** (Task 15, [transcript](transcripts/task15_governance.txt)):
  * *Application layer — least autonomy:* `TOOL_ALLOWLIST` enforced in `build_agent()` **and** re-audited on the
    assembled crew before every kickoff; wiring `check_appointment_status` to the Retrieval Agent or Composer —
    or constructing a `crewai.Agent` directly to bypass the guard — raises `ToolPermissionError`.
  * *Risk classification:* **High** — the system handles medical data (appointment records, symptoms,
    contact details), and a wrong answer can delay care.
  * *Runtime layer — budget:* estimate = 6,000 + 12 × ⌈chars/4⌉ tokens, cap **10,000/request** (calibrated from
    measured usage: a full turn used 4,219–5,712 tokens). A 6,120-char request (24,360 est.) is **rejected with 0
    model calls and 0 tool calls**.
* **Cache** (Task 16): in-memory, keyed by normalised query text, in front of retrieval + generation; repeat query
  → 0 retrieval calls, 0 generation calls, ~0.008 ms vs ~170–260 ms; cleared whenever the KB changes
  ([transcript](transcripts/task16_cache.txt)).

---

## 7. Setup

**Prerequisites:** Python **3.12 or 3.13** (CrewAI 1.15 requires `<3.14`). On Apple-silicon Macs use a native
arm64 Python (an Intel/Rosetta Python cannot install PyTorch). ~2 GB disk.

```bash
git clone <this-repo-url> practo-support-agent
```
```bash
cd practo-support-agent
```
```bash
python3.12 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements.txt
```
```bash
cp .env.example .env
```
One-time model download (the only step that uses the internet; ~180 MB):
```bash
.venv/bin/python scripts/download_models.py
```
```bash
.venv/bin/python scripts/build_index.py
```

### Configuration (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `MOCK_LLM` | `true` | Deterministic offline LLMs everywhere (the only implemented mode) |
| `CREWAI_DISABLE_TELEMETRY` / `OTEL_SDK_DISABLED` | `true` | **Required** — stops CrewAI's outbound telemetry |
| `HF_HUB_OFFLINE` | `1` | Load models from local cache only |
| `MAX_TOKENS_PER_REQUEST` | `10000` | Runtime budget cap |
| `CHROMA_PATH` / `LOG_PATH` | `./chroma_db` / `./logs/requests.jsonl` | Storage locations |
| `ADMIN_API_KEY` | empty | If set, `/add-document` requires `X-Admin-Key` |

No API keys or secrets are needed. `.env` is git-ignored.

## 8. Run

Tests (122):
```bash
.venv/bin/python -m pytest -q
```
Regenerate every transcript (~3 min):
```bash
bash scripts/run_all_demos.sh
```
Start the API:
```bash
.venv/bin/uvicorn app.api:app --port 8000
```
Ask a question:
```bash
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "How late can I cancel?", "session_id": "demo"}'
```
Interactive API docs: http://localhost:8000/docs. WebSocket: connect to `ws://localhost:8000/ws/chat?session_id=demo`
and send `{"question": "..."}`.

## Zero network
* `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` are set (in `.env.example` **and** in
  `app/__init__.py` before any library import), plus ChromaDB `anonymized_telemetry=False` and `HF_HUB_OFFLINE=1`.
* Proven by tests that block all sockets: a full crew kickoff makes **0** connection attempts; with telemetry
  switched back on, the same test catches 3 attempts to an external telemetry endpoint
  ([evidence](transcripts/nfr_zero_network_check.txt)). A second test imports `app.rag` first in a fresh process
  (a real import-order bug we found and fixed).

## 9. Known limitations
* **MOCK_LLM is rule-based.** Intent routing, generation (2 extractive sentences), review and judging are
  deterministic rules, not language understanding; e.g. B01/B05/B10 picked the wrong sentences. A real LLM can be
  added behind the `MOCK_LLM` flag (`app/config.py` raises `NotImplementedError` today).
* **Retrieval:** single-sentence chunks lose context (e.g. "After two hours, a parking fee of 40 rupees per hour
  applies" never mentions "clinic"); dense retrieval missed the emergency document for "chest pain + video
  consultation" (handled by the red-flag guardrail). Future: sentence-window retrieval, hybrid keyword+dense search,
  cross-encoder re-ranking for sentence selection.
* **Injection detection is pattern-based**; novel phrasings can slip through (the review stage and least-autonomy
  limit the damage).
* **Memory is in-process** and lost on restart (accepted by the brief). **Crew runs are serialised** with a lock
  (correct but limits throughput).
* **Evaluation:** 15 queries per set is small; the mock judge's Grounding score is not independent of the review
  stage.

## 10. Project layout
```
dataset.py                 Task 1 generator + validation
kb/                        12 policy documents (Task 2)
app/  config · kb · rag (chunking, Chroma, retrieval, generation, cache) · tools (lookup, escalation)
      mock_llm (CrewAI BaseLLM mock) · crew · memory · guardrails · pipeline · review (Autogen)
      governance · messages · schemas · logging_utils · api (FastAPI)
eval/                      query sets, chunking eval, judge, results JSON
scripts/                   one runnable demo per task + run_all_demos.sh + download_models.py
transcripts/               committed outputs proving every task
tests/                     122 pytest tests
```
