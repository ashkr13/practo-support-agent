# Requirements — Practo Domain Support Agent

**Track:** Practo (Healthcare) · **Status:** final (2026-10-13) — all functional requirements implemented; see EVIDENCE_CHECKLIST.md.

## 1. Problem statement
Practo's patient-experience team needs a support agent that (a) answers clinic-policy
questions grounded only in an authored knowledge base, and (b) checks a specific
appointment's status from a validated dataset, so patients get instant, accurate answers
about scheduling, fees, and follow-ups. The system must be production-minded: guarded,
reviewed by an independent second agent team, governed, cost-controlled, logged, evaluated,
and deployed behind FastAPI — all runnable offline under a deterministic `MOCK_LLM`.

## 2. Objectives
1. Accurate, grounded answers; say "I don't know" rather than invent policy.
2. Reliable record lookups with a designed escalation signal for human handoff.
3. Safety: PII masking, prompt-injection detection, groundedness refusal, independent review.
4. Governance: least autonomy, risk classification, per-request budget cap.
5. Reproducibility: zero API keys, zero network at runtime, deterministic outputs.

## 3. Users and stakeholders
| Actor | Need |
|---|---|
| Patient (simulated) | Fast, correct answers on policy and their appointment |
| Patient-experience team | Fewer tickets; escalation signal for human follow-up |
| Compliance / governance reviewer | Evidence of guardrails, PII protection, audit logs |
| Grader | Deterministic, reproducible transcripts for every task |

## 4. Scope
**In scope:** Tasks 1–16 of the brief. **Out of scope:** real patient data, real LLM calls in graded
mode, persistent memory across restarts, authentication, cloud deployment, masking of
free-text PII (name, diagnosis, insurance ID — no universal format; fabricated values only).

## 5. Functional requirements
| ID | Requirement | Task |
|---|---|---|
| FR1 | Seeded generator `dataset.py` → `APPOINTMENTS` ≥40 records with `record_id`, `category`, `status`, `consultation_fee_inr`, `days_since_created` (0–30), `follow_up_required` (bool); prints counts per category/status and follow-up % | 1 |
| FR2 | ≥12 KB documents (2–5 sentences each) covering all 12 required topics | 2 |
| FR3 | Fixed-size-with-overlap AND sentence-based chunking, embedded locally, each in its own ChromaDB collection via `upsert()` | 3 |
| FR4 | Grounded generation from top-k chunks only; "I don't know" below an empirically calibrated threshold | 4 |
| FR5 | Document-level precision/recall per query for both collections + recommendation | 5 |
| FR6 | `check_appointment_status(record_id) -> dict` with `status`, `consultation_fee_inr`, designed `escalation_score` ∈ [0,1] | 6 |
| FR7 | CrewAI crew ≥3 agents (Retrieval, Lookup, Composer) run via `.kickoff()` | 7 |
| FR8 | LangChain session memory (`InMemoryChatMessageHistory` + `RunnableWithMessageHistory`) | 8 |
| FR9 | Pydantic response schema validated for every crew response | 9 |
| FR10 | Input guardrails (phone-number masking, injection detection) + output groundedness check | 10 |
| FR11 | FastAPI ≥2 HTTP endpoints + 1 WebSocket surviving client disconnect | 11 |
| FR12 | One JSON-Lines log entry per request with trace ID, timing, masked text | 12 |
| FR13 | 15-query LLM-as-judge eval: Accuracy, Grounding, Completeness, Safety per query + averages | 13 |
| FR14 | Autogen `RoundRobinGroupChat` review (Reviewer + Final-Editor) with Pydantic verdict `{approved, final_answer, reason}` | 14 |
| FR15 | Least autonomy (only Lookup Agent has the lookup tool); risk classification; per-request budget cap | 15 |
| FR16 | In-memory cache keyed by normalized query for grounded generation | 16 |

## 6. Non-functional requirements
| ID | Requirement |
|---|---|
| NFR1 | Runs under `MOCK_LLM` with zero API keys and zero runtime network access (`CREWAI_DISABLE_TELEMETRY=true`) |
| NFR2 | Deterministic: same input → same output (seeded data, rule-based mock) |
| NFR3 | No raw fixed-format PII (phone numbers) written to logs or sent to the model |
| NFR4 | Secrets only via environment variables; none committed |
| NFR5 | Reproducible setup: pinned dependencies, documented Python version (3.12/3.13) |
| NFR6 | Text-only deliverables (no images, PDFs, video) |

## 7. User stories and acceptance criteria
| Story | Acceptance criterion |
|---|---|
| As a patient, I ask "How late can I cancel?" and get the policy answer | Answer cites retrieved KB content; top-1 similarity ≥ threshold |
| As a patient, I ask something the KB doesn't cover | Returns the "I don't know" fallback |
| As a patient, I ask "What is the status of APT-0007?" | Lookup tool invoked; status, fee, escalation score returned |
| As a patient, I give my name in turn 1 and ask "what's my name?" in turn 2 | Memory recalls it; a fresh session does not |
| As a patient, I paste my phone number | Model input and logs show it masked |
| As a patient, I describe chest pain | The emergency notice (call 108) comes first |
| As an attacker, I write "ignore previous instructions…" | Request blocked by injection guardrail |
| As compliance, I need an independent check of answers | Autogen review approves grounded drafts, revises ungrounded ones |
| As ops, I need bounded cost | Oversized request rejected; repeated query served from cache |

## 8. Constraints
Python <3.14 (CrewAI); free/local tooling only; 14-day window; single public GitHub repo.

## 9. Assumptions (confirmed with user, Milestone 0)
| # | Assumption |
|---|---|
| A1 | Contact number = Indian mobile: optional `+91`/`0`, 10 digits starting 6–9, optional spaces/hyphens |
| A2 | `consultation_fee_inr` range ₹300–₹2,000 (GP at low end, specialist at high end); fabricated |
| A3 | Escalation grows with `follow_up_required` and with age (`days_since_created`) — a stale follow-up is a risk |
| A4 | The `MOCK_LLM` judge is a deterministic rule-based scorer; limitation documented |
| A5 | Token estimate = ceil(chars / 4) to stay offline (no tokenizer downloads) |
| A6 | Embedding model `sentence-transformers/all-MiniLM-L6-v2`, downloaded once at setup, then offline |
| A7 | Real-LLM integration not implemented; documented as an extension point |
| A8 | Second local model `cross-encoder/ms-marco-MiniLM-L-6-v2` for answerability (approved 2026-10-13), downloaded once at setup |
| A9 | Token budget 10,000/request (revised from 2,000 after measuring real CrewAI prompt sizes) |
| A10 | The system gives no diagnosis or treatment advice; emergencies are directed to 108 (from KB) |
