# Requirements → Implementation → Test Evidence Checklist

Every acceptance criterion from the brief, where it is implemented, which automated tests check it, and which
committed transcript demonstrates it. All evidence is produced under `MOCK_LLM` with zero API keys.
Test suite: **126 passed** (`.venv/bin/python -m pytest -q`), also verified in a fresh virtualenv built only from
`requirements.txt`.

## Part 1 — Dataset & RAG core

| # | Acceptance criterion | Implementation | Tests | Transcript |
|---|---|---|---|---|
| 1 | ≥40 seeded records; every category ≥3, every status ≥1; follow-up 10–30 %; choices in README | `dataset.py` (`SEED=7`, weights, `validate()`) | `test_dataset_and_kb.py::test_dataset_passes_all_thresholds`, `::test_dataset_is_deterministic`, `::test_dataset_structure`, `::test_validate_detects_violation` | `task01_dataset_report.txt` (50 records, 24.0 %) |
| 2 | ≥12 KB docs covering every topic, 2–5 sentences | `kb/01…12_*.md`, `app/kb.py` | `test_dataset_and_kb.py::test_kb_has_12_docs_each_2_to_5_sentences`, `::test_kb_covers_every_required_topic` | — |
| 3 | Two chunking strategies, embedded, two separate Chroma collections, sensible retrieval | `app/rag.py` `chunk_fixed`, `chunk_sentences`, `index_chunks` (`upsert`) | `test_rag.py::test_fixed_chunks_have_size_and_overlap`, `::test_two_separate_populated_collections`, `::test_both_collections_retrieve_sensibly` | `task03_index_build.txt` |
| 4 | Threshold calibrated empirically (≥3 in, ≥2 out); ≥5 in-scope + 1 OOS demo with fallback | `scripts/task04_calibrate_threshold.py`; `config.SIMILARITY_THRESHOLD`; `rag.grounded_answer` | `test_rag.py::test_out_of_scope_triggers_fallback`, `::test_in_scope_answers_are_grounded_in_context` | `task04_threshold_calibration.txt`, `task04_grounded_generation.txt` |
| 5 | Doc-level P/R for BOTH collections on the same queries, per-query arithmetic, cited recommendation | `eval/chunking_eval.py`, `scripts/task05_chunking_eval.py` | `test_chunking_eval.py` (4 tests) | `task05_chunking_evaluation.txt` |

## Part 2 — Orchestration, tools, memory, guardrails

| # | Acceptance criterion | Implementation | Tests | Transcript |
|---|---|---|---|---|
| 6 | `check_appointment_status` looks up a record; designed, justified `escalation_score` (not bare OR) | `app/tools.py` `escalation_score`, `ESCALATION_THRESHOLD=0.72` (P40 of days) | `test_tools.py::test_score_formula_and_range`, `::test_score_is_not_a_bare_boolean_or`, `::test_threshold_matches_dataset_percentile`, `::test_lookup_found_and_fields` | `task06_appointment_lookup.txt` |
| 7 | Crew ≥3 agents (Retrieval, Lookup, Composer); both tools invoked via `.kickoff()` on different queries | `app/crew.py`, `app/mock_llm.py` (`BaseLLM` subclass) | `test_crew.py::test_crew_has_three_agents`, `::test_policy_query_invokes_rag_tool_only`, `::test_lookup_query_invokes_lookup_tool_only`; pitfalls: `test_mock_llm.py` (8 tests) | `task07_crew_kickoff.txt` |
| 8 | Multi-turn memory + separate fresh-session transcript | `app/memory.py` (`InMemoryChatMessageHistory` + `RunnableWithMessageHistory`) | `test_memory_and_pipeline.py::test_memory_carries_appointment_across_turns`, `::test_fresh_session_has_no_state`, `::test_recall_question_uses_history` | `task08_memory_multiturn.txt`, `task08_memory_fresh_session.txt` |
| 9 | Every crew response validates against a Pydantic schema | `app/schemas.py` `CrewResponse`; `output_pydantic`; `crew.validate_crew_output` | `test_crew.py::test_output_validates_against_schema`; `test_memory_and_pipeline.py::test_invalid_crew_output_rejected` (3 cases) | `task09_structured_output.txt` |
| 10 | Input guardrails (PII mask, injection) and output groundedness each fire on a deliberate case | `app/guardrails.py`, `app/pipeline.py` | `test_guardrails.py` (phone formats, non-phones, injections, benign, red flags, groundedness, cross-encoder); `test_memory_and_pipeline.py::test_phone_never_reaches_memory_or_tools`, `::test_injection_blocked_and_not_stored_in_memory`, `::test_output_groundedness_overrides_crew_draft` | `task10_guardrails.txt`, `task10_answerability_calibration.txt`, `task10_groundedness_design.txt` |

## Part 3 — Evaluation, observability, deployment

| # | Acceptance criterion | Implementation | Tests | Transcript |
|---|---|---|---|---|
| 11 | ≥2 HTTP endpoints + 1 WebSocket surviving disconnect, Pydantic models | `app/api.py` (`/health`, `/ask`, `/add-document`, `/ws/chat`; plus UI at `/`, `app/static/index.html`) | `test_api.py` (13 test functions, 19 cases incl. `::test_ui_page_served_and_self_contained`, `::test_websocket_disconnect_mid_conversation_server_survives`) | `task11_12_api_and_logging.txt` (real uvicorn; close code 1006) |
| 12 | One JSONL entry per request with trace ID; no raw fixed-format PII on disk | `app/logging_utils.py`, middleware in `app/api.py` | `test_api.py::test_every_http_request_logs_one_jsonl_entry`, `::test_phone_number_never_reaches_disk` | `task11_12_api_and_logging.txt` |
| 13 | 4 scores per query for 15 queries (every topic + ≥2 OOS/edge) + averages, under MOCK_LLM | `eval/eval_set_b.py`, `eval/judge.py`, `scripts/task13_evaluate.py` | (evaluation script; judge rubric documented in `eval/judge.py`) | **`task13_evaluation_v2_setB_heldout.txt`** (official), plus `…v1_setA.txt`, `…v2_setA_dev.txt`, `…v2_setB_postfix.txt` |

## Part 4 — Resilience & governance

| # | Acceptance criterion | Implementation | Tests | Transcript |
|---|---|---|---|---|
| 14 | Autogen RoundRobin review, approve-unchanged AND revise cases, structured verdicts | `app/review.py` (`max_turns=2`, `custom_message_types`, `output_content_type`) | `test_review.py` (9 tests incl. `::test_custom_message_types_registration_is_required`, `::test_altered_number_is_caught`) | `task14_autogen_review.txt` |
| 15a | Only Lookup Agent can call lookup tool; demonstrated + explained | `app/crew.py` `TOOL_ALLOWLIST`, `build_agent`, `audit_crew_tools` | `test_crew.py::test_only_lookup_agent_may_hold_lookup_tool`; `test_governance_and_cache.py::test_audit_catches_agent_built_without_guard` | `task15_governance.txt` |
| 15b | Risk classification with justification | `app/governance.py` `RISK_LEVEL="High"`, `RISK_JUSTIFICATION` | `test_governance_and_cache.py::test_risk_level_is_high` | `task15_governance.txt` |
| 15c | Budget cap rejects an oversized request | `app/governance.py` `enforce_budget`; `pipeline.ask` | `test_governance_and_cache.py::test_oversized_request_rejected_before_any_model_or_tool_call`, `::test_actual_usage_within_estimate` | `task15_governance.txt` |
| 16 | Cache hit with before/after evidence | `app/rag.py` `_CACHE`, `normalize_query`, `CACHE_STATS`, counters | `test_governance_and_cache.py::test_cache_hit_skips_retrieval_and_generation`, `::test_cache_cleared_when_kb_changes` | `task16_cache.txt` |

## Submission requirements

| Requirement | Evidence |
|---|---|
| README states Practo (Healthcare) track + dataset choices | `README.md` top + §1 |
| Runs under MOCK_LLM, zero API keys | `.env.example` (no keys), `app/config.py` |
| `CREWAI_DISABLE_TELEMETRY=true` set and confirmed | `.env.example`, `app/__init__.py`, README "Zero network", `test_offline.py` (2 tests), `nfr_zero_network_check.txt` |
| No images/PDFs/media | repository is code + text only |
| Every transcript reproducible | `bash scripts/run_all_demos.sh` |
