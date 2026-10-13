# Implementation Plan

Legend: ✅ done · 🔄 in progress · ⏳ pending

**Status (2026-10-13): all milestones complete. 122 tests pass (also in a fresh venv from requirements.txt). Pending: publish to GitHub (user).**

| M | Milestone | Tasks | Completion criteria | Status |
|---|---|---|---|---|
| 0 | Environment & planning docs | Stages 1–3 | venv on native Python 3.12, deps pinned, all imports OK, embedding model cached, docs drafted, smoke tests pass | ✅ |
| 1 | Dataset & knowledge base | 1, 2 | Generator printout satisfies every threshold; ≥12 KB docs, all topics | ✅ |
| 2 | Chunking, indexing, calibration, grounded answers | 3, 4 | 2 collections populated; similarity clusters measured & separated; 5 in-scope + 1 OOS transcript | ✅ |
| 3 | Chunking evaluation | 5 | Per-query P/R arithmetic for both collections + recommendation | ✅ |
| 4 | Lookup tool, mock LLM, crew | 6, 7 | Escalation formula justified; both tools invoked via `.kickoff()` | ✅ |
| 5 | Memory, schema, guardrails | 8, 9, 10 | Memory + fresh-session transcripts; schema validation; each guardrail fires; **red-flag emergency guardrail**; **key-term groundedness check** for near-domain queries | ✅ |
| 6 | Review stage, governance, cache | 14, 15, 16 | Approve + revise verdicts; tool-wiring guard; budget rejection; cache hit evidence | ✅ |
| 7 | FastAPI, WebSocket, JSONL logging | 11, 12 | Integration tests incl. disconnect; logs contain no raw phone numbers | ✅ |
| 8 | Evaluation & final docs | 13, Stage 7 | 15-query scores + averages; evidence checklist; demo guide; viva Q&A | ✅ |

## Dependencies
M1 → M2 → M3 (chooses collection) → M4 → M5 → M6 → M7 → M8.

## Risks
| Risk | Mitigation |
|---|---|
| Library version conflicts | Exact pins; tested install on Python 3.12 |
| CrewAI prompt format drift breaks the mock | Mock parses only the model's own text; regression test for template-leak bug |
| In-scope / OOS similarity overlap | Measure early; tune KB wording/chunk size before fixing threshold |
| Rule-based judge seen as circular | Transparent rubric, adversarial items, limitation stated |

## Deviations log
| Date | Deviation | Reason |
|---|---|---|
| 2026-10-12 | Use uv-managed native arm64 Python 3.12 instead of system Python 3.13 | System 3.13 is an x86_64 (Rosetta) build; PyTorch ships no x86-macOS wheels, so sentence-transformers could not install. System 3.14 is arm64 but CrewAI requires <3.14. |
| 2026-10-12 | ChromaDB telemetry disabled (`anonymized_telemetry=False`) | Not mentioned in brief, but Chroma also phones home by default; required for zero-network |
| 2026-10-12 | Added red-flag emergency guardrail to M5 scope | Dense retrieval ranks "video consultation" above "chest pain" for Q6, so emergency guidance fell outside top-3; life-safety must not depend on retrieval ranking |
| 2026-10-12 | Added key-term groundedness check to M5 scope | Near-domain OOS queries (parking fee, visiting hours, crypto payment) score 0.50-0.66, overlapping in-scope (min 0.586); similarity threshold alone cannot block them |
| 2026-10-12 | Hybrid (dense + keyword) retrieval deferred to future work | Tuning retrieval on the evaluation queries would overfit the test set |
| 2026-10-12 | Least-autonomy guard (`build_agent` + `TOOL_ALLOWLIST`) built in M4 instead of M6 | Cheaper to enforce at construction time than retrofit; demo transcript still produced in M6 |
| 2026-10-12 | `CrewResponse` schema defined in M4; formal Task 9 demo in M5 | Composer needed a stable output contract from the first kickoff |
| 2026-10-12 | Composer strips the patient's question before reading upstream JSON | Prevents a user forging a tool result by pasting JSON into their question |
| 2026-10-12 | Groundedness rule changed from "every question term in retrieved top-3 context" to "no concept absent from the whole KB" (+ similarity threshold) | Strict rule refused 7/11 in-scope dev queries (patients paraphrase); chosen rule: 11/11 answered, 5/5 near-domain refused (dev set; held-out check in Task 13) |
| 2026-10-12 | Blocked (injection) turns are not written to session memory | Prevents an injection attempt from persisting into later turns |
| 2026-10-12 | Token budget raised from 2,000 to 10,000 per request | Measured CrewAI prompts are 2,601-4,074 tokens per crew run (4,219-5,712 for a full turn incl. review); 2,000 would reject every request |
| 2026-10-12 | Added `audit_crew_tools()` before every kickoff | Defence in depth: catches agents constructed directly with `crewai.Agent`, bypassing `build_agent()` |
| 2026-10-12 | Review stage runs on every pipeline answer (not only demos) | Independent second check is part of the production flow; adds 2 mock calls per turn |
| 2026-10-12 | Reviewer normalisation keeps digits | Bug found in testing: "24 hours" was approved against "4 hours" because digits were stripped |
| 2026-10-12 | Zero-network env switches moved to `app/__init__.py` | Bug found: importing `app.rag` before `app.config` let huggingface_hub contact the HF Hub; regression test added and all transcripts regenerated |
| 2026-10-12 | Crew runs serialised with a lock | Shared tool-call audit list is not safe under concurrent API requests; throughput limitation documented |
| 2026-10-12 | `/add-document` rejects injection patterns, masks phones, optional `ADMIN_API_KEY` | Knowledge-base poisoning defence |
| 2026-10-12 | Added `scripts/run_all_demos.sh` | One command regenerates every transcript; verified deterministic outputs are byte-identical across runs |
| 2026-10-13 | v1 groundedness rule (KB-vocabulary check) replaced by v2 (cross-encoder answerability, threshold -1.04) | Held-out eval set A: v1 wrongly refused 5/12 in-scope questions (overfitted to dev set). A semantic word-similarity variant was measured and rejected (ranges overlapped). v2 validated once on new held-out set B (written before v2, sha256 recorded): 13/15 correct |
| 2026-10-13 | Added second local model `cross-encoder/ms-marco-MiniLM-L-6-v2` (user approved download) | Needed to judge "does this passage answer this question"; one-time download via `scripts/download_models.py` |
| 2026-10-13 | Answerability ignores appointment IDs | Mixed lookup+policy questions were refused (-2.36 → +1.33 after fix) |
| 2026-10-13 | Fixed review-stage bug: "not found" lookup results are passed to the reviewer | Found by held-out set B (B14); regression test added; official set-B numbers kept as run |
| 2026-10-13 | Partial-policy fallback wording when a lookup answered but the policy part was refused | Avoids "Rs 1000. I don't know." contradiction in multi-turn answers |
| 2026-10-13 | Judge fix: approved refusal templates recognised in Grounding | E15 (correct block) scored Grounding 1; disclosed in transcript |
| 2026-10-13 | `requirements.txt`: added numpy, websockets (imported directly); removed unused autogen-ext | Verified by fresh-venv install + full test run |

## Pending / future work
* Publish the repository to GitHub (user action).
* Real LLM behind `MOCK_LLM=false`; independent LLM judge; larger evaluation sets.
* Sentence-window or hybrid retrieval; cross-encoder re-ranking for sentence selection.
* Per-request tool-call context (remove the crew lock); persistent memory.
