# Demo Guide & Viva Preparation

## A. 10-minute live demo

Preparation (once): follow README §7 Setup, then run the tests.
```bash
.venv/bin/python -m pytest -q
```
Expected: `122 passed` (2 LangChain deprecation warnings are expected).

| Step | Run | What to point out |
|---|---|---|
| 1. Data | `.venv/bin/python dataset.py` | Seed 7 → same 50 records every time; every threshold PASS; 24 % follow-up |
| 2. Calibration | `.venv/bin/python scripts/task04_calibrate_threshold.py` | Two clusters (in-scope ≥0.586, out-of-scope ≤0.162) → threshold in the gap; near-domain questions overlap → why a second check exists |
| 3. Crew | `.venv/bin/python scripts/task07_crew_demo.py` | ReAct trace: Action → real tool execution → Final Answer; Q3 uses BOTH tools |
| 4. Guardrails | `.venv/bin/python scripts/task10_guardrails_demo.py` | Phone masked before model; injection blocked; parking refused despite 0.656 similarity; chest pain → 108 first |
| 5. Review | `.venv/bin/python scripts/task14_review_demo.py` | Approve unchanged vs injected refund claim removed vs "7 → 30 days" caught |
| 6. Governance | `.venv/bin/python scripts/task15_governance_demo.py` | Tool-wiring blocked twice; High risk; oversized request rejected with 0 model calls |
| 7. API live | `.venv/bin/python scripts/task11_live_api_demo.py` | Real uvicorn; WebSocket client drops mid-answer → server keeps serving; log has `[PHONE_REDACTED]` |
| 8. Evaluation | open `transcripts/task13_evaluation_v2_setB_heldout.txt` | Held-out 4.07 / 5.00 / 3.67 / 5.00; the v1→v2 story |

**Story to tell (one line each):** measure before choosing (thresholds, budget) · defence in depth (similarity →
answerability → review) · least autonomy · prove the negative (network-blocked test) · held-out evaluation caught
our own overfitting.

## B. Likely viva questions — with accurate answers

**1. Why does everything run on a mock LLM? Isn't that cheating?**
The brief requires graded transcripts under a deterministic `MOCK_LLM` with zero keys, so results are reproducible
and free. Our mocks are not canned strings: the CrewAI mock emits real ReAct actions that CrewAI parses and
executes, and the composer only uses real tool outputs. The limitation is that intent routing, generation and
judging are rules, not language understanding — stated in README §9.

**2. How did you choose the "I don't know" threshold?**
Measured top-1 cosine for 10 in-scope and 5 out-of-scope queries per collection; in-scope ranged 0.586–0.800 and
out-of-scope 0.031–0.159 (sentence). We took the midpoint of the gap: 0.37 (0.38 for fixed). A 0.6 preset would
have refused 3 of 10 real questions.

**3. Why sentence chunking when fixed-size had higher precision?**
Recall was tied (0.917). Fixed-size's precision lead (1.000 vs 0.917) came from one unused third chunk, while only
67 % of its answer pieces were complete sentences versus 100 %. Users read answers, not precision scores.

**4. Explain the escalation score.**
`0.6 × follow_up_required + 0.4 × days/30`. Follow-up need dominates; age ranks urgency. Threshold 0.72 =
0.6 + 0.4 × P40(days)/30 with P40 = 9.2 days, so a follow-up record escalates once older than 40 % of records —
5/50 (10 %) records, a manageable workload. Age alone is capped at 0.4 and never escalates.

**5. What were the two CrewAI mock pitfalls and how did you avoid them?**
(1) CrewAI's system prompt contains `Observation: the result of the action`; we only read observations from the
model's own assistant messages. (2) A tool named `rag_lookup` contains "lookup"; we dispatch on the tool's argument
schema (`query` vs `record_id`). Both have regression tests, including one showing a naive parser *would* fail.

**6. How do you know there is no network access?**
A test replaces `socket.connect` and runs a full crew kickoff: 0 attempts. With telemetry re-enabled the same test
catches 3 attempts to a telemetry endpoint — so the test can fail. We also found and fixed an import-order bug where
Hugging Face went online, and added a fresh-process regression test.

**7. Why both a similarity threshold and a cross-encoder?**
The bi-encoder embeds question and passage separately, so "parking fee" looks like "consultation fee" (0.656). The
cross-encoder reads them together and scores whether the passage answers the question (−1.48 → refused). The
threshold is cheap and blocks obvious off-topic questions; the cross-encoder handles near-domain ones.

**8. Your first groundedness rule failed. What happened?**
It refused questions containing any word absent from the KB. It passed 11/11 development questions because we kept
adding ignore-words — overfitting. On held-out set A it wrongly refused 5/12 good questions ("far", "money",
"extra"). We kept that result, designed v2 on seen data, wrote a new set B *before* designing v2 (hash recorded),
and ran it once: 13/15 correct behaviour.

**9. What does the Autogen stage add if drafts are already grounded?**
Independence. In normal runs it approves (drafts are extractive). Its value is catching what a generator might
invent: our demo injects a false refund claim and an altered number ("7 → 30 days"); both are removed. It found a
real bug too: digits were stripped during comparison, so "24 hours" matched "4 hours".

**10. Why `max_turns=2` and `custom_message_types`?**
Two agents, one turn each, bounds cost. The editor returns `StructuredMessage[ReviewVerdict]`; without registering
that type the team raises "not registered" — we have a test proving it.

**11. How is least autonomy enforced?**
`TOOL_ALLOWLIST` per role, checked in `build_agent()` and again by `audit_crew_tools()` on the assembled crew before
every kickoff (catches agents built directly with `crewai.Agent`). If the Retrieval Agent were hijacked by an
injection it still could not read appointment records.

**12. Why is the system High risk?**
It processes medical data — appointment records, symptoms, contact details — and wrong answers can delay care. That
is why it has masking, injection blocking, groundedness refusal, emergency triage, independent review, least
autonomy and a budget cap, and never gives diagnosis or treatment advice.

**13. How did you set the token budget?**
We measured: a crew run used 2,601–4,074 prompt tokens and a full turn 4,219–5,712. The pre-flight estimate is
6,000 + 12 × question tokens, capped at 10,000; actual usage is counted and stayed within the estimate. Our first
idea (2,000) would have rejected every request — measuring first prevented that.

**14. How does the cache stay correct?**
Key = normalised query + strategy + k + threshold. It is cleared whenever the KB changes (`build_index`,
`add-document`), so a newly added policy is never hidden behind a cached "I don't know". Only policy answers are
cached — never appointment lookups, which are patient-specific.

**15. How is PII kept out of logs?**
The same `mask_pii` runs on input before the model, memory and tools, and again inside the logger on every field.
A test sends a phone number over HTTP and WebSocket and asserts the raw digits never appear in the log file.

**16. What happens when a WebSocket client disconnects mid-answer?**
Real uvicorn raises `WebSocketDisconnect` on send (close code 1006); the test client cancels the task instead. Both
are caught and logged; only that connection ends. Verified live: new HTTP and WebSocket clients were served after.

**17. What would you improve next?**
Real LLM behind the flag; sentence-window or hybrid retrieval (B12 retrieval miss, parking sentence); cross-encoder
re-ranking for sentence selection (B01/B05/B10); an independent LLM judge; larger evaluation sets; persistent
memory and per-request tool-call context instead of a global lock.

**18. Concurrency?**
FastAPI runs sync endpoints in a thread pool; crew kickoffs are serialised with a lock because the tool-call audit
list is process-wide. Correct, but it limits throughput — documented.

## C. Debugging cheat-sheet
| Symptom | Likely cause | Fix |
|---|---|---|
| `pip` cannot install torch | Intel (x86) Python on Apple silicon | use native arm64 Python 3.12/3.13 |
| CrewAI install fails | Python 3.14 | CrewAI 1.15 needs `<3.14` |
| `Collection ... is empty` | index not built | `.venv/bin/python scripts/build_index.py` |
| Model load error offline | models not cached | `.venv/bin/python scripts/download_models.py` once |
| "unauthenticated requests to the HF Hub" | a script imported a model library before `app` | import any `app.*` module first |
| `ValueError: Message type ... not registered` | review team built without `custom_message_types` | use `build_review_team()` |
