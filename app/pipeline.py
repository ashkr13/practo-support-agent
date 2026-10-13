"""End-to-end request pipeline for one patient turn.

  mask PII -> detect injection (block) -> [memory: recall / resolve references]
  -> CrewAI crew -> output groundedness check -> red-flag emergency notice -> schema validation

Blocked turns are NOT written to memory, so an injection attempt cannot poison later turns.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from app.crew import InvalidCrewOutput, run_crew
from app.governance import BudgetExceeded, enforce_budget
from app.guardrails import (
    EMERGENCY_DOC_ID, EMERGENCY_NOTICE, check_groundedness, detect_injection, detect_red_flags, mask_pii,
)
from app.memory import REFERENCE_PATTERN, contextualize, recall_answer, with_memory
from app.mock_llm import compose
from app.rag import FALLBACK_ANSWER
from app.review import review_draft
from app.schemas import AgentResponse, CrewResponse, ReviewSummary

from app.messages import ASK_FOR_ID, BUDGET_REFUSAL, INJECTION_REFUSAL  # noqa: E402


def _respond(question: str, history: list[BaseMessage]) -> AIMessage:
    """One turn with memory available. `question` is already PII-masked and injection-checked."""
    events: list[str] = []

    recalled = recall_answer(question, history)
    if recalled is not None:
        events.append("memory: answered from conversation history")
        final = CrewResponse(answer=recalled, intents=["conversation"], grounded=True)
        return _message(final, question, events, None, 0)

    # Red-flag triage FIRST and deterministically: an emergency answer must not depend on a borderline
    # retrieval or answerability score (found in UI testing: "can I book a video consultation?" with chest
    # pain passed answerability and appended telemedicine advice). Emergency notice only; crew not run.
    flags = detect_red_flags(question)
    if flags:
        events.append(f"red_flag_emergency: {flags} -> emergency notice only, crew skipped")
        final = CrewResponse(answer=f"This may be a medical emergency. {EMERGENCY_NOTICE}",
                             intents=["policy"], grounded=True, sources=[EMERGENCY_DOC_ID])
        context = [{"doc_id": EMERGENCY_DOC_ID, "text": EMERGENCY_NOTICE}]
        outcome = review_draft(question, final.answer, context, None)
        events.append(f"review: {'approved unchanged' if outcome.verdict.approved else 'revised'}")
        return _message(final, question, events, None, outcome.llm_calls,
                        review=outcome.verdict.model_dump(), tokens=outcome.tokens_used)

    resolved, remembered = contextualize(question, history)
    if remembered:
        events.append(f"memory: resolved reference to {remembered}")

    try:
        run = run_crew(resolved)
    except InvalidCrewOutput as e:
        events.append(f"schema_validation_failed: {str(e).splitlines()[0]}")
        final = CrewResponse(answer=FALLBACK_ANSWER, intents=["refused"], grounded=False)
        return _message(final, resolved, events, remembered, 0)

    final = run.response
    results = [c["result"] for c in run.tool_calls]
    policy = next((r for r in results if r.get("kind") == "policy_lookup"), None)
    if policy is not None:
        verdict = check_groundedness(resolved, policy)
        if not verdict.supported:
            events.append(f"groundedness_refusal: {verdict.reason}")
            if policy["grounded"]:  # crew drafted an answer; the output guardrail overrides it
                results = [dict(r, grounded=False) if r is policy else r for r in results]
                final = CrewResponse.model_validate(compose(results))

    if REFERENCE_PATTERN.search(question) and remembered is None and final.appointment is None:
        final = final.model_copy(update={"answer": f"{final.answer} {ASK_FOR_ID}"})


    # Task 14: independent Autogen review of the draft against the retrieved context.
    context = policy["context"] if policy else []
    appt = next((r for r in results if r.get("kind") == "appointment_status"), None)  # found or not
    outcome = review_draft(resolved, final.answer, context, appt)
    review = outcome.verdict
    events.append(f"review: {'approved unchanged' if review.approved else 'revised'}")
    if not review.approved:
        final = final.model_copy(update={"answer": review.final_answer})

    final = CrewResponse.model_validate(final.model_dump())  # Task 9: re-validate after guardrail edits
    return _message(final, resolved, events, remembered, run.llm_calls + outcome.llm_calls,
                    review=review.model_dump(), tokens=run.tokens_used + outcome.tokens_used,
                    cache_hit=bool(policy and policy.get("cache_hit")))


def _message(final: CrewResponse, resolved: str, events: list[str], remembered: str | None,
             llm_calls: int, review: dict | None = None, tokens: int = 0, cache_hit: bool = False) -> AIMessage:
    return AIMessage(content=final.answer, response_metadata={"pipeline": {
        "response": final.model_dump(), "resolved_question": resolved, "events": events,
        "remembered_record_id": remembered, "llm_calls": llm_calls, "review": review,
        "tokens": tokens, "cache_hit": cache_hit}})


_CHAIN = with_memory(_respond)


def ask(question: str, session_id: str = "default") -> AgentResponse:
    masked, n_phones = mask_pii(question)
    events = [f"pii_masked: {n_phones} phone number(s)"] if n_phones else []

    # Task 15 runtime layer: reject BEFORE any model call if the projected cost is over budget.
    try:
        budget = enforce_budget(masked)
    except BudgetExceeded as e:
        events.append(f"budget_rejected: {e.check.describe()}")
        return AgentResponse(
            session_id=session_id, question_masked=masked[:200] + ("..." if len(masked) > 200 else ""),
            resolved_question="", blocked=True, estimated_tokens=e.check.estimated_tokens,
            response=CrewResponse(answer=BUDGET_REFUSAL, intents=["refused"], grounded=False),
            guardrail_events=events)

    injection = detect_injection(masked)
    if injection:
        events.append(f"prompt_injection_blocked: {injection}")
        return AgentResponse(
            session_id=session_id, question_masked=masked, resolved_question=masked, blocked=True,
            response=CrewResponse(answer=INJECTION_REFUSAL, intents=["refused"], grounded=False),
            guardrail_events=events, estimated_tokens=budget.estimated_tokens)

    msg = _CHAIN.invoke({"question": masked}, config={"configurable": {"session_id": session_id}})
    meta = msg.response_metadata["pipeline"]
    return AgentResponse(
        session_id=session_id, question_masked=masked, resolved_question=meta["resolved_question"],
        response=CrewResponse.model_validate(meta["response"]),
        guardrail_events=events + meta["events"], remembered_record_id=meta["remembered_record_id"],
        llm_calls=meta["llm_calls"],
        review=ReviewSummary(**meta["review"]) if meta.get("review") else None,
        estimated_tokens=budget.estimated_tokens, actual_tokens=meta.get("tokens", 0),
        cache_hit=meta.get("cache_hit", False))
