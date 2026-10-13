"""Task 14: Autogen RoundRobinGroupChat review of the CrewAI Composer's draft.

Case 1: real crew draft -> approved unchanged.
Case 2: crew draft + a deliberately injected ungrounded claim -> revised.
Case 3: crew draft with a deliberately altered number -> revised.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.crew import run_crew  # noqa: E402
from app.review import review_draft  # noqa: E402


def crew_draft(question: str):
    run = run_crew(question)
    policy = next(c["result"] for c in run.tool_calls if c["tool"] == "rag_lookup")
    return run.response.answer, policy["context"]


def show(title: str, question: str, draft: str, context: list[dict]) -> None:
    outcome = review_draft(question, draft, context)
    v = outcome.verdict
    print(f"\n{'=' * 90}\n{title}\nQuestion: {question}\nDraft from CrewAI Composer:\n  {draft}")
    print("Group chat (RoundRobinGroupChat, max_turns=2):")
    for line in outcome.transcript:
        print("  " + line.replace("\n", "\n    "))
    print(f"ReviewVerdict -> approved={v.approved}\n  final_answer: {v.final_answer}\n  reason: {v.reason}")
    print(f"  draft unchanged: {v.final_answer == draft}   review LLM calls: {outcome.llm_calls}")


if __name__ == "__main__":
    q1 = "How long will my thyroid test results take?"
    d1, c1 = crew_draft(q1)
    show("CASE 1 - grounded draft (expected: APPROVED unchanged)", q1, d1, c1)

    q2 = "How many hours before my appointment can I cancel without paying anything?"
    d2, c2 = crew_draft(q2)
    injected = d2 + " Patients marked as No-Show also receive a full refund within 1 hour."
    show("CASE 2 - draft with a deliberately INJECTED ungrounded claim (expected: REVISED)", q2, injected, c2)

    q3 = "Is a follow-up visit free if I go back to the same doctor within a week?"
    d3, c3 = crew_draft(q3)
    tampered = d3.replace("within 7 days", "within 30 days")
    assert tampered != d3
    show("CASE 3 - draft with a deliberately ALTERED number, 7 -> 30 days (expected: REVISED)", q3, tampered, c3)
