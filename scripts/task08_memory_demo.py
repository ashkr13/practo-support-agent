"""Task 8: session memory. Writes two transcripts: a multi-turn session where state carries across
turns, and a separate fresh session where that state is correctly absent."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory import get_session_history, reset_session  # noqa: E402
from app.pipeline import ask  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "transcripts"

MULTI_TURN = [
    "What is the status of APT-1003?",
    "Can I reschedule that appointment?",
    "What is its fee?",
    "What did I ask you first?",
]
FRESH = [
    "What did I ask you first?",
    "Can I reschedule that appointment?",
]


def run(session_id: str, turns: list[str], title: str) -> str:
    reset_session(session_id)
    lines = [title, f"session_id = {session_id!r}", ""]
    for i, q in enumerate(turns, 1):
        before = len(get_session_history(session_id).messages)
        r = ask(q, session_id)
        lines += [f"Turn {i} | history before this turn: {before} messages",
                  f"  Patient : {q}",
                  f"  Resolved: {r.resolved_question}",
                  f"  Memory  : remembered_record_id={r.remembered_record_id}  events={r.guardrail_events}",
                  f"  Tools   : {r.response.tools_called}",
                  f"  Agent   : {r.response.answer}", ""]
    lines.append(f"Final history length: {len(get_session_history(session_id).messages)} messages "
                 "(HumanMessage + AIMessage per turn)")
    return "\n".join(lines)


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = run("patient-session-A", MULTI_TURN, "TRANSCRIPT 1 - multi-turn session: state carried across turns")
        b = run("patient-session-B", FRESH, "TRANSCRIPT 2 - fresh session: no state from session A")
    notes = sorted({f"{w.category.__name__}: {str(w.message)[:110]}" for w in caught
                    if "LangChain" in w.category.__name__})
    footer = "\n\nExpected warning (not silenced, per brief):\n  " + "\n  ".join(notes) if notes else ""
    (OUT / "task08_memory_multiturn.txt").write_text(a + footer + "\n")
    (OUT / "task08_memory_fresh_session.txt").write_text(b + footer + "\n")
    print(a, "\n\n" + "=" * 80 + "\n\n" + b + footer)
