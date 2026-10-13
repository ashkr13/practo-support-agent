"""Task 15: four-layer governance -- least autonomy, risk classification, runtime token budget."""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewai import Agent, Crew, Task  # noqa: E402

from app.crew import (  # noqa: E402
    LOOKUP_AGENT, RETRIEVAL_AGENT, COMPOSER_AGENT, TOOL_ALLOWLIST, ToolPermissionError, audit_crew_tools,
    build_agent, build_crew,
)
from app.governance import (  # noqa: E402
    BASE_OVERHEAD_TOKENS, PER_QUESTION_TOKEN_MULTIPLIER, RISK_JUSTIFICATION, RISK_LEVEL, check_budget,
)
from app.memory import reset_session  # noqa: E402
from app.mock_llm import MockCrewLLM  # noqa: E402
from app.pipeline import ask  # noqa: E402
from app.tools import TOOL_CALLS, AppointmentStatusTool  # noqa: E402

wrap = lambda t: textwrap.fill(t, 100, initial_indent="  ", subsequent_indent="  ")  # noqa: E731

if __name__ == "__main__":
    llm = MockCrewLLM(model="mock-react-llm")
    print("=" * 100 + "\nAPPLICATION LAYER - principle of least autonomy\n" + "=" * 100)
    print("Tool allow-list (app/crew.py TOOL_ALLOWLIST):")
    for role, tools in TOOL_ALLOWLIST.items():
        print(f"  {role:<18} -> {sorted(tools) or '(no tools)'}")
    print("\nTools actually wired in the production crew:")
    for a in build_crew(llm).agents:
        print(f"  {a.role:<18} -> {[t.name for t in a.tools]}")

    print("\nAttempt 1: wire check_appointment_status to the Retrieval Agent via build_agent()")
    for role in (RETRIEVAL_AGENT, COMPOSER_AGENT):
        try:
            build_agent(role, "g", "b", [AppointmentStatusTool()], llm)
            print("  NOT BLOCKED (unexpected)")
        except ToolPermissionError as e:
            print(f"  BLOCKED: {e}")

    print("\nAttempt 2: bypass build_agent() by constructing a crewai.Agent directly")
    rogue = Agent(role=RETRIEVAL_AGENT, goal="g", backstory="b", tools=[AppointmentStatusTool()], llm=llm)
    rogue_crew = Crew(agents=[rogue], tasks=[Task(description="x", expected_output="y", agent=rogue)])
    try:
        audit_crew_tools(rogue_crew)
        print("  NOT BLOCKED (unexpected)")
    except ToolPermissionError as e:
        print(f"  BLOCKED before kickoff by audit_crew_tools(): {e}")

    print("\nHow the guard works:")
    print(wrap(
        "Only the Lookup Agent may call check_appointment_status, because it is the only agent whose job "
        "needs appointment records. Every agent is created through build_agent(), which compares each "
        "requested tool's name with TOOL_ALLOWLIST[role] and raises ToolPermissionError on any tool outside "
        "that list, so a mis-wiring fails loudly at start-up instead of silently giving an agent extra power. "
        "Because a developer could still construct a crewai.Agent directly, run_crew() also calls "
        "audit_crew_tools() on the fully assembled crew immediately before every kickoff(). In the "
        "production crew the tool is simply never wired to the Retrieval Agent or the Composer; the Composer "
        "has no tools at all and only reads the results it is given. If the Retrieval Agent were "
        "manipulated by a prompt injection, it still could not read appointment records."))

    print("\n" + "=" * 100 + f"\nRISK CLASSIFICATION: {RISK_LEVEL}\n" + "=" * 100)
    print(wrap(RISK_JUSTIFICATION))

    print("\n" + "=" * 100 + "\nRUNTIME LAYER - per-request token budget\n" + "=" * 100)
    print(f"  estimate = {BASE_OVERHEAD_TOKENS} + {PER_QUESTION_TOKEN_MULTIPLIER} x question_tokens "
          f"(question_tokens = ceil(chars/4)); budget = {check_budget('x').budget} tokens\n")
    print("  Normal requests (estimate is checked against actual counted usage):")
    for q in ["How long will my thyroid test results take?",
              "Can I reschedule APT-1003, and what is the cancellation window?",
              "I have chest pain, can I just book a video consultation? " * 8]:
        reset_session("gov")
        r = ask(q, "gov")
        print(f"    q_chars={len(q):4d} estimated={r.estimated_tokens:5d} actual={r.actual_tokens:5d} "
              f"within_estimate={r.actual_tokens <= r.estimated_tokens} blocked={r.blocked}")

    oversized = "Please explain every clinic policy in full detail. " * 120
    tool_calls_before = len(TOOL_CALLS)
    reset_session("gov")
    r = ask(oversized, "gov")
    print(f"\n  Deliberately oversized request ({len(oversized)} chars):")
    print(f"    blocked={r.blocked} estimated={r.estimated_tokens} llm_calls={r.llm_calls} "
          f"tool_calls={len(TOOL_CALLS) - tool_calls_before}")
    print(f"    events: {r.guardrail_events}")
    print(f"    answer: {r.response.answer}")
