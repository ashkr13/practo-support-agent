"""CrewAI crew: Retrieval Agent + Lookup Agent + Response Composer (Task 7).

Least autonomy (Task 15, application layer): every agent is built through `build_agent`, which
checks the requested tools against TOOL_ALLOWLIST. Only the Lookup Agent may hold
`check_appointment_status`; wiring it to any other agent raises ToolPermissionError.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app import config  # noqa: F401  (sets zero-network env vars before crewai runs)
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import ValidationError

from app.mock_llm import MockCrewLLM
from app.schemas import CrewResponse
from app.tools import TOOL_CALLS, AppointmentStatusTool, RagLookupTool

RETRIEVAL_AGENT = "Retrieval Agent"
LOOKUP_AGENT = "Lookup Agent"
COMPOSER_AGENT = "Response Composer"

TOOL_ALLOWLIST: dict[str, set[str]] = {
    RETRIEVAL_AGENT: {"rag_lookup"},
    LOOKUP_AGENT: {"check_appointment_status"},
    COMPOSER_AGENT: set(),
}


class ToolPermissionError(PermissionError):
    """Raised when an agent is wired with a tool outside its allow-list."""


def audit_crew_tools(crew: Crew) -> None:
    """Defence in depth: re-check the ASSEMBLED crew, catching agents built without build_agent()."""
    for agent in crew.agents:
        allowed = TOOL_ALLOWLIST.get(agent.role, set())
        for t in agent.tools or []:
            if t.name not in allowed:
                raise ToolPermissionError(
                    f"Least-autonomy violation in assembled crew: '{agent.role}' holds '{t.name}'")


def build_agent(role: str, goal: str, backstory: str, tools: list[BaseTool], llm) -> Agent:
    allowed = TOOL_ALLOWLIST.get(role, set())
    for t in tools:
        if t.name not in allowed:
            raise ToolPermissionError(
                f"Least-autonomy violation: '{role}' may not use tool '{t.name}' "
                f"(allowed: {sorted(allowed) or 'none'})")
    return Agent(role=role, goal=goal, backstory=backstory, tools=tools, llm=llm,
                 allow_delegation=False, verbose=False, max_iter=4)


QUESTION_BLOCK = 'Patient question: """{question}"""'


@dataclass
class CrewRun:
    response: CrewResponse
    raw_output: str
    tool_calls: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    tokens_used: int = 0


def build_crew(llm: MockCrewLLM) -> Crew:
    retrieval = build_agent(
        RETRIEVAL_AGENT,
        goal="Find the Practo clinic policy that answers the patient's question.",
        backstory="You search the clinic policy knowledge base and never answer from memory.",
        tools=[RagLookupTool()], llm=llm)
    lookup = build_agent(
        LOOKUP_AGENT,
        goal="Look up the patient's appointment when they give an appointment ID.",
        backstory="You are the only agent authorised to read appointment records.",
        tools=[AppointmentStatusTool()], llm=llm)
    composer = build_agent(
        COMPOSER_AGENT,
        goal="Combine the policy result and appointment result into one accurate draft answer.",
        backstory="You write clear patient-facing answers using only the results you are given.",
        tools=[], llm=llm)

    retrieval_task = Task(
        description=QUESTION_BLOCK + "\nIf this is a clinic-policy question, call rag_lookup with the "
                    "question and return the tool result unchanged.",
        expected_output="The rag_lookup JSON result, or a skipped marker.", agent=retrieval)
    lookup_task = Task(
        description=QUESTION_BLOCK + "\nIf the question contains an appointment ID, call "
                    "check_appointment_status with it and return the tool result unchanged.",
        expected_output="The check_appointment_status JSON result, or a skipped marker.", agent=lookup)
    compose_task = Task(
        description=QUESTION_BLOCK + "\nUsing ONLY the results in your context, write the final "
                    "draft answer as JSON matching the CrewResponse schema.",
        expected_output="A JSON object matching CrewResponse.", agent=composer,
        context=[retrieval_task, lookup_task],
        output_pydantic=CrewResponse)  # Task 9: structured response_format for the final output

    return Crew(agents=[retrieval, lookup, composer], tasks=[retrieval_task, lookup_task, compose_task],
                process=Process.sequential, cache=False, memory=False, verbose=False)


# One crew run at a time: TOOL_CALLS is a process-wide audit list, so concurrent kickoffs (e.g. two
# API requests) could otherwise read each other's tool calls. Known throughput limitation (README).
_CREW_LOCK = threading.Lock()


def run_crew(question: str, llm: MockCrewLLM | None = None) -> CrewRun:
    """Run the crew via .kickoff() and validate the Composer's output against CrewResponse."""
    with _CREW_LOCK:
        return _run_crew_locked(question, llm)


def _run_crew_locked(question: str, llm: MockCrewLLM | None) -> CrewRun:
    llm = llm or MockCrewLLM(model="mock-react-llm")
    calls_before, tools_before = llm.call_count, len(TOOL_CALLS)
    tokens_before = llm.prompt_tokens + llm.completion_tokens
    crew = build_crew(llm)
    audit_crew_tools(crew)
    result = crew.kickoff(inputs={"question": question})
    raw = result.tasks_output[-1].raw.strip()
    return CrewRun(response=validate_crew_output(raw), raw_output=raw,
                   tool_calls=TOOL_CALLS[tools_before:], llm_calls=llm.call_count - calls_before,
                   tokens_used=llm.prompt_tokens + llm.completion_tokens - tokens_before)


class InvalidCrewOutput(ValueError):
    """The crew produced output that does not conform to CrewResponse."""


def validate_crew_output(raw: str) -> CrewResponse:
    """Task 9: validate the Composer's output against the CrewResponse schema, in our own code.

    CrewAI already parses it via output_pydantic; we re-validate the raw text so a malformed
    response can never reach a patient, regardless of framework behaviour.
    """
    try:
        return CrewResponse.model_validate_json(raw)
    except ValidationError as e:
        raise InvalidCrewOutput(f"Crew output failed schema validation: {e.error_count()} error(s)\n{e}") from e
