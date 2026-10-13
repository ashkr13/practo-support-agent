"""Milestone 4 tests: the CrewAI crew runs via kickoff() and invokes the right tools (Task 7)."""
import pytest

from app.crew import (
    COMPOSER_AGENT, LOOKUP_AGENT, RETRIEVAL_AGENT, ToolPermissionError, build_agent, build_crew,
    run_crew,
)
from app.mock_llm import MockCrewLLM
from app.rag import build_index
from app.schemas import CrewResponse
from app.tools import AppointmentStatusTool


@pytest.fixture(scope="module", autouse=True)
def index():
    build_index()


def llm():
    return MockCrewLLM(model="mock-react-llm")


def test_crew_has_three_agents():
    crew = build_crew(llm())
    assert [a.role for a in crew.agents] == [RETRIEVAL_AGENT, LOOKUP_AGENT, COMPOSER_AGENT]


def test_policy_query_invokes_rag_tool_only():
    run = run_crew("How many hours before my appointment can I cancel without paying anything?", llm())
    assert [c["tool"] for c in run.tool_calls] == ["rag_lookup"]
    assert run.response.grounded and "4 hours" in run.response.answer


def test_lookup_query_invokes_lookup_tool_only():
    run = run_crew("What is the status of APT-1027?", llm())
    assert [c["tool"] for c in run.tool_calls] == ["check_appointment_status"]
    assert run.response.appointment.status == "Completed"


def test_out_of_scope_query_falls_back():
    run = run_crew("What is the capital of France?", llm())
    assert not run.response.grounded and run.response.answer.startswith("I don't know")


def test_output_validates_against_schema():
    run = run_crew("Can I reschedule APT-1003, and what is the cancellation window?", llm())
    assert isinstance(CrewResponse.model_validate_json(run.raw_output), CrewResponse)
    assert set(run.response.intents) == {"policy", "appointment_lookup"}


def test_only_lookup_agent_may_hold_lookup_tool():
    for role in (RETRIEVAL_AGENT, COMPOSER_AGENT):
        with pytest.raises(ToolPermissionError):
            build_agent(role, "g", "b", [AppointmentStatusTool()], llm())
    build_agent(LOOKUP_AGENT, "g", "b", [AppointmentStatusTool()], llm())  # allowed
