"""Milestone 6 tests: governance (Task 15) and response caching (Task 16)."""
import pytest
from crewai import Agent, Crew, Task

from app import rag
from app.crew import RETRIEVAL_AGENT, ToolPermissionError, audit_crew_tools
from app.governance import RISK_LEVEL, BudgetExceeded, check_budget, enforce_budget
from app.memory import reset_session
from app.mock_llm import MockCrewLLM
from app.pipeline import ask
from app.tools import TOOL_CALLS, AppointmentStatusTool


@pytest.fixture(scope="module", autouse=True)
def index():
    rag.build_index()


def test_risk_level_is_high():
    assert RISK_LEVEL == "High"


def test_audit_catches_agent_built_without_guard():
    llm = MockCrewLLM(model="mock-react-llm")
    rogue = Agent(role=RETRIEVAL_AGENT, goal="g", backstory="b", tools=[AppointmentStatusTool()], llm=llm)
    crew = Crew(agents=[rogue], tasks=[Task(description="x", expected_output="y", agent=rogue)])
    with pytest.raises(ToolPermissionError):
        audit_crew_tools(crew)


def test_budget_allows_normal_and_rejects_oversized():
    assert enforce_budget("How do I cancel?").allowed
    with pytest.raises(BudgetExceeded):
        enforce_budget("x" * 6000)
    assert not check_budget("short question", budget=100).allowed


def test_oversized_request_rejected_before_any_model_or_tool_call():
    before = len(TOOL_CALLS)
    reset_session("budget")
    r = ask("Explain everything. " * 400, "budget")
    assert r.blocked and r.llm_calls == 0 and len(TOOL_CALLS) == before
    assert any(e.startswith("budget_rejected") for e in r.guardrail_events)


def test_actual_usage_within_estimate():
    reset_session("budget")
    r = ask("Can I reschedule APT-1003, and what is the cancellation window?", "budget")
    assert 0 < r.actual_tokens <= r.estimated_tokens


def test_pipeline_attaches_review_verdict():
    reset_session("budget")
    r = ask("How long will my thyroid test results take?", "budget")
    assert r.review is not None and r.review.approved


def test_cache_hit_skips_retrieval_and_generation():
    rag.clear_cache()
    first = rag.grounded_answer("How long will my thyroid test results take?")
    g, ret = rag.GENERATOR_CALLS, rag.RETRIEVE_CALLS
    second = rag.grounded_answer("  how long will my THYROID test results take ")
    assert not first.cache_hit and second.cache_hit
    assert (rag.GENERATOR_CALLS, rag.RETRIEVE_CALLS) == (g, ret)
    assert second.answer == first.answer and rag.CACHE_STATS == {"hits": 1, "misses": 1}


def test_cache_cleared_when_kb_changes():
    rag.grounded_answer("How long will my thyroid test results take?")
    rag.build_index()
    assert rag.CACHE_STATS == {"hits": 0, "misses": 0}
    assert not rag.grounded_answer("How long will my thyroid test results take?").cache_hit
