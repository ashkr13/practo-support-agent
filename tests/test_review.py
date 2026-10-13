"""Milestone 6 tests: Autogen review stage (Task 14)."""
import pytest

from app.rag import build_index
from app.review import ReviewVerdict, build_review_team, review_draft, sentence_supported
from app.tools import rag_lookup

Q = "How many hours before my appointment can I cancel without paying anything?"


@pytest.fixture(scope="module")
def policy():
    build_index()
    return rag_lookup(Q)


def test_team_is_bounded_to_two_turns():
    team, _, _ = build_review_team()
    assert team._max_turns == 2


def test_custom_message_types_registration_is_required(policy):
    """Without custom_message_types the editor's StructuredMessage is rejected (pitfall in the brief)."""
    import asyncio

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat

    from app.review import EDITOR_NAME, REVIEWER_NAME, MockReviewClient, build_review_task

    reviewer = AssistantAgent(REVIEWER_NAME, model_client=MockReviewClient("reviewer"))
    editor = AssistantAgent(EDITOR_NAME, model_client=MockReviewClient("editor"),
                            output_content_type=ReviewVerdict)
    unregistered = RoundRobinGroupChat([reviewer, editor], max_turns=2)  # no custom_message_types
    task = build_review_task(Q, policy["answer"], policy["context"], None)
    with pytest.raises(Exception, match="not registered"):
        asyncio.run(unregistered.run(task=task))


def test_grounded_draft_approved_unchanged(policy):
    o = review_draft(Q, policy["answer"], policy["context"])
    assert o.verdict.approved and o.verdict.final_answer == policy["answer"]
    assert o.llm_calls == 2 and len(o.transcript) == 3  # task + reviewer + editor


def test_injected_claim_is_removed(policy):
    draft = policy["answer"] + " No-Show patients get a full refund within 1 hour."
    o = review_draft(Q, draft, policy["context"])
    assert not o.verdict.approved
    assert "refund" not in o.verdict.final_answer and o.verdict.final_answer == policy["answer"]


def test_altered_number_is_caught(policy):
    draft = policy["answer"].replace("up to 4 hours", "up to 24 hours")
    o = review_draft(Q, draft, policy["context"])
    assert not o.verdict.approved and "24 hours" not in o.verdict.final_answer


def test_fully_ungrounded_draft_falls_back(policy):
    o = review_draft(Q, "All cancellations are always free.", policy["context"])
    assert not o.verdict.approved and o.verdict.final_answer.startswith("I don't know")


def test_verdict_schema_enforced():
    with pytest.raises(Exception):
        ReviewVerdict(approved=True, final_answer="", reason="x")


def test_sentence_supported_rules():
    ctx = "Follow-ups within 7 days are free of charge."
    assert sentence_supported("Follow-ups within 7 days are free of charge.", ctx)
    assert not sentence_supported("Follow-ups within 30 days are free of charge.", ctx)


def test_not_found_lookup_message_survives_review():
    """Regression (held-out set B, B14): the reviewer must see a 'not found' lookup result."""
    from app.memory import reset_session
    from app.pipeline import ask

    reset_session("b14")
    r = ask("What is the status of APT-7777?", "b14")
    assert "No appointment found with ID APT-7777" in r.response.answer
    assert r.review.approved
