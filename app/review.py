"""Autogen review stage (Task 14): an independent 2-agent RoundRobinGroupChat reviews the crew's draft.

  task message = question + DRAFT + retrieved CONTEXT (+ appointment record)
  Policy_Compliance_Reviewer -> checks every draft sentence against the context (TextMessage)
  Final_Editor               -> approves unchanged or removes unsupported sentences
                                (StructuredMessage[ReviewVerdict] via output_content_type)

Bounded by max_turns=2 (one turn each). The team must be built with
custom_message_types=[StructuredMessage[ReviewVerdict]], or Autogen raises
"ValueError: Message type ... is not registered".

Under MOCK_LLM both agents use MockReviewClient, a deterministic ChatCompletionClient that applies
explicit verification rules (see `sentence_supported`). It reads only USER-role messages (the task and
the other agent's output), never its own system prompt.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Literal, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import (
    ChatCompletionClient, CreateResult, LLMMessage, ModelCapabilities, ModelInfo, RequestUsage,
)
from pydantic import BaseModel, Field

from app.guardrails import EMERGENCY_NOTICE
from app.kb import split_sentences
from app.messages import ASK_FOR_ID, BUDGET_REFUSAL, INJECTION_REFUSAL, PARTIAL_POLICY_FALLBACK
from app.rag import FALLBACK_ANSWER

REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"


class ReviewVerdict(BaseModel):
    approved: bool = Field(..., description="True if the draft is approved unchanged.")
    final_answer: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


# Fixed, policy-approved template sentences the pipeline itself adds (not factual claims).
SAFE_TEMPLATE_SENTENCES = {
    s for template in (FALLBACK_ANSWER, INJECTION_REFUSAL, BUDGET_REFUSAL, ASK_FOR_ID, PARTIAL_POLICY_FALLBACK)
    for s in split_sentences(template)
} | {"This may be a medical emergency.", EMERGENCY_NOTICE}

_NUMBER = re.compile(r"\d[\d,]*")
_WORD = re.compile(r"[a-z]+")
_COMMON = set("""a an the and or of to in on at by for with is are be it its this that as from can
may will if your you our we not no any""".split())


_TOKEN = re.compile(r"[a-z]+|\d[\d,]*")


def _norm(text: str) -> str:
    """Lower-case word AND number tokens; numbers must survive so "24 hours" != "4 hours"."""
    return " ".join(_TOKEN.findall(text.lower()))


def sentence_supported(sentence: str, context: str) -> bool:
    """A draft sentence is supported if it is a policy template, appears verbatim in the context, or
    (paraphrase) every number in it appears in the context AND >= 80% of its content words do."""
    if sentence in SAFE_TEMPLATE_SENTENCES or _norm(sentence) in _norm(context):
        return True
    context_numbers = set(_NUMBER.findall(context))
    if any(n not in context_numbers for n in _NUMBER.findall(sentence)):
        return False
    words = [w for w in _WORD.findall(sentence.lower()) if w not in _COMMON]
    context_words = set(_WORD.findall(context.lower()))
    return bool(words) and sum(w in context_words for w in words) / len(words) >= 0.8


def build_review_task(question: str, draft: str, context: list[dict], appointment: dict | None) -> str:
    lines = ["REVIEW TASK", f"Question: {question}", "DRAFT ANSWER:", f"<<<{draft}>>>", "RETRIEVED CONTEXT:"]
    lines += [f"[{c['doc_id']}] {c['text']}" for c in context] or ["(none)"]
    if appointment and not appointment.get("found", True):
        # A "not found" lookup is a real tool result: the draft may state it (bug B14, set B).
        lines.append(f"[appointment_record] I couldn't find that appointment: {appointment['error']}")
    elif appointment:
        lines.append(
            f"[appointment_record] Appointment {appointment['record_id']} ({appointment['category']}) is "
            f"currently {appointment['status']}, with a consultation fee of Rs {appointment['consultation_fee_inr']}.")
        if appointment.get("escalate"):
            lines.append("[appointment_record] It has been flagged for follow-up by our patient-experience team.")
    return "\n".join(lines)


def _parse_task(text: str) -> tuple[str, str]:
    draft = re.search(r"<<<(.*?)>>>", text, re.DOTALL)
    context = text.split("RETRIEVED CONTEXT:", 1)[1] if "RETRIEVED CONTEXT:" in text else ""
    return (draft.group(1).strip() if draft else ""), context


class MockReviewClient(ChatCompletionClient):
    """Deterministic offline ChatCompletionClient for the two review agents."""

    def __init__(self, role: Literal["reviewer", "editor"]):
        self.role = role
        self.calls = 0
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = [],
                     tool_choice: Any = "auto", json_output: Any = None,
                     extra_create_args: dict | None = None, cancellation_token: Any = None) -> CreateResult:
        self.calls += 1
        # Only user-role messages: the task and the other agent's output (never the system prompt).
        user_msgs = [m for m in messages if getattr(m, "type", "") == "UserMessage"]
        task = next((str(m.content) for m in user_msgs if "REVIEW TASK" in str(m.content)), "")
        draft, context = _parse_task(task)

        if self.role == "reviewer":
            sentences = split_sentences(draft)
            unsupported = [s for s in sentences if not sentence_supported(s, context)]
            content = "COMPLIANCE_FINDINGS: " + json.dumps(
                {"checked": len(sentences), "unsupported": unsupported})
        else:
            findings_msg = next((str(m.content) for m in reversed(user_msgs)
                                 if getattr(m, "source", "") == REVIEWER_NAME), "")
            findings = json.loads(findings_msg.split("COMPLIANCE_FINDINGS:", 1)[1]) if findings_msg else \
                {"checked": 0, "unsupported": []}
            content = self._verdict(draft, findings).model_dump_json()

        prompt_tokens = sum(len(str(m.content)) for m in messages) // 4
        usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=len(content) // 4)
        self._usage = RequestUsage(prompt_tokens=self._usage.prompt_tokens + usage.prompt_tokens,
                                   completion_tokens=self._usage.completion_tokens + usage.completion_tokens)
        return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

    @staticmethod
    def _verdict(draft: str, findings: dict) -> ReviewVerdict:
        bad = findings["unsupported"]
        if not bad:
            return ReviewVerdict(approved=True, final_answer=draft,
                                 reason=f"All {findings['checked']} sentence(s) are supported by the retrieved "
                                        "context; approved unchanged.")
        kept = [s for s in split_sentences(draft) if s not in bad]
        final = " ".join(kept) if kept else FALLBACK_ANSWER
        return ReviewVerdict(approved=False, final_answer=final,
                             reason=f"Removed {len(bad)} unsupported claim(s) not found in the retrieved "
                                    f"context: {bad}")

    def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                      extra_create_args=None, cancellation_token=None) -> AsyncGenerator[str | CreateResult, None]:
        async def gen():
            yield await self.create(messages, json_output=json_output)
        return gen()

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages, *, tools=[]) -> int:
        return sum(len(str(m.content)) for m in messages) // 4

    def remaining_tokens(self, messages, *, tools=[]) -> int:
        return 32_000 - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore[override]
        return ModelCapabilities(vision=False, function_calling=False, json_output=True)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True, family="unknown",
                         structured_output=True, multiple_system_messages=True)


def build_review_team() -> tuple[RoundRobinGroupChat, MockReviewClient, MockReviewClient]:
    reviewer_client, editor_client = MockReviewClient("reviewer"), MockReviewClient("editor")
    reviewer = AssistantAgent(
        REVIEWER_NAME, model_client=reviewer_client,
        system_message="You check every sentence of the draft against the retrieved context and list any "
                       "claim the context does not support.")
    editor = AssistantAgent(
        EDITOR_NAME, model_client=editor_client, output_content_type=ReviewVerdict,
        system_message="You approve the draft unchanged if the reviewer found no problems; otherwise remove "
                       "the unsupported claims and explain why.")
    team = RoundRobinGroupChat(
        [reviewer, editor], max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]])
    return team, reviewer_client, editor_client


class ReviewOutcome(BaseModel):
    verdict: ReviewVerdict
    transcript: list[str]
    llm_calls: int
    tokens_used: int = 0


async def review_draft_async(question: str, draft: str, context: list[dict],
                             appointment: dict | None = None) -> ReviewOutcome:
    team, rc, ec = build_review_team()
    result = await team.run(task=build_review_task(question, draft, context, appointment))
    transcript = []
    for m in result.messages:
        body = m.content.model_dump_json() if isinstance(m, StructuredMessage) else str(m.content)
        transcript.append(f"[{m.source} | {type(m).__name__}] {body}")
    last = result.messages[-1]
    if not isinstance(last, StructuredMessage) or not isinstance(last.content, ReviewVerdict):
        raise RuntimeError(f"Review team did not end with a ReviewVerdict (got {type(last).__name__})")
    return ReviewOutcome(verdict=ReviewVerdict.model_validate(last.content.model_dump()),
                         transcript=transcript, llm_calls=rc.calls + ec.calls,
                         tokens_used=sum(u.prompt_tokens + u.completion_tokens
                                         for u in (rc.total_usage(), ec.total_usage())))


def review_draft(question: str, draft: str, context: list[dict], appointment: dict | None = None) -> ReviewOutcome:
    """Synchronous wrapper (scripts/tests). Inside an event loop (FastAPI), await review_draft_async."""
    return asyncio.run(review_draft_async(question, draft, context, appointment))

