"""Session memory (Task 8): LangChain InMemoryChatMessageHistory + RunnableWithMessageHistory.

In-process only: history lives in a dict keyed by session_id and is lost on restart (sufficient
per the brief). Only MASKED text is ever stored, so phone numbers never enter memory.

Note: RunnableWithMessageHistory emits a LangChainDeprecationWarning pointing to LangGraph
persistence. This is expected; the class still works and the brief says not to silence it.
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.tools import normalize_record_id

_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def reset_session(session_id: str) -> None:
    _STORE.pop(session_id, None)


# Phrases that refer back to an appointment mentioned earlier in the conversation.
REFERENCE_PATTERN = re.compile(
    r"\b(that|this|my|the same|the) (appointment|booking|visit)\b|\b(reschedule|cancel|move|check) it\b"
    r"|\bits (status|fee)\b", re.I)

RECALL_PATTERN = re.compile(
    r"\bwhat (did|was) (i|my)\b.*\b(ask|question)\b|\b(previous|earlier|first) question\b", re.I)


def last_record_id(history: list[BaseMessage]) -> str | None:
    """Most recent appointment ID mentioned by the patient or the agent in this session."""
    for msg in reversed(history):
        rid = normalize_record_id(str(msg.content))
        if rid:
            return rid
    return None


def contextualize(question: str, history: list[BaseMessage]) -> tuple[str, str | None]:
    """Resolve 'that appointment' / 'it' to the appointment ID remembered from earlier turns.

    Returns (resolved_question, remembered_record_id or None).
    """
    if normalize_record_id(question) or not REFERENCE_PATTERN.search(question):
        return question, None
    rid = last_record_id(history)
    if rid is None:
        return question, None
    return f"{question} (appointment {rid})", rid


def recall_answer(question: str, history: list[BaseMessage]) -> str | None:
    """Answer 'what did I ask first/before?' directly from memory, or None if not a recall question."""
    if not RECALL_PATTERN.search(question):
        return None
    asked = [str(m.content) for m in history if isinstance(m, HumanMessage)]
    if not asked:
        return "I don't have any earlier questions from you in this conversation."
    wants_first = re.search(r"\bfirst\b", question, re.I)
    target = asked[0] if wants_first else asked[-1]
    return f"Your {'first' if wants_first else 'previous'} question in this conversation was: \"{target}\""


def with_memory(respond: Callable[[str, list[BaseMessage]], AIMessage]) -> RunnableWithMessageHistory:
    """Wrap a respond(question, history) -> AIMessage function with per-session history.

    RunnableWithMessageHistory loads the session's history before the call and appends the
    HumanMessage(question) and the returned AIMessage after it.
    """
    core = RunnableLambda(lambda inputs: respond(inputs["question"], list(inputs["history"])))
    return RunnableWithMessageHistory(
        core, get_session_history, input_messages_key="question", history_messages_key="history")
