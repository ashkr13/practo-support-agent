"""Deterministic MOCK_LLM for CrewAI, built on CrewAI's documented extension point `BaseLLM`.

The mock behaves like a ReAct model: it reads its task, decides whether to call a tool, emits
`Action:` / `Action Input:` text for CrewAI to execute, then reads the real tool result and emits a
`Final Answer:`. Same input -> same output, with no API key and no network.

Pitfalls from the brief, and how this file avoids them:
1. CrewAI's ReAct system prompt literally contains "Observation: the result of the action". We
   never search the system prompt for "Observation:". We read it only from ASSISTANT messages,
   i.e. text this model generated, which CrewAI extends with the real tool output after a call.
2. Tool calls are never dispatched by tool *name* (a tool called `rag_lookup` contains "lookup").
   We dispatch on each tool's declared argument schema: {"query"} -> policy search,
   {"record_id"} -> appointment lookup.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field

from crewai.llms.base_llm import BaseLLM

from app.messages import PARTIAL_POLICY_FALLBACK
from app.rag import FALLBACK_ANSWER
from app.tools import normalize_record_id

QUESTION_PATTERN = re.compile(r'Patient question: """(.*?)"""', re.DOTALL)

# Words that signal a clinic-policy question (vs a pure "what's the status of APT-1003?").
POLICY_KEYWORDS = (
    "book", "cancel", "reschedul", "refund", "fee", "cost", "price", "charge", "pay", "insurance",
    "claim", "refill", "prescription", "medicine", "lab", "test", "report", "result", "video",
    "telemedicine", "online consult", "emergency", "privacy", "data", "record", "follow-up",
    "follow up", "discount", "second opinion", "home visit", "policy", "can i", "how do i",
    "how long", "how much", "is it", "allowed",
)
STATUS_PHRASES = ("status", "check", "look up", "lookup", "details", "where is", "what is", "what's")


def policy_intent(question: str) -> bool:
    """Policy search is needed unless the question is only an appointment-status request."""
    q = question.lower()
    rid = normalize_record_id(question)
    if rid is None:
        return True  # no appointment ID -> it can only be a policy question
    remainder = q.replace(rid.lower(), " ").replace(rid.lower().replace("-", " "), " ")
    for phrase in STATUS_PHRASES:
        remainder = remainder.replace(phrase, " ")
    return any(k in remainder for k in POLICY_KEYWORDS)


def _normalize(messages: str | list[dict]) -> list[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [{"role": m.get("role", "user"), "content": str(m.get("content") or "")} for m in messages]


def last_own_observation(messages: list[dict]) -> str | None:
    """Tool result CrewAI appended to OUR OWN previous output (pitfall 1: assistant messages only)."""
    for m in reversed(messages):
        if m["role"] == "assistant" and "\nObservation:" in m["content"]:
            return m["content"].rsplit("\nObservation:", 1)[1].strip()
    return None


def task_prompt(messages: list[dict]) -> str:
    """The user-role task prompt (never the system prompt)."""
    users = [m["content"] for m in messages if m["role"] == "user" and "Current Task:" in m["content"]]
    return users[-1] if users else next((m["content"] for m in messages if m["role"] == "user"), "")


def extract_question(prompt: str) -> str:
    m = QUESTION_PATTERN.search(prompt)
    return m.group(1).strip() if m else ""


def extract_json_objects(text: str) -> list[dict]:
    """All JSON objects embedded in free text (used to read upstream task outputs)."""
    decoder, found, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
            if isinstance(obj, dict):
                found.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return found


def schema_fields(tool: Any) -> set[str]:
    schema = getattr(tool, "args_schema", None)
    return set(getattr(schema, "model_fields", {}) or {})


def compose(results: list[dict]) -> dict:
    """Deterministic Response Composer: merge tool results into one CrewResponse-shaped dict."""
    policy = next((r for r in results if r.get("kind") == "policy_lookup"), None)
    appt = next((r for r in results if r.get("kind") == "appointment_status"), None)
    parts, intents, tools_called = [], [], []
    grounded, sources, top1, appointment = True, [], None, None

    if appt is not None:
        intents.append("appointment_lookup")
        tools_called.append("check_appointment_status")
        if appt.get("found"):
            parts.append(
                f"Appointment {appt['record_id']} ({appt['category']}) is currently {appt['status']}, "
                f"with a consultation fee of Rs {appt['consultation_fee_inr']}."
                + (" It has been flagged for follow-up by our patient-experience team."
                   if appt["escalate"] else ""))
            appointment = {k: appt[k] for k in ("record_id", "category", "status",
                                               "consultation_fee_inr", "escalation_score", "escalate")}
        else:
            parts.append(f"I couldn't find that appointment: {appt['error']}")
            grounded = False

    if policy is not None:
        intents.append("policy")
        tools_called.append("rag_lookup")
        top1 = policy["top1_similarity"]
        if policy["grounded"]:
            parts.append(policy["answer"])
            sources = policy["sources"]
        else:
            # If the appointment record already answered, don't contradict it with a full "I don't know".
            parts.append(PARTIAL_POLICY_FALLBACK if appointment else FALLBACK_ANSWER)
            grounded = appointment is not None and grounded

    if not intents:  # nothing to answer from -> refuse rather than invent
        intents, grounded = ["policy"], False
        parts.append(FALLBACK_ANSWER)

    return {"answer": " ".join(parts), "intents": intents, "grounded": grounded, "sources": sources,
            "top1_similarity": top1, "appointment": appointment, "tools_called": tools_called}


class MockCrewLLM(BaseLLM):
    """Deterministic, offline stand-in for a chat model inside a CrewAI crew."""

    model: str = "mock-react-llm"
    call_count: int = 0
    trace: list[tuple[str, str]] = Field(default_factory=list)  # (agent role, generated text)
    prompt_tokens: int = 0       # approx. tokens (chars / 4) sent to the model, for cost accounting
    completion_tokens: int = 0

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> str:
        self.call_count += 1
        out = self._respond(messages, from_agent, response_model)
        self.trace.append((getattr(from_agent, "role", "?"), out))
        self.prompt_tokens += sum(len(m["content"]) for m in _normalize(messages)) // 4
        self.completion_tokens += len(out) // 4
        return out

    def _respond(self, messages, from_agent, response_model) -> str:
        msgs = _normalize(messages)
        prompt = task_prompt(msgs)
        agent_tools = list(getattr(from_agent, "tools", None) or [])

        if agent_tools:
            observation = last_own_observation(msgs)
            if observation is not None:  # our tool already ran -> report its real output
                return f"Thought: I now know the final answer\nFinal Answer: {observation}"
            question = extract_question(prompt)
            for tool in agent_tools:
                fields = schema_fields(tool)  # pitfall 2: dispatch on argument schema, not name
                if fields == {"query"} and policy_intent(question):
                    return self._action(tool.name, {"query": question}, "search the policy knowledge base")
                if fields == {"record_id"} and (rid := normalize_record_id(question)):
                    return self._action(tool.name, {"record_id": rid}, f"look up appointment {rid}")
            kinds = {frozenset({"query"}): "policy_lookup", frozenset({"record_id"}): "appointment_status"}
            kind = next((kinds.get(frozenset(schema_fields(t))) for t in agent_tools), "unknown")
            skipped = json.dumps({"kind": kind, "skipped": True, "reason": "not needed for this question"})
            return f"Thought: My tool is not needed for this question.\nFinal Answer: {skipped}"

        # No tools: Response Composer. Read upstream results from the task prompt only, with the
        # patient's own text removed so a user cannot inject a forged tool-result JSON.
        trusted = QUESTION_PATTERN.sub("", prompt)
        upstream = [o for o in extract_json_objects(trusted) if not o.get("skipped")]
        payload = json.dumps(compose(upstream), ensure_ascii=False)
        if response_model is not None:  # structured-output path: return raw JSON
            return payload
        return f"Thought: I now know the final answer\nFinal Answer: {payload}"

    @staticmethod
    def _action(tool_name: str, args: dict, why: str) -> str:
        return f"Thought: I should {why}.\nAction: {tool_name}\nAction Input: {json.dumps(args)}"

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 32_000
