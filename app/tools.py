"""Agent tools: RAG policy lookup (Tasks 3-5) and appointment status lookup (Task 6).

Each tool declares an explicit Pydantic argument schema. The mock LLM dispatches on these
schemas (never on tool names), and the governance layer allow-lists them per agent.
"""

from __future__ import annotations

import json
import re
import statistics

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app import config
from app.rag import grounded_answer
from dataset import APPOINTMENTS, APPOINTMENTS_BY_ID

# ---------------------------------------------------------------- escalation score (Task 6)

FOLLOW_UP_WEIGHT = 0.6
STALENESS_WEIGHT = 0.4
MAX_DAYS = 30

# Threshold justified from the generated data: the 40th percentile of days_since_created is
# 9.2 days, so 0.6 + 0.4 * 9.2 / 30 = 0.7227 -> 0.72. A follow-up-required record escalates once it
# is older than 40% of all appointments (and past the 7-day free follow-up window, kb/10).
ESCALATION_THRESHOLD = 0.72

RECORD_ID_PATTERN = re.compile(r"\bAPT[-\s]?(\d{4})\b", re.IGNORECASE)


def escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    """0.6 * follow_up_required + 0.4 * (days_since_created / 30), in [0, 1].

    Follow-up need dominates; staleness (age of the record) ranks urgency. Without a follow-up the
    score is capped at 0.4, so age alone never crosses the 0.72 threshold.
    """
    staleness = min(max(days_since_created, 0), MAX_DAYS) / MAX_DAYS
    return round(FOLLOW_UP_WEIGHT * float(follow_up_required) + STALENESS_WEIGHT * staleness, 4)


def days_percentile(p: int) -> float:
    """p-th percentile (1-99) of days_since_created in the generated dataset."""
    days = [r["days_since_created"] for r in APPOINTMENTS]
    return statistics.quantiles(days, n=100, method="inclusive")[p - 1]


def normalize_record_id(text: str) -> str | None:
    """Extract 'APT-1003' from inputs like 'apt 1003' or 'status of APT-1003?'."""
    m = RECORD_ID_PATTERN.search(text or "")
    return f"APT-{m.group(1)}" if m else None


def check_appointment_status(record_id: str) -> dict:
    """Look up one appointment and return its status, fee and designed escalation score."""
    rid = normalize_record_id(record_id)
    if rid is None:
        return {"kind": "appointment_status", "found": False, "record_id": record_id,
                "error": "Invalid appointment ID. Expected the format APT-1001."}
    rec = APPOINTMENTS_BY_ID.get(rid)
    if rec is None:
        return {"kind": "appointment_status", "found": False, "record_id": rid,
                "error": f"No appointment found with ID {rid}."}
    score = escalation_score(rec["follow_up_required"], rec["days_since_created"])
    return {
        "kind": "appointment_status",
        "found": True,
        "record_id": rid,
        "category": rec["category"],
        "status": rec["status"],
        "consultation_fee_inr": rec["consultation_fee_inr"],
        "days_since_created": rec["days_since_created"],
        "follow_up_required": rec["follow_up_required"],
        "escalation_score": score,
        "escalate": score > ESCALATION_THRESHOLD,
        "escalation_threshold": ESCALATION_THRESHOLD,
    }


# ---------------------------------------------------------------- RAG tool payload

def rag_lookup(query: str) -> dict:
    """Grounded policy answer from the deployed (sentence) collection, with its evidence."""
    r = grounded_answer(query, config.DEPLOY_STRATEGY)
    return {
        "kind": "policy_lookup",
        "query": query,
        "answer": r.answer,
        "grounded": r.grounded,
        "top1_similarity": r.top1_similarity,
        "threshold": r.threshold,
        "sources": r.sources,
        "context": [{"doc_id": c.doc_id, "text": c.text, "similarity": c.similarity}
                    for c in r.context],
        "cache_hit": r.cache_hit,
    }


# ---------------------------------------------------------------- CrewAI tool wrappers

# Audit trail of real tool executions (evidence that kickoff() actually invoked each tool).
TOOL_CALLS: list[dict] = []

class RagLookupInput(BaseModel):
    query: str = Field(..., description="The patient's clinic-policy question.")


class AppointmentLookupInput(BaseModel):
    record_id: str = Field(..., description="Appointment ID in the format APT-1001.")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = ("Search Practo's clinic policy knowledge base and return a grounded answer "
                        "with its sources and retrieved context.")
    args_schema: type[BaseModel] = RagLookupInput

    def _run(self, query: str) -> str:
        result = rag_lookup(query)
        TOOL_CALLS.append({"tool": self.name, "args": {"query": query}, "result": result})
        return json.dumps(result, ensure_ascii=False)


class AppointmentStatusTool(BaseTool):
    name: str = "check_appointment_status"
    description: str = ("Look up one appointment by ID and return its status, consultation fee and "
                        "escalation score.")
    args_schema: type[BaseModel] = AppointmentLookupInput

    def _run(self, record_id: str) -> str:
        result = check_appointment_status(record_id)
        TOOL_CALLS.append({"tool": self.name, "args": {"record_id": record_id}, "result": result})
        return json.dumps(result, ensure_ascii=False)
