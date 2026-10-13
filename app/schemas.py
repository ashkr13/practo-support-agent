"""Pydantic data contracts shared by the crew, review stage and API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["policy", "appointment_lookup", "conversation", "refused"]


class AppointmentInfo(BaseModel):
    record_id: str
    category: str
    status: str
    consultation_fee_inr: int = Field(..., ge=0)
    escalation_score: float = Field(..., ge=0.0, le=1.0)
    escalate: bool


class CrewResponse(BaseModel):
    """Structured output every crew response must conform to (Task 9)."""

    answer: str = Field(..., min_length=1, description="Patient-facing answer text.")
    intents: list[Intent] = Field(..., min_length=1, description="What the crew handled.")
    grounded: bool = Field(..., description="True if every fact came from the KB or the dataset.")
    sources: list[str] = Field(default_factory=list, description="KB doc_ids quoted in the answer.")
    top1_similarity: float | None = Field(None, ge=-1.0, le=1.0)
    appointment: AppointmentInfo | None = None
    tools_called: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """What the full pipeline (guardrails + memory + crew) returns for one patient turn."""

    session_id: str
    question_masked: str = Field(..., description="Patient text after PII masking (what the model saw).")
    resolved_question: str = Field(..., description="Question after memory resolved references.")
    response: CrewResponse
    blocked: bool = False
    guardrail_events: list[str] = Field(default_factory=list)
    remembered_record_id: str | None = None
    llm_calls: int = 0
    review: "ReviewSummary | None" = None
    estimated_tokens: int = 0
    actual_tokens: int = 0
    cache_hit: bool = False


class ReviewSummary(BaseModel):
    """Autogen review verdict (Task 14) as exposed on the API."""

    approved: bool
    final_answer: str
    reason: str


AgentResponse.model_rebuild()
