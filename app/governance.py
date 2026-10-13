"""AI governance (Task 15).

Application layer: least autonomy -- tool allow-list per agent, enforced when agents are built
                   (`app.crew.build_agent`) and re-audited on the assembled crew before kickoff.
Risk:              system risk classification with justification.
Runtime layer:     per-request token budget, checked BEFORE any model call (pre-flight estimate),
                   and audited AFTER the run against actual counted usage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app import config

# ---------------------------------------------------------------- risk classification

RISK_LEVEL = "High"
RISK_JUSTIFICATION = (
    "Under the Low/Medium/High scheme this system is HIGH risk, because it handles medical data: it "
    "reads patient appointment records (specialty, status, fees, follow-up need) and patients may type "
    "health conditions, symptoms or contact details into it. A wrong answer can delay care -- for "
    "example telling a patient with chest pain to book a video call, or misstating a cancellation fee -- "
    "and leaked data is sensitive health information under Indian data-protection law. It is not merely a "
    "Medium-risk customer-support bot because the answers concern medical services and records, not "
    "generic tickets. High risk is why the system adds PII masking before the model and logs, injection "
    "blocking, a calibrated groundedness refusal, a deterministic red-flag emergency notice, an "
    "independent second-team review of every draft, least-autonomy tool access and a hard cost budget, "
    "and why it never gives diagnoses or treatment advice."
)

# ---------------------------------------------------------------- token budget

# Calibrated from measured usage: a crew run alone used 2,601-4,074 prompt + 267-667 completion
# tokens; a full pipeline turn (crew + Autogen review) used 4,219-5,712 tokens
# (transcripts/task15_governance.txt). 6,000 base keeps the estimate an upper bound.
BASE_OVERHEAD_TOKENS = 6_000
# The question is re-sent in each of up to 7 model calls and echoed in tool output and answers.
PER_QUESTION_TOKEN_MULTIPLIER = 12


def estimate_tokens(text: str) -> int:
    """Approximate tokens as ceil(chars / 4): offline, no tokenizer download."""
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class BudgetCheck:
    question_tokens: int
    estimated_tokens: int
    budget: int

    @property
    def allowed(self) -> bool:
        return self.estimated_tokens <= self.budget

    def describe(self) -> str:
        return (f"estimated {self.estimated_tokens} tokens = {BASE_OVERHEAD_TOKENS} base + "
                f"{PER_QUESTION_TOKEN_MULTIPLIER} x {self.question_tokens} question tokens; budget {self.budget}")


class BudgetExceeded(Exception):
    def __init__(self, check: BudgetCheck):
        super().__init__(f"Request rejected: {check.describe()}")
        self.check = check


def check_budget(question: str, budget: int | None = None) -> BudgetCheck:
    q = estimate_tokens(question)
    return BudgetCheck(q, BASE_OVERHEAD_TOKENS + PER_QUESTION_TOKEN_MULTIPLIER * q,
                       config.MAX_TOKENS_PER_REQUEST if budget is None else budget)


def enforce_budget(question: str, budget: int | None = None) -> BudgetCheck:
    """Raise BudgetExceeded before any model call if the projected cost is over budget."""
    check = check_budget(question, budget)
    if not check.allowed:
        raise BudgetExceeded(check)
    return check
