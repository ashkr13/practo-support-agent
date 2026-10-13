"""Guardrails (Task 10).

Input side : PII masking (fixed-format Indian mobile numbers), prompt-injection detection,
             red-flag emergency-symptom detection.
Output side: groundedness check -- refuse when the retrieved context does not support the question.

All checks are deterministic rules, so they work identically under MOCK_LLM and can be unit-tested.
Out of scope for masking (no universal format): patient name, diagnosis, insurance ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# ---------------------------------------------------------------- PII masking

PHONE_MASK = "[PHONE_REDACTED]"

# Indian mobile: optional +91 / 91 / 0 prefix, then 10 digits starting 6-9, allowing one
# space or hyphen inside (e.g. +91 98765 43210, 098765-43210, 9876543210).
PHONE_PATTERN = re.compile(
    r"(?<![\w+])(?:\+91[\s-]?|91[\s-]|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
)


def mask_pii(text: str) -> tuple[str, int]:
    """Replace every fixed-format phone number with PHONE_MASK. Returns (masked_text, count)."""
    return PHONE_PATTERN.subn(PHONE_MASK, text)


# ---------------------------------------------------------------- prompt injection

INJECTION_PATTERNS: dict[str, re.Pattern] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b[^.?!]{0,40}\b(instruction|rule|prompt|guideline|polic)",
        re.I),
    "reveal_system_prompt": re.compile(
        r"\b(reveal|show|print|repeat|leak|tell me)\b[^.?!]{0,30}\b(system prompt|instructions|hidden prompt)", re.I),
    "role_hijack": re.compile(
        r"\b(you are now|from now on you are|pretend (to be|you are)|act as (a|an|the)? ?(admin|developer|system))",
        re.I),
    "jailbreak_keyword": re.compile(r"\b(jailbreak|developer mode|DAN mode|do anything now)\b", re.I),
    "fake_system_tag": re.compile(r"(<\s*/?\s*system\s*>|\[\s*system\s*\]|^\s*system\s*:)", re.I | re.M),
}


def detect_injection(text: str) -> list[str]:
    """Names of the injection patterns found in text (empty list = clean)."""
    return [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text)]


# ---------------------------------------------------------------- red-flag emergency symptoms

RED_FLAGS: dict[str, re.Pattern] = {
    "chest pain": re.compile(r"\bchest (pain|tightness|pressure)\b", re.I),
    "breathing difficulty": re.compile(
        r"\b(difficulty breathing|breathing difficult\w*|shortness of breath|can'?t breathe|cannot breathe)\b",
        re.I),
    "severe bleeding": re.compile(r"\b(severe|heavy|uncontrolled) bleeding\b|\bbleeding (heavily|a lot)\b", re.I),
    "loss of consciousness": re.compile(r"\b(unconscious|loss of consciousness|passed out|fainted)\b", re.I),
    "stroke symptoms": re.compile(r"\b(stroke|face drooping|slurred speech)\b", re.I),
}

# Verbatim from kb/08_emergency_visits.md, so the emergency notice is itself grounded in the KB.
EMERGENCY_NOTICE = (
    "Patients with chest pain, severe bleeding, difficulty breathing, loss of consciousness, or stroke "
    "symptoms should call the national ambulance number 108 or go to the nearest hospital emergency "
    "department immediately."
)
EMERGENCY_DOC_ID = "08_emergency_visits"


def detect_red_flags(text: str) -> list[str]:
    return [name for name, pattern in RED_FLAGS.items() if pattern.search(text)]


# ---------------------------------------------------------------- output-side groundedness

STOPWORDS = set("""
a an the and or but if then so of to in on at by for with from about as into than too very is are was
were be been being am do does did done have has had having can could will would shall should may might
must i me my mine we our you your he she it its they them their this that these those there here what
which who whom whose when where why how not no yes any all some each every more most other such only
own same just also please tell know get got let like want need much many long take takes taking still
really thing things okay ok hi hello thanks thank way also again old new one two three anything
something someone anyone us
see seeing saw come coming came go going went back make making give giving use using find look
try help able without book booking
""".split())

# Words with no policy-specific meaning in this domain; they never decide groundedness.
GENERIC_DOMAIN_TERMS = {"practo", "result", "results", "appointment", "appointments", "doctor", "doctors",
                        "patient", "patients", "clinic", "clinics", "visit", "visits", "year", "years",
                        "day", "days", "week", "weeks", "hour", "hours", "time", "times"}

_WORD = re.compile(r"[a-z]+")


IRREGULAR_FORMS = {"paid": "pay", "saw": "see", "seen": "see", "went": "go", "gone": "go",
                   "gave": "give", "given": "give", "took": "take", "taken": "take", "sent": "send",
                   "children": "child", "kids": "child", "kid": "child"}


def normalize_word(word: str) -> str:
    """Tiny stemmer: irregular forms, then strip -ing / -ed / -es / -s (keeping >= 3 letters)."""
    word = IRREGULAR_FORMS.get(word, word)
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _stem_match(term: str, vocabulary: set[str]) -> bool:
    """Light morphology: 'paying'~'paid', 'refilled'~'refill', 'cardiologist'~'cardiology'."""
    if term in vocabulary:
        return True
    stem = normalize_word(term)
    if any(normalize_word(w) == stem for w in vocabulary):
        return True
    prefix = term[:5]
    return len(term) >= 5 and any(len(w) >= 5 and w[:5] == prefix for w in vocabulary)


def content_terms(text: str) -> list[str]:
    text = re.sub(r"\bAPT[-\s]?\d{4}\b", " ", text, flags=re.I).replace(PHONE_MASK, " ")
    terms = [w for w in _WORD.findall(text.lower()) if len(w) >= 3]
    return list(dict.fromkeys(t for t in terms if t not in STOPWORDS and t not in GENERIC_DOMAIN_TERMS))


@dataclass
class GroundednessVerdict:
    supported: bool
    reason: str
    uncovered_terms: list[str] = field(default_factory=list)
    answerability: float | None = None


_KB_VOCAB: set[str] | None = None


def kb_vocabulary() -> set[str]:
    """Every word in the indexed knowledge base (texts + titles), cached; refreshed on add-document."""
    global _KB_VOCAB
    if _KB_VOCAB is None:
        from app import config
        from app.rag import get_collection

        data = get_collection(config.DEPLOY_STRATEGY).get(include=["documents", "metadatas"])
        texts = data["documents"] + [m["title"] for m in data["metadatas"]]
        _KB_VOCAB = {w for t in texts for w in _WORD.findall(t.lower())}
    return _KB_VOCAB


def invalidate_kb_vocabulary() -> None:
    global _KB_VOCAB
    _KB_VOCAB = None


def out_of_domain_terms(question: str, vocabulary: set[str] | None = None) -> list[str]:
    vocabulary = kb_vocabulary() if vocabulary is None else vocabulary
    return [t for t in content_terms(question) if not _stem_match(t, vocabulary)]


def check_groundedness_v1(question: str, policy_result: dict) -> GroundednessVerdict:
    """SUPERSEDED v1 rule, kept only for the documented comparison
    (transcripts/task10_groundedness_design.txt, transcripts/task13_evaluation_v1_setA.txt).

    Refused unless similarity passed AND the question had no word absent from the KB vocabulary.
    It overfitted: 11/11 dev questions answered, but on held-out eval set A it wrongly refused 5/12
    in-scope questions over ordinary words ("far", "money", "extra", "four months ago").
    """
    if not policy_result.get("grounded"):
        return GroundednessVerdict(
            False, f"top-1 similarity {policy_result.get('top1_similarity')} below threshold "
                   f"{policy_result.get('threshold')}")
    unknown = out_of_domain_terms(question)
    if unknown:
        return GroundednessVerdict(False, f"question mentions concepts absent from the knowledge base: {unknown}",
                                   unknown)
    return GroundednessVerdict(True, "similarity above threshold and no out-of-domain concepts")


_APPOINTMENT_REF = re.compile(r"\(appointment APT-\d{4}\)|\bAPT[-\s]?\d{4}\b,?", re.I)


@lru_cache(maxsize=1)
def get_answerability_model():
    from sentence_transformers import CrossEncoder

    from app import config
    return CrossEncoder(config.ANSWERABILITY_MODEL)


def answerability(question: str, passages: list[str]) -> float:
    """Best cross-encoder score over the passages: reads question and passage TOGETHER and scores whether
    the passage answers the question (unlike the bi-encoder, which embeds them separately)."""
    if not passages:
        return float("-inf")
    return float(max(get_answerability_model().predict([(question, p) for p in passages])))


def check_groundedness(question: str, policy_result: dict) -> GroundednessVerdict:
    """Output-side groundedness check (v2). Refuse unless the retrieved context supports the question:
    (1) bi-encoder top-1 similarity passed the calibrated threshold (Task 4), AND
    (2) cross-encoder answerability of the retrieved passages >= ANSWERABILITY_THRESHOLD.

    (2) catches near-domain questions such as "What is the parking fee at the clinic?" (similarity 0.656
    overlaps the in-scope range because "fee" and "clinic" match the fee policy; answerability -1.48).
    """
    from app import config

    if not policy_result.get("grounded"):
        return GroundednessVerdict(
            False, f"top-1 similarity {policy_result.get('top1_similarity')} below threshold "
                   f"{policy_result.get('threshold')}")
    # Judge only the POLICY part of the question: appointment IDs are the Lookup Agent's job and
    # confuse the cross-encoder ("Can I reschedule APT-1003, and what is the cancellation window?").
    policy_question = " ".join(_APPOINTMENT_REF.sub(" ", question).split())
    score = answerability(policy_question, [c["text"] for c in policy_result.get("context", [])])
    if score < config.ANSWERABILITY_THRESHOLD:
        return GroundednessVerdict(
            False, f"retrieved context does not answer the question (cross-encoder answerability "
                   f"{score:.2f} < {config.ANSWERABILITY_THRESHOLD})", answerability=round(score, 3))
    return GroundednessVerdict(True, f"similarity above threshold and answerability {score:.2f} >= "
                                     f"{config.ANSWERABILITY_THRESHOLD}", answerability=round(score, 3))
