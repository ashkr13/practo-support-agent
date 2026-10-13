"""Milestone 1 tests: dataset thresholds (Task 1) and knowledge-base coverage (Task 2)."""

from dataset import (
    APPOINTMENTS,
    CATEGORY_WEIGHTS,
    STATUS_WEIGHTS,
    generate_appointments,
    summarize,
    validate,
)
from app.kb import load_documents, split_sentences

REQUIRED_FIELDS = {
    "record_id", "category", "status", "consultation_fee_inr",
    "days_since_created", "follow_up_required",
}


def test_dataset_passes_all_thresholds():
    assert validate(APPOINTMENTS) == []


def test_dataset_is_deterministic():
    assert generate_appointments() == generate_appointments()


def test_dataset_structure():
    s = summarize(APPOINTMENTS)
    assert s["total"] >= 40
    assert all(s["per_category"][c] >= 3 for c in CATEGORY_WEIGHTS)
    assert all(s["per_status"][st] >= 1 for st in STATUS_WEIGHTS)
    assert 10 <= s["follow_up_pct"] <= 30
    for r in APPOINTMENTS:
        assert set(r) == REQUIRED_FIELDS
        assert isinstance(r["days_since_created"], int) and 0 <= r["days_since_created"] <= 30
        assert isinstance(r["follow_up_required"], bool)
        assert 300 <= r["consultation_fee_inr"] <= 2000
    assert len({r["record_id"] for r in APPOINTMENTS}) == len(APPOINTMENTS)


def test_validate_detects_violation():
    bad = [dict(r, follow_up_required=True) for r in APPOINTMENTS]  # 100% follow-up
    assert any("follow_up_required" in p for p in validate(bad))


REQUIRED_TOPICS = [
    "booking", "cancellation", "fee", "insurance", "refill", "lab test",
    "telemedicine", "emergency", "privacy", "follow-up", "second opinion", "home visit",
]


def test_kb_has_12_docs_each_2_to_5_sentences():
    docs = load_documents()
    assert len(docs) >= 12
    for d in docs:
        assert 2 <= len(split_sentences(d.text)) <= 5, d.doc_id


def test_kb_covers_every_required_topic():
    titles = " ".join(d.title.lower() for d in load_documents())
    missing = [t for t in REQUIRED_TOPICS if t not in titles]
    assert not missing, missing
