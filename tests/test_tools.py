"""Milestone 4 tests: check_appointment_status and the escalation score (Task 6)."""
import pytest

from app.tools import (
    ESCALATION_THRESHOLD, check_appointment_status, days_percentile, escalation_score,
    normalize_record_id,
)
from dataset import APPOINTMENTS


def test_score_formula_and_range():
    assert escalation_score(False, 0) == 0.0
    assert escalation_score(True, 30) == 1.0
    assert escalation_score(True, 15) == pytest.approx(0.6 + 0.4 * 0.5)
    for r in APPOINTMENTS:
        assert 0.0 <= escalation_score(r["follow_up_required"], r["days_since_created"]) <= 1.0


def test_score_is_not_a_bare_boolean_or():
    # Same follow-up flag, different age -> different scores (recency matters) ...
    assert escalation_score(True, 2) < escalation_score(True, 25)
    # ... and age alone (no follow-up) can never reach the threshold.
    assert escalation_score(False, 30) < ESCALATION_THRESHOLD
    # Follow-up alone (fresh record) does not escalate either.
    assert escalation_score(True, 0) < ESCALATION_THRESHOLD


def test_threshold_matches_dataset_percentile():
    assert ESCALATION_THRESHOLD == pytest.approx(0.6 + 0.4 * days_percentile(40) / 30, abs=0.005)


def test_lookup_found_and_fields():
    r = check_appointment_status("APT-1027")
    assert r["found"] and r["status"] == "Completed" and r["consultation_fee_inr"] == 850
    assert r["escalation_score"] == 1.0 and r["escalate"] is True


@pytest.mark.parametrize("raw", ["apt 1010", "APT1010", "status of APT-1010?"])
def test_record_id_normalisation(raw):
    assert normalize_record_id(raw) == "APT-1010"


def test_lookup_not_found_and_invalid():
    assert check_appointment_status("APT-9999")["found"] is False
    assert "Invalid" in check_appointment_status("hello")["error"]
