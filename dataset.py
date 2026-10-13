"""Seeded, deterministic generator for the fabricated Practo appointment dataset (Task 1).

Run `python dataset.py` to print the validation report.

Design choices (also stated in README.md):
- SEED = 7 and N_RECORDS = 50; a private random.Random(SEED) instance makes output
  identical on every run and independent of any other code using `random`.
- Category and status values are drawn with the weights below.
- consultation_fee_inr is drawn per specialty from a band inside Rs 300-2,000, rounded to the
  nearest Rs 50: a general physician consult in an Indian city typically costs a few hundred
  rupees while a senior cardiologist can charge close to two thousand, so specialty-specific
  bands keep fees realistic.
- days_since_created is uniform on 0..30; follow_up_required is True with probability 0.20.
All records are fabricated; nothing is real patient data. Never hand-edit records -- if a
threshold fails, change SEED or the weights and regenerate.
"""

from __future__ import annotations

import random
from collections import Counter

SEED = 7
N_RECORDS = 50

CATEGORY_WEIGHTS: dict[str, float] = {
    "General Medicine": 0.30,
    "Cardiology": 0.15,
    "Dermatology": 0.20,
    "Pediatrics": 0.20,
    "Orthopedics": 0.15,
}

STATUS_WEIGHTS: dict[str, float] = {
    "Scheduled": 0.30,
    "Completed": 0.35,
    "Cancelled": 0.12,
    "No-Show": 0.08,
    "Rescheduled": 0.15,
}

# (min, max) consultation fee in INR per specialty, all within the overall Rs 300-2,000 range.
FEE_BANDS_INR: dict[str, tuple[int, int]] = {
    "General Medicine": (300, 700),
    "Pediatrics": (400, 900),
    "Dermatology": (500, 1200),
    "Orthopedics": (600, 1500),
    "Cardiology": (800, 2000),
}

FOLLOW_UP_PROBABILITY = 0.20
MAX_DAYS_SINCE_CREATED = 30

# Thresholds from the brief, checked by validate().
MIN_RECORDS = 40
MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
FOLLOW_UP_PCT_BAND = (10.0, 30.0)


def _fee(rng: random.Random, category: str) -> int:
    low, high = FEE_BANDS_INR[category]
    return int(round(rng.randint(low, high) / 50) * 50)


def generate_appointments(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    """Return n fabricated appointment records, deterministic for a given seed."""
    rng = random.Random(seed)
    categories, cat_w = zip(*CATEGORY_WEIGHTS.items())
    statuses, stat_w = zip(*STATUS_WEIGHTS.items())
    records = []
    for i in range(n):
        category = rng.choices(categories, weights=cat_w, k=1)[0]
        records.append(
            {
                "record_id": f"APT-{1001 + i}",
                "category": category,
                "status": rng.choices(statuses, weights=stat_w, k=1)[0],
                "consultation_fee_inr": _fee(rng, category),
                "days_since_created": rng.randint(0, MAX_DAYS_SINCE_CREATED),
                "follow_up_required": rng.random() < FOLLOW_UP_PROBABILITY,
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    """Counts per category/status and the follow-up percentage."""
    follow_ups = sum(r["follow_up_required"] for r in records)
    return {
        "total": len(records),
        "per_category": Counter(r["category"] for r in records),
        "per_status": Counter(r["status"] for r in records),
        "follow_up_count": follow_ups,
        "follow_up_pct": 100.0 * follow_ups / len(records),
    }


def validate(records: list[dict]) -> list[str]:
    """Return a list of threshold violations (empty list = dataset is valid)."""
    s = summarize(records)
    problems = []
    if s["total"] < MIN_RECORDS:
        problems.append(f"only {s['total']} records (< {MIN_RECORDS})")
    for cat in CATEGORY_WEIGHTS:
        if s["per_category"][cat] < MIN_PER_CATEGORY:
            problems.append(f"category {cat!r} has {s['per_category'][cat]} (< {MIN_PER_CATEGORY})")
    for status in STATUS_WEIGHTS:
        if s["per_status"][status] < MIN_PER_STATUS:
            problems.append(f"status {status!r} has {s['per_status'][status]} (< {MIN_PER_STATUS})")
    low, high = FOLLOW_UP_PCT_BAND
    if not low <= s["follow_up_pct"] <= high:
        problems.append(f"follow_up_required = {s['follow_up_pct']:.1f}% (outside {low}-{high}%)")
    for r in records:
        if not 0 <= r["days_since_created"] <= MAX_DAYS_SINCE_CREATED:
            problems.append(f"{r['record_id']} days_since_created out of range")
    return problems


APPOINTMENTS: list[dict] = generate_appointments()
APPOINTMENTS_BY_ID: dict[str, dict] = {r["record_id"]: r for r in APPOINTMENTS}


def report(records: list[dict] = APPOINTMENTS) -> str:
    s = summarize(records)
    fees = [r["consultation_fee_inr"] for r in records]
    lines = [
        f"Practo appointment dataset  (seed={SEED}, n={s['total']})",
        "",
        "Count per category:",
        *(f"  {c:<18} {s['per_category'][c]:>3}" for c in CATEGORY_WEIGHTS),
        "",
        "Count per status:",
        *(f"  {st:<18} {s['per_status'][st]:>3}" for st in STATUS_WEIGHTS),
        "",
        f"follow_up_required=True: {s['follow_up_count']}/{s['total']} = {s['follow_up_pct']:.1f}%",
        f"consultation_fee_inr: min={min(fees)} max={max(fees)}",
        "",
    ]
    problems = validate(records)
    lines.append("VALIDATION: PASS" if not problems else "VALIDATION: FAIL\n  " + "\n  ".join(problems))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
    print("\nFirst 5 records:")
    for r in APPOINTMENTS[:5]:
        print(" ", r)
