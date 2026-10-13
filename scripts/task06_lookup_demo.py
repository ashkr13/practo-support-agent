"""Task 6: check_appointment_status + designed escalation score, justified from the dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools import (  # noqa: E402
    ESCALATION_THRESHOLD, FOLLOW_UP_WEIGHT, STALENESS_WEIGHT, check_appointment_status,
    days_percentile, escalation_score,
)
from dataset import APPOINTMENTS  # noqa: E402

if __name__ == "__main__":
    print(f"escalation_score = {FOLLOW_UP_WEIGHT} * follow_up_required + "
          f"{STALENESS_WEIGHT} * (days_since_created / 30)      range [0, 1]")
    print("\ndays_since_created distribution in the generated dataset:")
    for p in (20, 40, 50, 60, 80):
        print(f"  P{p} = {days_percentile(p):.1f} days")
    p40 = days_percentile(40)
    print(f"\nThreshold = {FOLLOW_UP_WEIGHT} + {STALENESS_WEIGHT} * (P40 {p40:.1f} / 30) = "
          f"{FOLLOW_UP_WEIGHT + STALENESS_WEIGHT * p40 / 30:.4f} -> {ESCALATION_THRESHOLD}")
    print("  => escalate only follow-up-required records older than the 40th percentile of record age.")
    print(f"  => without a follow-up the score is capped at {STALENESS_WEIGHT}, so age alone never escalates.")

    scored = sorted(((escalation_score(r["follow_up_required"], r["days_since_created"]), r)
                     for r in APPOINTMENTS), key=lambda x: -x[0])
    flagged = [r for s, r in scored if s > ESCALATION_THRESHOLD]
    print(f"\nRecords escalated: {len(flagged)}/{len(APPOINTMENTS)} "
          f"= {100 * len(flagged) / len(APPOINTMENTS):.0f}%")
    print("Top 8 scores:")
    for s, r in scored[:8]:
        print(f"  {r['record_id']}  score={s:.4f}  follow_up={r['follow_up_required']!s:<5} "
              f"days={r['days_since_created']:>2}  escalate={s > ESCALATION_THRESHOLD}")

    print("\nSample lookups:")
    for rid in [flagged[0]["record_id"], "APT-1002", "APT-1001", "apt 1010", "APT-9999", "hello"]:
        print(f"  check_appointment_status({rid!r}) ->\n    {check_appointment_status(rid)}")
