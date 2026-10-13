"""Task 9: every crew response must conform to the CrewResponse Pydantic schema."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.crew import InvalidCrewOutput, run_crew, validate_crew_output  # noqa: E402
from app.schemas import CrewResponse  # noqa: E402

if __name__ == "__main__":
    print("CrewResponse JSON schema (wired as output_pydantic on the Composer task):")
    print(json.dumps(CrewResponse.model_json_schema()["properties"], indent=1)[:1400], "...\n")

    print("1) Real crew responses, validated in code with CrewResponse.model_validate_json():")
    for q in ["How long will my thyroid test results take?", "What is the status of APT-1038?",
              "What is the capital of France?"]:
        run = run_crew(q)
        print(f"   VALID   {q!r} -> intents={run.response.intents} grounded={run.response.grounded}")

    print("\n2) Malformed outputs are rejected before reaching a patient:")
    bad = {
        "missing answer": {"intents": ["policy"], "grounded": True},
        "unknown intent": {"answer": "x", "intents": ["diagnose_patient"], "grounded": True},
        "score out of range": {"answer": "x", "intents": ["appointment_lookup"], "grounded": True,
                               "appointment": {"record_id": "APT-1001", "category": "Cardiology",
                                               "status": "Scheduled", "consultation_fee_inr": 900,
                                               "escalation_score": 1.5, "escalate": True}},
        "not JSON at all": "Sure! Your appointment is fine.",
    }
    for name, payload in bad.items():
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        try:
            validate_crew_output(raw)
            print(f"   UNEXPECTEDLY VALID: {name}")
        except InvalidCrewOutput as e:
            print(f"   REJECTED {name:<20} -> {str(e).splitlines()[0]}")
