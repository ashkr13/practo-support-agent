"""Query sets for Tasks 4-5. Relevant doc_ids are hand-labelled ground truth."""

# In-scope queries used for grounded-generation demo (Task 4) AND precision/recall (Task 5).
IN_SCOPE: list[dict] = [
    {"query": "How many hours before my appointment can I cancel without paying anything?",
     "relevant": {"02_cancellation_rescheduling"}},
    {"query": "How much does it cost to see a cardiologist?",
     "relevant": {"03_consultation_fees"}},
    {"query": "How long will my thyroid test results take?",
     "relevant": {"06_lab_test_turnaround"}},
    {"query": "Can I get my antibiotics refilled without seeing the doctor again?",
     "relevant": {"05_prescription_refills"}},
    {"query": "Is a follow-up visit free if I go back to the same doctor within a week?",
     "relevant": {"10_follow_up_discount"}},
    {"query": "I have chest pain, can I just book a video consultation?",
     "relevant": {"07_telemedicine_eligibility", "08_emergency_visits"}},
]

# Extra in-scope queries used only to widen the calibration sample.
IN_SCOPE_CALIBRATION_EXTRA: list[str] = [
    "Can a doctor come to my home if I am 70 years old?",
    "How do I file an insurance claim for my consultation?",
    "Will Practo share my medical records with anyone?",
    "How do I get a second opinion from another specialist?",
]

# Deliberately out-of-scope queries (nothing in the KB answers these).
OUT_OF_SCOPE: list[str] = [
    "What is the capital of France?",
    "Who won the cricket world cup in 2011?",
    "How do I bake sourdough bread at home?",
    "Can you recommend a good laptop for programming?",
    "What will the weather be like in Mumbai tomorrow?",
]

# Near-domain out-of-scope probes: health/clinic-sounding questions the KB does NOT answer.
# Reported as a stress test only; NOT used to pick the threshold (see README, Task 4).
NEAR_DOMAIN_OOS: list[str] = [
    "Do you offer dental implants?",
    "What are the visiting hours for admitted patients?",
    "Can I pay my consultation fee with cryptocurrency?",
    "What is the parking fee at the clinic?",
    "Which vaccines does my baby need at 6 months?",
]
