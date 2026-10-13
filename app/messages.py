"""Fixed, policy-approved response templates (not factual claims; the review stage and judge accept them)."""

INJECTION_REFUSAL = ("I can't help with that request. I can answer questions about Practo clinic "
                     "policies or check an appointment if you share its ID (for example APT-1001).")
ASK_FOR_ID = "If you share your appointment ID (for example APT-1001), I can check that appointment for you."
BUDGET_REFUSAL = ("Your message is too long for me to process in one request. Please shorten it "
                  "(for example to one question) and try again.")
# Used when an appointment lookup answered but the policy part of the question was refused.
PARTIAL_POLICY_FALLBACK = "I couldn't find a clinic policy that answers the rest of your question."
