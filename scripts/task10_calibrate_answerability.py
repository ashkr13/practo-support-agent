"""Calibrate the v2 groundedness check: cross-encoder answerability threshold.

Uses ONLY already-seen questions (development queries + evaluation set A). Evaluation set B is
held out and not touched here. Red-flag emergency questions are excluded from calibration because
the emergency guardrail always leads with the 108 notice for them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402,F401  (offline switches before model libraries load)
from app.guardrails import _APPOINTMENT_REF, answerability, detect_red_flags  # noqa: E402
from app.rag import build_index, retrieve  # noqa: E402
from eval.eval_set import EVAL_SET  # noqa: E402
from eval.rag_queries import IN_SCOPE, IN_SCOPE_CALIBRATION_EXTRA, NEAR_DOMAIN_OOS, OUT_OF_SCOPE  # noqa: E402

IN = [q["query"] for q in IN_SCOPE] + IN_SCOPE_CALIBRATION_EXTRA + \
     ["Can I reschedule APT-1003, and what is the cancellation window?"] + \
     [e["query"] for e in EVAL_SET if e["expect"] in ("answer", "emergency")]
OUT = NEAR_DOMAIN_OOS + OUT_OF_SCOPE + [e["query"] for e in EVAL_SET if e["expect"] == "refuse"]

if __name__ == "__main__":
    build_index()
    print(f"Model: {config.ANSWERABILITY_MODEL}; score = max over top-{config.TOP_K} retrieved sentence chunks\n")
    groups = {}
    for label, qs in (("IN-SCOPE", IN), ("NEAR/OUT-OF-DOMAIN", OUT)):
        print(label)
        rows = []
        for q in qs:
            s = answerability(_APPOINTMENT_REF.sub(" ", q), [c.text for c in retrieve(q, config.DEPLOY_STRATEGY)])
            flag = bool(detect_red_flags(q))
            rows.append((s, q, flag))
        for s, q, flag in sorted(rows):
            print(f"  {s:8.3f}  {'[red flag: excluded] ' if flag else ''}{q}")
        groups[label] = rows
    lo_in = min(s for s, _, f in groups["IN-SCOPE"] if not f)
    hi_out = max(s for s, _, _ in groups["NEAR/OUT-OF-DOMAIN"])
    print(f"\nin-scope min (excl. red flags) = {lo_in:.3f}   near/out-of-domain max = {hi_out:.3f}   "
          f"gap = {lo_in - hi_out:.3f}")
    print(f"midpoint = {(lo_in + hi_out) / 2:.3f}  ->  ANSWERABILITY_THRESHOLD = {config.ANSWERABILITY_THRESHOLD}")
