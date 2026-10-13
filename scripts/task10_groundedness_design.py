"""Task 10 design evidence: compare two groundedness rules on in-scope vs near-domain queries."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402,F401
from app.guardrails import _stem_match, check_groundedness, check_groundedness_v1, content_terms  # noqa: E402
from app.tools import rag_lookup  # noqa: E402
from eval.rag_queries import IN_SCOPE, IN_SCOPE_CALIBRATION_EXTRA, NEAR_DOMAIN_OOS  # noqa: E402

IN = [q["query"] for q in IN_SCOPE] + IN_SCOPE_CALIBRATION_EXTRA + [
    "Can I reschedule APT-1003, and what is the cancellation window?"]


def strict_rule(q: str, res: dict) -> bool:
    """Rejected variant: every content term must appear in the top-3 retrieved chunks."""
    vocab = {w for c in res["context"] for w in re.findall(r"[a-z]+", c["text"].lower())}
    return res["grounded"] and all(_stem_match(t, vocab) for t in content_terms(q))


if __name__ == "__main__":
    print("Development questions only (eval set B is held out).")
    print("A  = strict: every question term in retrieved top-3 context            (rejected in Task 10)")
    print("B  = v1: similarity AND no word absent from the KB vocabulary           (overfitted: 5/12 wrong on set A)")
    print("C  = v2 FINAL: similarity AND cross-encoder answerability >= threshold\n")
    tally = {}
    for label, qs in (("IN-SCOPE (should answer)", IN), ("NEAR-DOMAIN OOS (should refuse)", NEAR_DOMAIN_OOS)):
        print(label)
        for q in qs:
            res = rag_lookup(q)
            a, b, c = strict_rule(q, res), check_groundedness_v1(q, res), check_groundedness(q, res)
            for rule, ok in (("A", a), ("B", b.supported), ("C", c.supported)):
                tally.setdefault((label, rule), []).append(ok)
            word = lambda ok: "answer" if ok else "refuse"  # noqa: E731
            print(f"  A={word(a)}  B={word(b.supported)}  C={word(c.supported)}"
                  f"  sim={res['top1_similarity']:.3f}  {q}\n      C reason: {c.reason}")
    print("\nSUMMARY")
    for rule in ("A", "B", "C"):
        ok_in = sum(tally[("IN-SCOPE (should answer)", rule)])
        ok_oos = sum(not x for x in tally[("NEAR-DOMAIN OOS (should refuse)", rule)])
        print(f"  rule {rule}: in-scope answered {ok_in}/{len(IN)}   near-domain refused {ok_oos}/{len(NEAR_DOMAIN_OOS)}")
