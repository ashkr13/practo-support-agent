"""Task 4: empirically calibrate the "I don't know" threshold on top-1 cosine similarity.

Measures top-1 similarity for in-scope and out-of-scope queries in BOTH collections,
then proposes a threshold at the midpoint of the gap between the two observed clusters.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import STRATEGIES, retrieve  # noqa: E402
from eval.rag_queries import IN_SCOPE, IN_SCOPE_CALIBRATION_EXTRA, NEAR_DOMAIN_OOS, OUT_OF_SCOPE  # noqa: E402
from app.config import SIMILARITY_THRESHOLD  # noqa: E402

in_scope = [q["query"] for q in IN_SCOPE] + IN_SCOPE_CALIBRATION_EXTRA


def top1(query: str, strategy: str) -> tuple[float, str]:
    hit = retrieve(query, strategy, k=1)[0]
    return hit.similarity, hit.doc_id


if __name__ == "__main__":
    for strategy in STRATEGIES:
        print(f"\n=== Collection: {strategy} ===")
        ins, oos = [], []
        print("IN-SCOPE")
        for q in in_scope:
            s, d = top1(q, strategy)
            ins.append(s)
            print(f"  {s:.4f}  [{d}]  {q}")
        print("OUT-OF-SCOPE")
        for q in OUT_OF_SCOPE:
            s, d = top1(q, strategy)
            oos.append(s)
            print(f"  {s:.4f}  [{d}]  {q}")
        lo_in, hi_oos = min(ins), max(oos)
        print(f"in-scope cluster:     min={lo_in:.4f} max={max(ins):.4f}")
        print(f"out-of-scope cluster: min={min(oos):.4f} max={hi_oos:.4f}")
        if lo_in > hi_oos:
            print(f"gap = {lo_in - hi_oos:.4f}; midpoint threshold = {(lo_in + hi_oos) / 2:.4f}")
        else:
            print("WARNING: clusters overlap -- no clean threshold exists")
        t = SIMILARITY_THRESHOLD[strategy]
        print(f"CHOSEN threshold (app/config.py) = {t}")
        print("NEAR-DOMAIN OUT-OF-SCOPE stress test (not used for calibration)")
        for q in NEAR_DOMAIN_OOS:
            s, d = top1(q, strategy)
            verdict = "passes gate (needs groundedness check)" if s >= t else "blocked"
            print(f"  {s:.4f}  [{d}]  {q}  -> {verdict}")
