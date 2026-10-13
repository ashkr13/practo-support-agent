"""Task 5: compare fixed-size-with-overlap vs sentence chunking on the Task 4 queries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.rag import STRATEGIES  # noqa: E402
from eval.chunking_eval import answer_completeness, evaluate, mean  # noqa: E402
from eval.rag_queries import IN_SCOPE  # noqa: E402

if __name__ == "__main__":
    print(f"Document-level precision/recall, top-k={config.TOP_K} chunks -> parent docs (deduplicated)")
    print("precision = |retrieved ∩ relevant| / |retrieved|    recall = |retrieved ∩ relevant| / |relevant|")
    summary = {}
    for strategy in STRATEGIES:
        print(f"\n=== {strategy} ===")
        scores = evaluate(IN_SCOPE, strategy)
        for i, s in enumerate(scores, 1):
            print(f"Q{i}: {s.query}")
            print(f"    top-{config.TOP_K} chunk parents: {s.retrieved_chunks}")
            print(f"    retrieved docs (dedup) R = {s.retrieved_docs}")
            print(f"    relevant docs G = {sorted(s.relevant)}")
            print(f"    precision = {s.hits}/{len(s.retrieved_docs)} = {s.precision:.3f}   "
                  f"recall = {s.hits}/{len(s.relevant)} = {s.recall:.3f}")
        p = mean([s.precision for s in scores])
        r = mean([s.recall for s in scores])
        summary[strategy] = (p, r)
        print(f"MEAN ({strategy}): precision = {' + '.join(f'{s.precision:.3f}' for s in scores)} / {len(scores)}"
              f" = {p:.3f}")
        print(f"                 recall    = {' + '.join(f'{s.recall:.3f}' for s in scores)} / {len(scores)}"
              f" = {r:.3f}")
    print("\nSUMMARY")
    for strategy, (p, r) in summary.items():
        print(f"  {strategy:<9} mean precision={p:.3f}  mean recall={r:.3f}")

    print("\nSUPPLEMENTARY: answer usability (pieces of the Task 4 answers that are complete sentences)")
    for strategy in STRATEGIES:
        c, t = answer_completeness(IN_SCOPE, strategy)
        print(f"  {strategy:<9} complete pieces = {c}/{t} = {c / t:.3f}")

    print("""
RECOMMENDATION: deploy SENTENCE chunking.
Both strategies reach the same mean recall (0.917 vs 0.917, each missing only the emergency doc on the
chest-pain query), and fixed-size chunking wins mean precision (1.000 vs 0.917) by a single
third-ranked off-topic chunk on Q4 that the generator never quoted. However, only 8/12 = 0.667 of the
answer pieces built from fixed-size chunks are complete sentences versus 12/12 = 1.000 for sentence
chunks, and a patient-facing policy answer that cuts off mid-sentence is less usable and harder to verify
for groundedness, so the 0.083 precision gap is outweighed by a 0.333 usability gap.""")
