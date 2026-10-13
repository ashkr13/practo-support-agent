"""Task 4: grounded generation on >=5 in-scope queries + 1 out-of-scope query (both collections)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import STRATEGIES, grounded_answer  # noqa: E402
from eval.rag_queries import IN_SCOPE  # noqa: E402

DEMO_QUERIES = [q["query"] for q in IN_SCOPE] + ["What is the capital of France?"]

if __name__ == "__main__":
    print("MOCK_LLM=true | generator = deterministic extractive (answers only from retrieved context)")
    for strategy in STRATEGIES:
        print(f"\n{'=' * 30} collection: {strategy} {'=' * 30}")
        for i, q in enumerate(DEMO_QUERIES, 1):
            r = grounded_answer(q, strategy)
            tag = "GROUNDED" if r.grounded else "FALLBACK"
            print(f"\nQ{i}: {q}")
            print(f"  top-1 similarity={r.top1_similarity:.4f} threshold={r.threshold} -> {tag}")
            print(f"  retrieved: {[c.doc_id + f' ({c.similarity:.3f})' for c in r.context]}")
            print(f"  A: {r.answer}")
            if r.sources:
                print(f"  sources: {r.sources}")
