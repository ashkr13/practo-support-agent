"""Build (or rebuild) both ChromaDB collections from kb/*.md (Task 3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import COLLECTION_NAMES, build_chunks, build_index, retrieve  # noqa: E402

if __name__ == "__main__":
    counts = build_index()
    for strategy, n in counts.items():
        sample = build_chunks(strategy)[0]
        print(f"[{strategy}] collection={COLLECTION_NAMES[strategy]!r} chunks={n}")
        print(f"   e.g. {sample.chunk_id}: {sample.text!r}")
    q = "How late can I cancel my appointment?"
    print(f"\nSanity retrieval for: {q!r}")
    for strategy in counts:
        top = retrieve(q, strategy, k=1)[0]
        print(f"   [{strategy}] top-1 sim={top.similarity:.4f} doc={top.doc_id} :: {top.text[:90]}...")
