"""Task 16: in-memory cache keyed by normalized query text for the grounded-generation step."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag  # noqa: E402
from app.memory import reset_session  # noqa: E402
from app.pipeline import ask  # noqa: E402


def timed(query: str):
    g0, r0 = rag.GENERATOR_CALLS, rag.RETRIEVE_CALLS
    t = time.perf_counter()
    result = rag.grounded_answer(query, "sentence")
    ms = (time.perf_counter() - t) * 1000
    return result, ms, rag.GENERATOR_CALLS - g0, rag.RETRIEVE_CALLS - r0


if __name__ == "__main__":
    rag.clear_cache()
    rag.get_embedder()  # load the model first so call 1 timing measures the work, not model loading
    print("Part A - grounded-generation step directly")
    queries = ["How long will my thyroid test results take?",
               "How long will my thyroid test results take?",
               "  how long will my THYROID test results take  "]
    first = None
    for i, q in enumerate(queries, 1):
        r, ms, gen, ret = timed(q)
        first = first or r
        print(f"  call {i}: query={q!r}\n          key={rag.normalize_query(q)!r}")
        print(f"          cache_hit={r.cache_hit!s:<5} retrieval_calls={ret} generator_calls={gen} "
              f"time={ms:8.3f} ms  same_answer_as_call_1={r.answer == first.answer}")
    print(f"  cache stats: {rag.CACHE_STATS}")

    print("\nPart B - full pipeline (crew -> rag_lookup tool), repeated question in a fresh session")
    rag.clear_cache()
    q = "Can I get my antibiotics refilled without seeing the doctor again?"
    for i in (1, 2):
        reset_session(f"cache-{i}")
        g0, r0 = rag.GENERATOR_CALLS, rag.RETRIEVE_CALLS
        resp = ask(q, f"cache-{i}")
        print(f"  ask #{i}: cache_hit={resp.cache_hit!s:<5} retrieval_calls={rag.RETRIEVE_CALLS - r0} "
              f"generator_calls={rag.GENERATOR_CALLS - g0}  answer={resp.response.answer[:70]}...")
    print(f"  cache stats: {rag.CACHE_STATS}")
    print("\nNote: the cache is cleared whenever the KB changes (build_index / add_document), so a newly "
          "added document is never hidden behind a stale cached answer.")
