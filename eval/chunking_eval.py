"""Task 5: document-level precision/recall for each chunking strategy.

Chunks are mapped back to their parent doc_id and de-duplicated before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import config
from app.rag import retrieve


@dataclass
class QueryScore:
    query: str
    relevant: set[str]
    retrieved_chunks: list[str]  # parent doc_id of each top-k chunk, in rank order (with repeats)
    retrieved_docs: list[str]    # de-duplicated, rank order
    hits: int
    precision: float
    recall: float


def score_query(query: str, relevant: set[str], strategy: str, k: int = config.TOP_K) -> QueryScore:
    chunk_docs = [c.doc_id for c in retrieve(query, strategy, k)]
    docs = list(dict.fromkeys(chunk_docs))
    hits = len(set(docs) & relevant)
    return QueryScore(query, relevant, chunk_docs, docs, hits,
                      hits / len(docs) if docs else 0.0, hits / len(relevant))


def evaluate(queries: list[dict], strategy: str, k: int = config.TOP_K) -> list[QueryScore]:
    return [score_query(q["query"], q["relevant"], strategy, k) for q in queries]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def is_complete_sentence(piece: str) -> bool:
    """A complete sentence starts with a capital letter and ends with . ! or ? (not '...')."""
    return piece[:1].isupper() and piece[-1:] in ".!?" and not piece.endswith("...")


def answer_completeness(queries: list[dict], strategy: str) -> tuple[int, int]:
    """(complete answer pieces, total answer pieces) across the grounded answers."""
    from app.kb import split_sentences
    from app.rag import grounded_answer

    complete = total = 0
    for q in queries:
        pieces = split_sentences(grounded_answer(q["query"], strategy).answer)
        complete += sum(is_complete_sentence(p) for p in pieces)
        total += len(pieces)
    return complete, total
