"""RAG core: two chunking strategies, local embeddings, ChromaDB indexing, retrieval,
and grounded generation (Tasks 3 and 4).

Under MOCK_LLM the "generator" is deterministic and extractive: it can only return
sentences that appear in the retrieved chunks, so every answer is grounded by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app import config
from app.kb import Document, load_documents, split_sentences

STRATEGIES = ("fixed", "sentence")
COLLECTION_NAMES = {"fixed": "kb_fixed_overlap", "sentence": "kb_sentence"}

FALLBACK_ANSWER = (
    "I don't know. I couldn't find this in Practo's clinic policy documents. "
    "Please contact the clinic front desk for help."
)


# ---------------------------------------------------------------- chunking (Task 3)

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    strategy: str
    index: int
    text: str


def chunk_fixed(doc: Document, size: int = config.FIXED_CHUNK_WORDS,
                overlap: int = config.FIXED_CHUNK_OVERLAP_WORDS) -> list[Chunk]:
    """Fixed-size windows of `size` words; each window starts `size - overlap` words later."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    words = doc.text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        window = words[start:start + size]
        chunks.append(Chunk(f"{doc.doc_id}::fixed::{len(chunks)}", doc.doc_id, doc.title,
                            "fixed", len(chunks), " ".join(window)))
        if start + size >= len(words):
            break
        start += step
    return chunks


def chunk_sentences(doc: Document) -> list[Chunk]:
    """One chunk per sentence."""
    return [
        Chunk(f"{doc.doc_id}::sentence::{i}", doc.doc_id, doc.title, "sentence", i, s)
        for i, s in enumerate(split_sentences(doc.text))
    ]


def build_chunks(strategy: str, docs: list[Document] | None = None) -> list[Chunk]:
    docs = docs if docs is not None else load_documents()
    chunker = {"fixed": chunk_fixed, "sentence": chunk_sentences}[strategy]
    return [c for d in docs for c in chunker(d)]


# ---------------------------------------------------------------- embeddings + index

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Unit-normalised embeddings, so dot product == cosine similarity."""
    return get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    # anonymized_telemetry=False: ChromaDB otherwise sends usage telemetry (network call).
    return chromadb.PersistentClient(path=str(config.CHROMA_PATH),
                                     settings=Settings(anonymized_telemetry=False))


def get_collection(strategy: str):
    # hnsw:space=cosine -> Chroma returns distance = 1 - cosine similarity.
    return get_client().get_or_create_collection(
        COLLECTION_NAMES[strategy], metadata={"hnsw:space": "cosine"})


def index_chunks(chunks: list[Chunk], strategy: str, prune: bool = False) -> int:
    """Embed and upsert chunks into the strategy's collection.

    prune=True also deletes chunk ids no longer produced (e.g. after changing chunk size).
    """
    collection = get_collection(strategy)
    if prune:
        current = {c.chunk_id for c in chunks}
        stale = [i for i in collection.get()["ids"] if i not in current]
        if stale:
            collection.delete(ids=stale)
    if chunks:
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embed([c.text for c in chunks]).tolist(),
            metadatas=[{"doc_id": c.doc_id, "title": c.title, "strategy": c.strategy,
                        "chunk_index": c.index} for c in chunks],
        )
    return collection.count()


def build_index(prune: bool = True) -> dict[str, int]:
    """(Re)build both collections from kb/. Idempotent thanks to upsert.

    prune=True removes chunks not produced from kb/ (including documents added via the API);
    the API starts with prune=False so documents added at runtime survive a restart.
    """
    docs = load_documents()
    counts = {s: index_chunks(build_chunks(s, docs), s, prune=prune) for s in STRATEGIES}
    clear_cache()
    return counts


def add_document(doc: Document) -> dict[str, int]:
    """Index one new document into both collections (used by POST /add-document).

    Clears the answer cache: a cached "I don't know" could now be answerable (stale cache).
    """
    counts = {s: index_chunks(build_chunks(s, [doc]), s) for s in STRATEGIES}
    clear_cache()
    return counts


# ---------------------------------------------------------------- retrieval

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    similarity: float


RETRIEVE_CALLS = 0  # embedding + vector-search calls; used for cache evidence in Task 16


def retrieve(query: str, strategy: str = "sentence", k: int = config.TOP_K) -> list[RetrievedChunk]:
    global RETRIEVE_CALLS
    RETRIEVE_CALLS += 1
    collection = get_collection(strategy)
    if collection.count() == 0:
        raise RuntimeError(f"Collection for '{strategy}' is empty. Run: python scripts/build_index.py")
    res = collection.query(query_embeddings=embed([query]).tolist(), n_results=k)
    return [
        RetrievedChunk(cid, meta["doc_id"], meta["title"], text, round(1.0 - dist, 4))
        for cid, meta, text, dist in zip(res["ids"][0], res["metadatas"][0],
                                         res["documents"][0], res["distances"][0])
    ]


# ---------------------------------------------------------------- grounded generation (Task 4)

@dataclass
class GroundedAnswer:
    query: str
    answer: str
    grounded: bool  # False -> fallback was used
    top1_similarity: float
    threshold: float
    strategy: str
    sources: list[str] = field(default_factory=list)  # parent doc_ids actually quoted
    context: list[RetrievedChunk] = field(default_factory=list)
    cache_hit: bool = False


GENERATOR_CALLS = 0  # counts (mock) LLM generation calls; used for cache evidence in Task 16


def mock_generate(query: str, context: list[RetrievedChunk], max_sentences: int = 2) -> tuple[str, list[str]]:
    """Deterministic MOCK_LLM generator: extract the context sentences most similar to the query.

    It never writes text that is not in `context`, so the answer is grounded by construction.
    """
    global GENERATOR_CALLS
    GENERATOR_CALLS += 1
    candidates: list[tuple[str, str]] = []  # (sentence, doc_id)
    for c in context:
        for s in split_sentences(c.text):
            if (s, c.doc_id) not in candidates:
                candidates.append((s, c.doc_id))
    sims = embed([s for s, _ in candidates]) @ embed([query])[0]
    picked: list[tuple[str, str]] = []
    seen_words: set[str] = set()
    for i in np.argsort(-sims):  # most relevant first
        words = set(candidates[i][0].lower().split())
        # Skip near-duplicates: overlapping fixed-size windows repeat the same words.
        if seen_words and len(words & seen_words) / len(words) >= 0.6:
            continue
        picked.append(candidates[i])
        seen_words |= words
        if len(picked) == max_sentences:
            break
    sources = list(dict.fromkeys(doc for _, doc in picked))
    # Fixed-size chunks can end mid-sentence; mark those fragments with "..." so they read as
    # truncated and are never fused with the next sentence into text that isn't in the context.
    pieces = [s if s[-1] in ".!?" else s + "..." for s, _ in picked]
    return " ".join(pieces), sources


# ---------------------------------------------------------------- response cache (Task 16)

_CACHE: dict[tuple[str, str, int, float], GroundedAnswer] = {}
CACHE_STATS = {"hits": 0, "misses": 0}


def normalize_query(query: str) -> str:
    """Cache key text: lower-case, punctuation removed, whitespace collapsed.
    'How long do LAB results take?' and 'how long do lab results take' share one entry."""
    return " ".join(re.sub(r"[^\w\s-]", " ", query.lower()).split())


def clear_cache() -> None:
    _CACHE.clear()
    CACHE_STATS.update(hits=0, misses=0)
    from app.guardrails import invalidate_kb_vocabulary  # KB changed -> refresh groundedness vocabulary
    invalidate_kb_vocabulary()


def grounded_answer(query: str, strategy: str = "sentence", k: int = config.TOP_K,
                    threshold: float | None = None, use_cache: bool = True) -> GroundedAnswer:
    """Retrieve top-k chunks and answer ONLY from them, or fall back to "I don't know".

    Results are cached in memory keyed by normalized query text (plus strategy/k/threshold), so a
    repeated question skips embedding, vector search and generation.
    """
    threshold = config.SIMILARITY_THRESHOLD[strategy] if threshold is None else threshold
    key = (normalize_query(query), strategy, k, threshold)
    if use_cache and key in _CACHE:
        CACHE_STATS["hits"] += 1
        return replace(_CACHE[key], query=query, cache_hit=True)
    CACHE_STATS["misses"] += 1
    result = _grounded_answer_uncached(query, strategy, k, threshold)
    if use_cache:
        _CACHE[key] = result
    return result


def _grounded_answer_uncached(query: str, strategy: str, k: int, threshold: float) -> GroundedAnswer:
    context = retrieve(query, strategy, k)
    top1 = context[0].similarity if context else 0.0
    if top1 < threshold:
        return GroundedAnswer(query, FALLBACK_ANSWER, False, top1, threshold, strategy, [], context)
    answer, sources = mock_generate(query, context)
    return GroundedAnswer(query, answer, True, top1, threshold, strategy, sources, context)
