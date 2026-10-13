"""Milestone 2 tests: chunking (Task 3), indexing, retrieval and grounded generation (Task 4)."""
import pytest

from app import config
from app.kb import Document, load_documents, split_sentences
from app.rag import (
    COLLECTION_NAMES, FALLBACK_ANSWER, build_chunks, build_index, chunk_fixed,
    chunk_sentences, get_collection, grounded_answer, retrieve,
)
from eval.rag_queries import IN_SCOPE, OUT_OF_SCOPE

DOC = Document("x", "Test", " ".join(f"w{i}" for i in range(50)))


@pytest.fixture(scope="module", autouse=True)
def index():
    build_index()


def test_fixed_chunks_have_size_and_overlap():
    chunks = chunk_fixed(DOC, size=20, overlap=5)
    assert [len(c.text.split()) for c in chunks] == [20, 20, 20]  # starts at 0, 15, 30
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]  # 5-word overlap
    assert chunks[-1].text.split()[-1] == "w49"  # nothing lost at the end


def test_fixed_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_fixed(DOC, size=10, overlap=10)


def test_sentence_chunks_are_one_sentence_each():
    doc = load_documents()[0]
    assert [c.text for c in chunk_sentences(doc)] == split_sentences(doc.text)


def test_two_separate_populated_collections():
    assert COLLECTION_NAMES["fixed"] != COLLECTION_NAMES["sentence"]
    for strategy in ("fixed", "sentence"):
        assert get_collection(strategy).count() == len(build_chunks(strategy))


def test_both_collections_retrieve_sensibly():
    for strategy in ("fixed", "sentence"):
        assert retrieve("How late can I cancel?", strategy, k=1)[0].doc_id == "02_cancellation_rescheduling"


@pytest.mark.parametrize("strategy", ["fixed", "sentence"])
def test_out_of_scope_triggers_fallback(strategy):
    for q in OUT_OF_SCOPE:
        r = grounded_answer(q, strategy)
        assert not r.grounded and r.answer == FALLBACK_ANSWER


@pytest.mark.parametrize("strategy", ["fixed", "sentence"])
def test_in_scope_answers_are_grounded_in_context(strategy):
    for item in IN_SCOPE:
        r = grounded_answer(item["query"], strategy)
        assert r.grounded and r.top1_similarity >= config.SIMILARITY_THRESHOLD[strategy]
        context = " ".join(c.text for c in r.context)
        # Every answer sentence is copied verbatim from retrieved context (grounded by construction).
        for sentence in split_sentences(r.answer):
            assert sentence.rstrip(".") in context or sentence in context
