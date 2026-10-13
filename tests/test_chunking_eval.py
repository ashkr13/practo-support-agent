"""Milestone 3 tests: precision/recall arithmetic (Task 5)."""
from app.rag import build_index
from eval.chunking_eval import evaluate, is_complete_sentence, mean, score_query
from eval.rag_queries import IN_SCOPE


def setup_module():
    build_index()


def test_dedup_and_arithmetic_single_relevant_doc():
    s = score_query("How late can I cancel?", {"02_cancellation_rescheduling"}, "sentence")
    assert len(s.retrieved_chunks) == 3
    assert len(s.retrieved_docs) == len(set(s.retrieved_chunks))  # deduplicated
    assert s.precision == s.hits / len(s.retrieved_docs)
    assert s.recall == s.hits / 1


def test_unretrieved_relevant_doc_halves_recall_keeps_precision():
    q = "How late can I cancel?"
    one = score_query(q, {"02_cancellation_rescheduling"}, "sentence")
    two = score_query(q, {"02_cancellation_rescheduling", "12_home_visits"}, "sentence")
    assert "12_home_visits" not in two.retrieved_docs
    assert two.precision == one.precision
    assert two.recall == one.recall / 2


def test_scores_in_unit_interval_for_both_strategies():
    for strategy in ("fixed", "sentence"):
        scores = evaluate(IN_SCOPE, strategy)
        assert len(scores) >= 5
        assert 0 <= mean([s.precision for s in scores]) <= 1
        assert 0 <= mean([s.recall for s in scores]) <= 1


def test_complete_sentence_rule():
    assert is_complete_sentence("Fees are paid online.")
    assert not is_complete_sentence("less than 4 hours before the slot.")
    assert not is_complete_sentence("Cardiology costs between 800 and 2,000...")
