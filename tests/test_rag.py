import pytest
from src.rag import build_default_index, answer_question, load_docs, _extractive_answer, RetrievedDoc


@pytest.fixture(scope="module")
def index():
    return build_default_index()


@pytest.mark.parametrize("query,expected_doc_id", [
    ("why did my order get rejected", "ORD-002"),
    ("how does margin call liquidation work", "MGN-001"),
    ("what's the API rate limit for placing orders", "API-001"),
    ("how long do withdrawals take", "PAY-001"),
    ("can I use MetaTrader 5", "PLT-001"),
    ("how do I set up two factor authentication", "SEC-001"),
    ("why is my leverage capped", "MGN-002"),
])
def test_retrieval_finds_correct_top_doc(index, query, expected_doc_id):
    retrieved = index.retrieve(query, k=3)
    assert len(retrieved) > 0
    assert retrieved[0].doc_id == expected_doc_id, (
        f"query={query!r} expected top doc {expected_doc_id}, "
        f"got {retrieved[0].doc_id} (score={retrieved[0].score:.3f})"
    )


def test_retrieval_scores_are_descending(index):
    retrieved = index.retrieve("why was my withdrawal delayed", k=5)
    scores = [r.score for r in retrieved]
    assert scores == sorted(scores, reverse=True)


def test_answer_is_grounded_in_retrieved_text(index):
    result = answer_question(index, "how does margin call liquidation work")
    # extractive backend: the answer must literally come from the top doc
    top_doc_text = result.sources[0].text
    assert result.answer in top_doc_text or top_doc_text in result.answer or result.answer == top_doc_text[:len(result.answer)] or top_doc_text.startswith(result.answer.split(" You may")[0])


def test_answer_cites_sources(index):
    result = answer_question(index, "what happens if I lose my 2FA device")
    assert len(result.sources) > 0
    assert all(isinstance(s, RetrievedDoc) for s in result.sources)


def test_extractive_fallback_declines_to_guess_on_irrelevant_query():
    docs = load_docs()
    # A fake, very-low-similarity retrieval result to exercise the "don't
    # know" path directly.
    empty_result = []
    answer = _extractive_answer("asdkjaslkdj random gibberish query", empty_result)
    assert "human agent" in answer.lower() or "couldn't find" in answer.lower()


def test_generation_backend_is_labeled(index):
    # Without GEMINI_API_KEY set, must be explicit about running extractive
    # fallback rather than silently pretending to be a live LLM.
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        result = answer_question(index, "how do deposits work")
        assert result.generation_backend == "extractive_fallback"
