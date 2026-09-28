import pytest
from src.backlog import generate_backlog, _classify_cluster_type, _core_text, load_tickets


@pytest.fixture(scope="module")
def backlog():
    return generate_backlog()


def test_finds_all_three_deliberately_injected_clusters(backlog):
    # data/generate_data.py injects 3 specific recurring-issue clusters of
    # known size (34, 27, 19 tickets). The detector must surface all three
    # somewhere in the backlog (bug or FAQ bucket) with roughly the right
    # ticket counts -- this is the core correctness check for the whole
    # clustering approach.
    titles_and_counts = [(b.title, b.ticket_count) for b in backlog]

    def find_cluster_containing(substring):
        matches = [tc for tc in titles_and_counts if substring.lower() in tc[0].lower()]
        return matches

    api_matches = find_cluster_containing("429 rate limit")
    assert api_matches, "API 429 recurring cluster not found in backlog"
    assert api_matches[0][1] >= 25, f"expected ~34 tickets, got {api_matches[0][1]}"

    stoploss_matches = find_cluster_containing("stop loss on XAU/USD")
    assert stoploss_matches, "stop-loss XAU/USD recurring cluster not found"
    assert stoploss_matches[0][1] >= 20, f"expected ~27 tickets, got {stoploss_matches[0][1]}"

    swap_matches = find_cluster_containing("swap fee charged on my GBP/JPY")
    assert swap_matches, "GBP/JPY swap fee recurring cluster not found"
    assert swap_matches[0][1] >= 15, f"expected ~19 tickets, got {swap_matches[0][1]}"


def test_injected_bug_clusters_are_classified_as_bugs_not_faqs(backlog):
    bug_titles = [b.title.lower() for b in backlog if b.cluster_type == "recurring_bug"]
    assert any("429 rate limit" in t for t in bug_titles)
    assert any("stop loss on xau/usd" in t for t in bug_titles)


def test_all_clusters_meet_minimum_size(backlog):
    from src.backlog import MIN_CLUSTER_SIZE
    for item in backlog:
        assert item.ticket_count >= MIN_CLUSTER_SIZE


def test_priority_never_p1_for_faq_clusters(backlog):
    for item in backlog:
        if item.cluster_type == "recurring_question":
            assert item.priority != "P1", (
                f"{item.backlog_id} is a documentation gap, not an incident -- "
                "should never be P1"
            )


def test_classification_survives_truncation():
    # Regression test for the truncation bug found during development:
    # classifying the truncated (110-char) display title instead of the
    # full representative text could silently chop off a trailing "?" and
    # misroute a genuine question into the engineering backlog.
    long_question = (
        "Trying to set up the MT5 bridge but also want to use the REST API "
        "at the same time, is that supported together for my account?"
    )
    assert len(long_question) > 110
    assert _classify_cluster_type(long_question) == "recurring_question"


def test_core_text_strips_greeting_boilerplate():
    raw = "Hello, My withdrawal has been pending for 4 days now, when will it actually arrive?"
    core = _core_text(raw)
    assert not core.lower().startswith("hello")
    assert "withdrawal" in core.lower()


def test_core_text_strips_urgency_sentences():
    raw = "My API key keeps returning 429 errors. This is URGENT, I'm losing money right now."
    core = _core_text(raw)
    assert "urgent" not in core.lower()
    assert "429" in core


def test_backlog_ids_are_unique(backlog):
    ids = [b.backlog_id for b in backlog]
    assert len(ids) == len(set(ids))


def test_affected_ticket_ids_actually_exist_in_source_data(backlog):
    all_ids = {t["ticket_id"] for t in load_tickets()}
    for item in backlog:
        for tid in item.affected_ticket_ids:
            assert tid in all_ids
