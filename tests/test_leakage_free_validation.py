from arxiv_topic_trends.models.leakage_free_validation import (
    jaccard_similarity,
    matched_topic_jaccard,
)


def test_jaccard_similarity():
    assert abs(jaccard_similarity(["a", "b"], ["b", "c"]) - (1 / 3)) < 1e-12


def test_jaccard_empty_sets():
    assert jaccard_similarity([], []) == 1.0


def test_matched_topic_jaccard_identical_topics():
    topics = [["learning", "model"], ["network", "graph"]]
    assert matched_topic_jaccard(topics, topics) == 1.0


def test_matched_topic_jaccard_reordered_topics():
    left = [["learning", "model"], ["network", "graph"]]
    right = [["network", "graph"], ["learning", "model"]]
    assert matched_topic_jaccard(left, right) == 1.0
