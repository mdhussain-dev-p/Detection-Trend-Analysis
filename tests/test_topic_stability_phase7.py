from arxiv_topic_trends.models.compare_topics_phase7 import (
    jaccard_similarity,
    matched_topic_jaccard,
)


def test_jaccard_similarity():
    assert jaccard_similarity(["a", "b"], ["b", "c"]) == 1 / 3


def test_jaccard_empty_sets():
    assert jaccard_similarity([], []) == 1.0


def test_matched_topic_jaccard_identical_topics():
    topics = [["learning", "model"], ["network", "graph"]]
    assert matched_topic_jaccard(topics, topics) == 1.0


def test_matched_topic_jaccard_handles_reordered_topics():
    left = [["learning", "model"], ["network", "graph"]]
    right = [["network", "graph"], ["learning", "model"]]
    assert matched_topic_jaccard(left, right) == 1.0
