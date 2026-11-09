import numpy as np

from arxiv_topic_trends.models.train_pilot import (
    choose_sample_indices,
    extract_top_terms,
)


def test_sampling_is_reproducible():
    a = choose_sample_indices(100, 20, 42)
    b = choose_sample_indices(100, 20, 42)

    assert np.array_equal(a, b)


def test_sampling_has_unique_sorted_indices():
    result = choose_sample_indices(100, 20, 42)

    assert len(result) == 20
    assert len(np.unique(result)) == 20
    assert np.all(np.diff(result) >= 0)


def test_sampling_handles_small_dataset():
    result = choose_sample_indices(5, 20, 42)

    assert len(result) == 5


def test_top_term_extraction():
    components = np.array([
        [0.1, 0.9, 0.5],
        [0.8, 0.2, 0.6],
    ])

    names = np.array(["graph", "learning", "model"])

    result = extract_top_terms(
        components,
        names,
        "NMF",
        top_n=2,
    )

    assert len(result) == 4
    assert result[0]["term"] == "learning"
    assert result[2]["term"] == "graph"


def test_top_terms_preserve_model_identity():
    result = extract_top_terms(
        np.array([[0.1, 0.9]]),
        np.array(["graph", "learning"]),
        "LDA",
        top_n=1,
    )

    assert result[0]["model"] == "LDA"
