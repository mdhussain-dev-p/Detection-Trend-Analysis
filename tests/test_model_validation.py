import numpy as np
from scipy.sparse import csr_matrix

from arxiv_topic_trends.models.evaluate_pilot import (
    split_sample,
    sparse_reconstruction_error,
)


def test_train_and_validation_rows_are_disjoint():
    rows = np.arange(100)

    train, validation = split_sample(rows)

    assert len(train) == 80
    assert len(validation) == 20
    assert not set(train).intersection(set(validation))


def test_split_is_reproducible():
    rows = np.arange(100)

    train_a, valid_a = split_sample(rows, seed=42)
    train_b, valid_b = split_sample(rows, seed=42)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(valid_a, valid_b)


def test_split_covers_original_rows():
    rows = np.arange(50)

    train, validation = split_sample(rows)

    combined = np.sort(np.concatenate([train, validation]))

    assert np.array_equal(combined, rows)


def test_sparse_reconstruction_error():
    matrix = csr_matrix([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])

    components = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ])

    document_topics = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    error = sparse_reconstruction_error(
        matrix,
        document_topics,
        components,
    )

    assert error >= 0
    assert error < 2


def test_invalid_validation_fraction_is_rejected():
    try:
        split_sample(np.arange(10), validation_fraction=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid split to be rejected.")
