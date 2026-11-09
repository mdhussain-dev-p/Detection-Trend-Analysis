import json
import numpy as np

from scipy.sparse import isspmatrix_csr
from arxiv_topic_trends.features.vectorize import (
    create_vectorizers,
    iter_documents,
)


def small_vectorizers():
    return create_vectorizers(
        min_df=1,
        max_df=1.0,
        max_features=100,
    )


def test_jsonl_document_reader(tmp_path):
    path = tmp_path / "sample.jsonl"
    records = [
        {"tokens": ["graph", "learning", "model"]},
        {"tokens": ["neural", "network"]},
    ]

    path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )

    assert list(iter_documents(path)) == [
        "graph learning model",
        "neural network",
    ]


def test_vectorizers_have_identical_vocabularies():
    documents = [
        "graph learning model",
        "graph neural model",
        "deep learning network",
        "machine learning model",
    ]

    tfidf, counts = small_vectorizers()

    a = tfidf.fit_transform(iter(documents))
    b = counts.fit_transform(iter(documents))

    assert np.array_equal(
        tfidf.get_feature_names_out(),
        counts.get_feature_names_out(),
    )
    assert a.shape == b.shape


def test_tfidf_is_sparse_float32():
    tfidf, _ = small_vectorizers()

    matrix = tfidf.fit_transform(iter([
        "graph learning model",
        "graph neural model",
        "deep learning network",
    ]))

    assert isspmatrix_csr(matrix)
    assert matrix.dtype == np.float32


def test_counts_are_sparse_integer_values():
    _, counts = small_vectorizers()

    matrix = counts.fit_transform(iter([
        "graph learning model",
        "graph neural model",
        "deep learning network",
    ]))

    assert isspmatrix_csr(matrix)
    assert matrix.dtype == np.int32
    assert matrix.sum() > 0


def test_only_unigrams_are_used():
    tfidf, counts = small_vectorizers()
    assert tfidf.ngram_range == (1, 1)
    assert counts.ngram_range == (1, 1)


def test_production_parameters_are_preserved():
    tfidf, counts = create_vectorizers()

    for vectorizer in (tfidf, counts):
        assert vectorizer.min_df == 5
        assert vectorizer.max_df == 0.90
        assert vectorizer.max_features == 8000
