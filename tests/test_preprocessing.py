from nltk.stem import WordNetLemmatizer

from arxiv_topic_trends.preprocessing.clean_abstracts import clean_abstract


def test_stopword_removal_and_noun_lemmatisation():
    result = clean_abstract(
        "Models and methods are useful.",
        {"and", "are"},
        WordNetLemmatizer(),
    )
    assert result == ["model", "method", "useful"]


def test_domain_terms_are_not_automatically_removed():
    result = clean_abstract(
        "The paper proposes a learning method.",
        {"the", "a"},
        WordNetLemmatizer(),
    )
    assert "paper" in result
    assert "proposes" in result
    assert "learning" in result


def test_numeric_only_and_single_character_tokens_are_removed():
    result = clean_abstract(
        "The 123 3D model and x.",
        {"the", "and"},
        WordNetLemmatizer(),
    )
    assert "123" not in result
    assert "x" not in result
    assert "3d" in result


def test_empty_abstract_returns_empty_list():
    assert clean_abstract("", set(), WordNetLemmatizer()) == []
