import numpy as np

from arxiv_topic_trends.models.analyze_topic_trends import (
    calculate_year_metrics,
)


def test_annual_topic_shares_sum_to_100():
    years = np.array([2020, 2020, 2020])
    mask = np.array([True, True, True])

    proportions = np.zeros((3, 10), dtype=float)
    proportions[0, 0] = 1.0
    proportions[1, 1] = 1.0
    proportions[2, 0] = 0.5
    proportions[2, 1] = 0.5

    result = calculate_year_metrics(2020, years, mask, proportions)

    assert abs(result["mean_share_pct"].sum() - 100.0) < 1e-8
    assert abs(result["dominant_share_pct"].sum() - 100.0) < 1e-8


def test_unassigned_documents_are_excluded_from_topic_denominator():
    years = np.array([2020, 2020, 2020])
    mask = np.array([True, True, False])

    proportions = np.zeros((3, 10), dtype=float)
    proportions[0, 0] = 1.0
    proportions[1, 1] = 1.0

    result = calculate_year_metrics(2020, years, mask, proportions)

    assert result["total_papers"] == 3
    assert result["modelled_papers"] == 2
    assert result["unassigned_papers"] == 1
    assert result["dominant_counts"][0] == 1
    assert result["dominant_counts"][1] == 1


def test_mean_shares_sum_to_one_hundred_for_multiple_topics():
    years = np.array([2021, 2021])
    mask = np.array([True, True])

    proportions = np.zeros((2, 10), dtype=float)
    proportions[0, 2] = 0.7
    proportions[0, 3] = 0.3
    proportions[1, 2] = 0.2
    proportions[1, 3] = 0.8

    result = calculate_year_metrics(2021, years, mask, proportions)

    assert np.isclose(result["mean_share_pct"][2], 45.0)
    assert np.isclose(result["mean_share_pct"][3], 55.0)
    assert np.isclose(result["mean_share_pct"].sum(), 100.0)
