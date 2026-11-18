import numpy as np
import pytest

from noshows.evaluation import choose_threshold, total_cost


def test_total_cost_mixed_decisions():
    actual = [1, 1, 0, 0]
    calls = [1, 0, 1, 0]

    assert total_cost(actual, calls) == 51


def test_total_cost_call_nobody():
    actual = [1, 0, 1, 0]
    calls = [0, 0, 0, 0]

    assert total_cost(actual, calls) == 70


def test_total_cost_call_everybody():
    actual = [1, 0, 1, 0]
    calls = [1, 1, 1, 1]

    assert total_cost(actual, calls) == 32


def test_total_cost_no_missed_appointments():
    actual = [0, 0, 0]
    calls = [0, 0, 0]

    assert total_cost(actual, calls) == 0


def test_total_cost_custom_costs():
    actual = [1, 1, 0]
    calls = [1, 0, 1]

    assert total_cost(actual, calls, call_cost=10, miss_cost=40) == 60


def test_total_cost_mismatched_lengths():
    with pytest.raises(ValueError):
        total_cost([1, 0, 1], [1, 0])


def test_total_cost_invalid_labels():
    with pytest.raises(ValueError):
        total_cost([1, 2, 0], [1, 0, 0])


def test_choose_threshold_lowest_cost():
    actual = [1, 1, 0, 0]
    probabilities = [0.9, 0.6, 0.4, 0.1]
    thresholds = [0.3, 0.5, 0.7]

    assert choose_threshold(actual, probabilities, thresholds) == pytest.approx(0.5)


def test_choose_threshold_tie_chooses_lower():
    actual = [1, 0]
    probabilities = [0.9, 0.1]
    thresholds = [0.7, 0.3, 0.5]

    assert choose_threshold(actual, probabilities, thresholds) == pytest.approx(0.3)


def test_choose_threshold_probability_equals_threshold():
    actual = [1]
    probabilities = [0.5]
    thresholds = [0.5, 0.6]

    assert choose_threshold(actual, probabilities, thresholds) == pytest.approx(0.5)


def test_choose_threshold_invalid_probabilities():
    with pytest.raises(ValueError):
        choose_threshold([1, 0], [1.2, 0.3], [0.5])


def test_choose_threshold_empty_thresholds():
    with pytest.raises(ValueError):
        choose_threshold([1, 0], [0.8, 0.2], [])


def test_choose_threshold_full_grid():
    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)

    assert len(thresholds) == 19
    assert thresholds[0] == pytest.approx(0.05)
    assert thresholds[-1] == pytest.approx(0.95)
