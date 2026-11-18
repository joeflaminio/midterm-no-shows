import numpy as np


def total_cost(y_true, call_decisions, call_cost=8, miss_cost=35):
    """Calculate the total cost of patient reminder calls.

    Each call costs $8. An actual no-show that was not called
    costs an additional $35.
    """
    actual = np.asarray(y_true)
    calls = np.asarray(call_decisions)

    if actual.ndim != 1 or calls.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(actual) != len(calls):
        raise ValueError("Inputs must have the same length.")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.isin(calls, [0, 1]).all():
        raise ValueError("call_decisions must contain only 0 and 1.")

    if call_cost < 0 or miss_cost < 0:
        raise ValueError("Costs cannot be negative.")

    number_called = np.count_nonzero(calls)
    missed_without_call = np.count_nonzero((actual == 1) & (calls == 0))

    return int(number_called * call_cost + missed_without_call * miss_cost)


def choose_threshold(
    y_true,
    oof_probabilities,
    thresholds,
    call_cost=8,
    miss_cost=35,
):
    """Choose the threshold with the lowest cost using training predictions.

    A patient is called when their predicted probability is
    greater than or equal to the threshold.

    When costs are tied, choose the lower threshold.
    """
    actual = np.asarray(y_true)
    probabilities = np.asarray(oof_probabilities, dtype=float)
    candidates = np.asarray(thresholds, dtype=float)

    if actual.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")

    if len(actual) != len(probabilities):
        raise ValueError("Labels and probabilities must have the same length.")

    if len(actual) == 0:
        raise ValueError("Training data cannot be empty.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("Thresholds must be a non-empty one-dimensional array.")

    if not np.isfinite(candidates).all():
        raise ValueError("Thresholds must be finite.")

    if not ((candidates >= 0.05) & (candidates <= 0.95)).all():
        raise ValueError("Thresholds must be between 0.05 and 0.95.")

    best_threshold = None
    best_cost = float("inf")

    for threshold in sorted(candidates):
        calls = probabilities >= threshold
        cost = total_cost(actual, calls, call_cost, miss_cost)

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    return best_threshold
