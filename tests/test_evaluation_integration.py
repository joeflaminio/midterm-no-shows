"""Real evaluation integration on hand-worked toy values, never course data."""

import numpy as np
import pandas as pd
import pytest

from noshows.train import evaluate_final, project_interfaces


def test_real_threshold_and_cost_contract():
    api = project_interfaces()
    y = pd.Series([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.4, 0.1])
    threshold = api.choose_threshold(y, probabilities, np.arange(5, 96, 5) / 100)
    assert threshold == pytest.approx(0.45)
    assert api.total_cost(y, probabilities >= threshold) == 16


@pytest.mark.parametrize("threshold", [0.49, 0.50, 0.51])
def test_real_cost_reaches_threshold_boundary(threshold):
    api = project_interfaces()

    class ToyModel:
        def predict_proba(self, frame):
            return np.array([[0.5, 0.5], [0.8, 0.2]])

    frame = pd.DataFrame({feature: [0, 0] for feature in api.features})
    frame["no_show"] = [1, 0]
    result = evaluate_final(ToyModel(), frame, threshold, api)
    assert result["costs"]["call_nobody"] == 35
    assert result["costs"]["call_everybody"] == 16
    assert result["costs"]["model"] == (8 if threshold <= 0.5 else 35)
