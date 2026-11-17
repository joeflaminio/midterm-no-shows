"""Local integration check: toy rows only; never uses the course generator."""

import numpy as np
import pandas as pd

from noshows import pipeline as joe
from noshows.train import Interfaces, prepare_training, restore_split_frames


def test_joe_pipeline_training_integration():
    rng = np.random.default_rng(908)
    n = 160
    frame = pd.DataFrame({name: rng.normal(size=n) for name in joe.NUMERIC_FEATURES})
    for name in joe.CATEGORICAL_FEATURES:
        frame[name] = np.where(np.arange(n) % 2, "toy_a", "toy_b")
    frame["no_show"] = (np.arange(n) % 3 == 0).astype(int)
    frame["appointment_id"] = np.arange(n)
    frame["fee_charged"] = frame["no_show"] * 35
    frame.loc[::13, "distance_km"] = np.nan
    train, test = restore_split_frames(frame, joe.split_data(frame))
    assert len(train) == 120 and len(test) == 40
    assert not len(train.index.intersection(test.index))

    def cost(y, calls):
        return 8 * np.asarray(calls).sum() + 35 * np.sum(
            (np.asarray(y) == 1) & ~np.asarray(calls, dtype=bool)
        )

    def threshold(y, probabilities, grid):
        return min(grid, key=lambda t: cost(y, probabilities >= t))

    api = Interfaces(
        joe.SEED,
        tuple(joe.NUMERIC_FEATURES),
        tuple(joe.CATEGORICAL_FEATURES),
        joe.build_pipeline,
        cost,
        threshold,
    )
    model, comparison, details = prepare_training(train, api)
    assert len(comparison) == 4
    assert list(model.feature_names_in_) == joe.ALL_FEATURES
    assert details["selection_scoring"] == "f1"
    assert "fee_charged" not in model.feature_names_in_
    # Deliberately no prediction/evaluation on the toy test split either.
