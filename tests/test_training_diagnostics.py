"""Toy-only safety and independent checks for training diagnostics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from noshows import pipeline, train
from noshows import training_diagnostics as diagnostics


def toy_frame():
    rng = np.random.default_rng(908)
    n = 120
    frame = pd.DataFrame({key: rng.normal(size=n) for key in pipeline.NUMERIC_FEATURES})
    for key in pipeline.CATEGORICAL_FEATURES:
        frame[key] = np.where(np.arange(n) % 2, "toy_a", "toy_b")
    frame["no_show"] = (np.arange(n) % 3 == 0).astype(int)
    frame["fee_charged"] = frame["no_show"] * 35
    return frame


def test_diagnostic_oof_cost_matches_independent_sklearn(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Final evaluation/output must never run")

    monkeypatch.setattr(train, "evaluate_final", forbidden)
    monkeypatch.setattr(train, "save_outputs", forbidden)
    frame = toy_frame()
    costs, folds, summary = diagnostics.analyze(frame)
    model = train.candidates(pipeline.SEED)["logistic_balanced"]
    model.set_params(C=0.1, solver="lbfgs")
    p = cross_val_predict(
        pipeline.build_pipeline(model),
        frame[pipeline.ALL_FEATURES],
        frame.no_show,
        cv=StratifiedKFold(5, shuffle=True, random_state=11),
        method="predict_proba",
    )[:, 1]
    expected = []
    for t in costs.threshold:
        calls = p >= t
        expected.append(8 * calls.sum() + 35 * ((frame.no_show == 1) & ~calls).sum())
    assert costs.cost.tolist() == expected
    assert summary["threshold"] == costs.iloc[np.argmin(expected)].threshold
    assert len(folds) == 5
    assert not summary["holdout_evaluated"]
    assert not summary["final_model_saved"]


def test_existing_diagnostic_directory_is_protected(tmp_path):
    with pytest.raises(FileExistsError):
        diagnostics.reproduce(tmp_path)


def test_partition_keeps_only_training_rows(monkeypatch):
    frame = toy_frame()
    x = frame[pipeline.ALL_FEATURES]
    monkeypatch.setattr(
        pipeline,
        "split_data",
        lambda _: (
            x.iloc[:90],
            x.iloc[90:],
            frame.no_show.iloc[:90],
            frame.no_show.iloc[90:],
        ),
    )
    actual = diagnostics.training_frame(lambda **_: frame)
    pd.testing.assert_frame_equal(actual, frame.iloc[:90][actual.columns])
