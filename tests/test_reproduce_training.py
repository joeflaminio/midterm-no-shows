"""Safety checks for training-only reproduction."""

import numpy as np
import pandas as pd
import pytest

from noshows import reproduce_training as reproduce


def test_existing_output_is_never_overwritten(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        reproduce.reproduce(tmp_path)
    assert marker.read_text() == "keep"


def test_training_partition_returns_only_training_values(monkeypatch):
    frame = pd.DataFrame({"x": [1, 2, 99], "no_show": [0, 1, 1]})
    monkeypatch.setattr(
        reproduce.pipeline,
        "split_data",
        lambda data: (
            data.loc[:1, ["x"]],
            data.loc[2:, ["x"]],
            data.loc[:1, "no_show"],
            data.loc[2:, "no_show"],
        ),
    )
    x, y, manifest = reproduce.training_partition(lambda **kwargs: frame)
    assert x["x"].tolist() == [1, 2]
    assert y.tolist() == [0, 1]
    assert manifest == {"train_indices": [0, 1], "reserved_test_indices": [2]}


def test_generator_hash_is_checked_before_execution(monkeypatch):
    monkeypatch.setattr(
        reproduce.subprocess, "check_output", lambda args: b"bad source"
    )
    with pytest.raises(ValueError, match="hash"):
        reproduce.load_generator(".")


def test_summary_population_std():
    scores = {
        f"test_{metric}": np.array([0.0, 1.0]) for metric in reproduce.train.SCORING
    }
    row = reproduce.summarize("fixture", scores)
    assert row["f1_mean"] == 0.5
    assert row["f1_std"] == 0.5
