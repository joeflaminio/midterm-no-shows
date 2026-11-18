"""Synthetic-only finalization safety checks."""

import json

import pandas as pd
import pytest

from noshows import finalize_training as final
from noshows import train
from test_training_diagnostics import toy_frame


def test_frozen_finalization_once_only(tmp_path, monkeypatch):
    frame = toy_frame()
    calls = []
    original = train.evaluate_final

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(train, "evaluate_final", counted)
    diagnostics = {
        "threshold": 0.45,
        "settings": {"C": 0.1, "solver": "lbfgs", "class_weight": "balanced"},
        "honest_auc_mean": 0.7,
        "leaky_auc_mean": 0.9,
    }
    metrics = final.finalize(
        tmp_path, frame, pd.DataFrame({"model": ["toy"]}), diagnostics
    )
    assert len(calls) == 1
    assert metrics["test_size"] == 30
    assert (
        json.loads((tmp_path / "models/model_info.json").read_text())["threshold"]
        == 0.45
    )
    assert (
        json.loads((tmp_path / "reports/final_run_lock.json").read_text())["status"]
        == "completed_once"
    )
    with pytest.raises(FileExistsError):
        final.finalize(tmp_path, frame, pd.DataFrame(), diagnostics)
    assert len(calls) == 1


def test_reject_changed_frozen_settings(tmp_path):
    with pytest.raises(ValueError):
        final.finalize(tmp_path, toy_frame(), pd.DataFrame(), {"threshold": 0.5})
    assert not (tmp_path / "models").exists()
