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


def seed_reproduction(root):
    (root / "reports").mkdir()
    diagnostics = {
        "threshold": 0.45,
        "settings": {"C": 0.1, "solver": "lbfgs", "class_weight": "balanced"},
        "honest_auc_mean": 0.7,
        "leaky_auc_mean": 0.9,
    }
    (root / "reports/training_diagnostics.json").write_text(json.dumps(diagnostics))
    pd.DataFrame({"model": ["toy"]}).to_csv(
        root / "reports/training_cv_summary.csv", index=False
    )


def test_repeat_frozen_reproduction_preserves_original_audit(tmp_path, monkeypatch):
    import subprocess

    seed_reproduction(tmp_path)
    marker = tmp_path / "reports/final_run_lock.json"
    marker.write_text("original evaluation audit")
    verification = tmp_path / "reports/final_verification.json"
    verification.write_text("original verification audit")
    monkeypatch.setattr(final, "make_appointments", lambda **_: toy_frame())

    def no_git(*args, **kwargs):
        raise AssertionError("Frozen reproduction must not execute Git")

    monkeypatch.setattr(subprocess, "check_output", no_git)
    monkeypatch.setenv("PATH", "")
    first = final.reproduce(tmp_path)
    original_model = (tmp_path / "models/no_show.joblib").read_bytes()
    second = final.reproduce(tmp_path)
    assert first == second
    assert original_model == (tmp_path / "models/no_show.joblib").read_bytes()
    assert marker.read_text() == "original evaluation audit"
    assert verification.read_text() == "original verification audit"
    assert not list(tmp_path.glob(".frozen-reproduction-*"))


def test_failed_reproduction_leaves_existing_outputs(tmp_path, monkeypatch):
    seed_reproduction(tmp_path)
    (tmp_path / "models").mkdir()
    model = tmp_path / "models/no_show.joblib"
    model.write_bytes(b"original")
    monkeypatch.setattr(final, "make_appointments", lambda **_: toy_frame())

    def fail(*args, **kwargs):
        raise RuntimeError("simulated fit failure")

    monkeypatch.setattr(final, "finalize", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        final.reproduce(tmp_path)
    assert model.read_bytes() == b"original"
    assert not list(tmp_path.glob(".frozen-reproduction-*"))


def test_reproduction_requires_no_cached_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "make_appointments", lambda **_: toy_frame())
    metrics = final.reproduce(tmp_path)
    assert metrics["test_size"] == 30
    assert len(pd.read_csv(tmp_path / "reports/cv_results.csv")) == 4
    assert len(pd.read_csv(tmp_path / "reports/reproduced_leakage_folds.csv")) == 5
    info = json.loads((tmp_path / "models/model_info.json").read_text())
    assert info["threshold"] == 0.45
    assert info["best_settings"] == {"model__C": 0.1, "model__solver": "lbfgs"}
    manifest = json.loads((tmp_path / "reports/reproduction_run.json").read_text())
    assert manifest["training_evidence_recomputed"]


def test_training_evidence_excludes_reserved_rows(monkeypatch):
    frame = toy_frame()
    x_train, _, _, _ = final.pipeline.split_data(frame)
    seen = []
    compare = train.compare_models
    analyze = final.training_diagnostics.analyze

    def checked_compare(x, y, api, cv):
        assert x.index.equals(x_train.index)
        seen.append("cv")
        return compare(x, y, api, cv)

    def checked_analyze(data):
        assert data.index.equals(x_train.index)
        seen.append("leakage")
        return analyze(data)

    monkeypatch.setattr(train, "compare_models", checked_compare)
    monkeypatch.setattr(final.training_diagnostics, "analyze", checked_analyze)
    final.training_evidence(frame)
    assert seen == ["cv", "leakage"]
