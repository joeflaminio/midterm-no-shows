"""Draft tests; fixtures are NOT team data, production modules or final metrics."""

import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from noshows.train import (
    Interfaces,
    candidates,
    evaluate_final,
    prepare_training,
    restore_split_frames,
    save_outputs,
    tuning_grid,
)


@pytest.fixture
def fixture_api():
    def build(model, numeric_features=None):
        numbers = ["x"] if numeric_features is None else numeric_features
        return Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            (
                                "num",
                                Pipeline(
                                    [
                                        ("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                numbers,
                            ),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), ["kind"]),
                        ]
                    ),
                ),
                ("model", model),
            ]
        )

    def cost(y, calls):
        return 8 * np.asarray(calls).sum() + 35 * np.sum(
            (np.asarray(y) == 1) & ~np.asarray(calls, dtype=bool)
        )

    def threshold(y, probabilities, grid):
        # Fixture only: production implementation belongs to Vedant.
        return min(grid, key=lambda t: cost(y, probabilities >= t))

    return Interfaces(11, ("x",), ("kind",), build, cost, threshold)


@pytest.fixture
def fixture_frame():
    rng = np.random.default_rng(21)
    x = rng.normal(size=160)
    y = (rng.random(160) < 1 / (1 + np.exp(-2 * x))).astype(int)
    x[::17] = np.nan
    return pd.DataFrame(
        {
            "x": x,
            "kind": np.where(np.arange(160) % 2, "a", "b"),
            "no_show": y,
            "fee_charged": y,
            "appointment_id": np.arange(160),
        }
    )


def test_candidates_include_baseline_and_three_models():
    models = candidates(11)
    assert len(models) == 4
    assert "dummy" in models
    assert models["dummy"].strategy == "most_frequent"
    assert any(
        m.get_params().get("class_weight") == "balanced" for m in models.values()
    )


@pytest.mark.parametrize(
    "name", ["logistic_balanced", "decision_tree", "random_forest"]
)
def test_grids_meet_minimum(name):
    grid = tuning_grid(name)
    assert len(grid) >= 2
    assert len(list(ParameterGrid(grid))) >= 6


@pytest.mark.parametrize("bad", ["appointment_id", "fee_charged", "no_show"])
def test_rejects_excluded_production_features(bad, fixture_api, fixture_frame):
    bad_api = replace(fixture_api, numeric_features=("x", bad))
    with pytest.raises(ValueError, match="excluded"):
        prepare_training(fixture_frame, bad_api)


def test_training_smoke_and_artifact_roundtrip(fixture_api, fixture_frame, tmp_path):
    # All rows are test fixtures; no professor holdout is accessed here.
    model, comparison, details = prepare_training(fixture_frame, fixture_api)
    assert len(comparison) == 4
    for metric in ["precision", "recall", "f1", "roc_auc", "accuracy"]:
        assert f"{metric}_mean" in comparison
        assert f"{metric}_std" in comparison
    assert 0.05 <= details["threshold"] <= 0.95
    assert details["selection_scoring"] == "f1"
    expected = (
        comparison[comparison["model"] != "dummy"]
        .sort_values(["f1_mean", "model"], ascending=[False, True])
        .iloc[0]
    )
    assert details["selected_model"] == expected["model"]
    assert details["untuned_cv_mean"] == expected["f1_mean"]
    assert "fee_charged" not in model.feature_names_in_
    metrics = evaluate_final(model, fixture_frame, details["threshold"], fixture_api)
    save_outputs(tmp_path, model, comparison, details, metrics, fixture_api)
    info = json.loads((tmp_path / "models/model_info.json").read_text())
    assert info["threshold"] == details["threshold"]
    assert info["numeric_features"] == ["x"]
    restored = joblib.load(tmp_path / "models/no_show.joblib")
    np.testing.assert_allclose(
        model.predict_proba(fixture_frame[fixture_api.features]),
        restored.predict_proba(fixture_frame[fixture_api.features]),
    )
    assert (tmp_path / "reports/cv_results.csv").is_file()
    assert (tmp_path / "reports/metrics.json").is_file()


def test_final_threshold_boundary_and_costs(fixture_api):
    class ProbabilityFixture:
        def predict_proba(self, frame):
            p = np.array([0.49, 0.50, 0.51])
            return np.column_stack([1 - p, p])

    frame = pd.DataFrame({"x": [0, 0, 0], "kind": ["a"] * 3, "no_show": [0, 1, 1]})
    metrics = evaluate_final(ProbabilityFixture(), frame, 0.5, fixture_api)
    assert metrics["costs"] == {
        "call_nobody": 70.0,
        "call_everybody": 24.0,
        "model": 16.0,
    }
    assert metrics["accuracy"] == 1.0
    assert metrics["beats_both_alternatives"]


def test_four_output_split_preserves_rows_and_leakage_column(fixture_frame):
    train = fixture_frame.iloc[::2]
    test = fixture_frame.iloc[1::2]
    split = (
        train[["x", "kind"]],
        test[["x", "kind"]],
        train["no_show"],
        test["no_show"],
    )
    restored_train, restored_test = restore_split_frames(fixture_frame, split)
    pd.testing.assert_frame_equal(restored_train, train)
    pd.testing.assert_frame_equal(restored_test, test)


def test_four_output_split_rejects_reset_indices(fixture_frame):
    train = fixture_frame.iloc[::2].reset_index(drop=True)
    test = fixture_frame.iloc[1::2].reset_index(drop=True)
    with pytest.raises(ValueError, match="overlap"):
        restore_split_frames(
            fixture_frame,
            (
                train[["x", "kind"]],
                test[["x", "kind"]],
                train["no_show"],
                test["no_show"],
            ),
        )
