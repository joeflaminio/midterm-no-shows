"""Tests for loading the model and predicting no-show risk."""

import json
from dataclasses import replace

import joblib
import numpy as np
import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor


class FakeModel:
    """Simple test model with a configurable probability."""

    def __init__(self, probability=0.5):
        self.probability = probability

    def predict_proba(self, frame):
        probability = self.probability
        return np.array([[1 - probability, probability]])


@pytest.fixture
def appointment():
    return Appointment(
        lead_days=10,
        age=35,
        prior_no_shows=1,
        hour=10,
        first_visit=0,
        distance_km=5.5,
        weekday="mon",
        booking="online",
        insurance="private",
        treatment="physio",
    )


@pytest.fixture
def predictor_files(tmp_path):
    model_path = tmp_path / "no_show.joblib"
    info_path = tmp_path / "model_info.json"

    joblib.dump(FakeModel(), model_path)
    info_path.write_text(
        json.dumps({"threshold": 0.5}),
        encoding="utf-8",
    )

    return model_path, info_path


def test_predictor_loads_model_and_metadata(predictor_files):
    predictor = NoShowPredictor(*predictor_files)

    assert predictor.threshold == 0.5
    assert isinstance(predictor.model, FakeModel)


@pytest.mark.parametrize(
    ("probability", "expected_call"),
    [
        (0.49, False),
        (0.50, True),
        (0.51, True),
        (0.90, True),
        (0.10, False),
    ],
)
def test_prediction_threshold(predictor_files, appointment, probability, expected_call):
    predictor = NoShowPredictor(*predictor_files)
    predictor.model = FakeModel(probability)

    result_probability, should_call = predictor.predict(appointment)

    assert result_probability == pytest.approx(probability)
    assert should_call is expected_call


def test_threshold_comes_from_metadata(predictor_files, appointment):
    model_path, info_path = predictor_files

    info_path.write_text(
        json.dumps({"threshold": 0.7}),
        encoding="utf-8",
    )

    predictor = NoShowPredictor(model_path, info_path)
    predictor.model = FakeModel(0.6)

    probability, should_call = predictor.predict(appointment)

    assert probability == pytest.approx(0.6)
    assert should_call is False


def test_model_loaded_only_once(predictor_files, appointment, monkeypatch):
    original_load = joblib.load
    load_count = 0

    def counted_load(path):
        nonlocal load_count
        load_count += 1
        return original_load(path)

    monkeypatch.setattr(joblib, "load", counted_load)

    predictor = NoShowPredictor(*predictor_files)
    predictor.predict(appointment)
    predictor.predict(appointment)

    assert load_count == 1


@pytest.mark.parametrize("invalid_threshold", [-0.1, 1.1])
def test_invalid_threshold_rejected(predictor_files, invalid_threshold):
    model_path, info_path = predictor_files
    info_path.write_text(
        json.dumps({"threshold": invalid_threshold}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="threshold"):
        NoShowPredictor(model_path, info_path)


def test_invalid_appointment_rejected(predictor_files):
    predictor = NoShowPredictor(*predictor_files)

    with pytest.raises(TypeError, match="Appointment"):
        predictor.predict({"age": 35})


def test_missing_distance_prediction(predictor_files, appointment):
    predictor = NoShowPredictor(*predictor_files)
    missing_distance = replace(appointment, distance_km=None)

    probability, should_call = predictor.predict(missing_distance)

    assert probability == pytest.approx(0.5)
    assert should_call is True


def test_predictor_repr(predictor_files):
    predictor = NoShowPredictor(*predictor_files)

    assert "NoShowPredictor" in repr(predictor)
    assert "threshold=0.50" in repr(predictor)


def test_prediction_has_named_fields(predictor_files, appointment):
    """Gradio can access prediction.probability and prediction.call."""
    from noshows.predictor import Prediction

    predictor = NoShowPredictor(*predictor_files)
    result = predictor.predict(appointment)

    assert isinstance(result, Prediction)
    assert result.probability == pytest.approx(0.5)
    assert result.call is True


def test_default_model_paths_are_project_relative():
    """Default paths must work regardless of the current directory."""
    from noshows.predictor import PROJECT_ROOT

    assert NoShowPredictor.__init__.__defaults__ == (None, None)

    assert (PROJECT_ROOT / "models" / "no_show.joblib").is_absolute()
    assert (PROJECT_ROOT / "models" / "model_info.json").is_absolute()
