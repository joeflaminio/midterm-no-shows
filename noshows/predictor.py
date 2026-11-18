"""Load a trained model and predict appointment no-show risk."""

import json
from pathlib import Path
from typing import NamedTuple

import joblib

from noshows.appointment import Appointment

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Prediction(NamedTuple):
    """A predicted no-show probability and call recommendation."""

    probability: float
    call: bool


class NoShowPredictor:
    """Predict no-show probability and recommend whether to call."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        info_path: str | Path | None = None,
    ):
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else PROJECT_ROOT / "models" / "no_show.joblib"
        )
        self.info_path = (
            Path(info_path)
            if info_path is not None
            else PROJECT_ROOT / "models" / "model_info.json"
        )

        self.model = joblib.load(self.model_path)

        with self.info_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

        self.threshold = float(metadata["threshold"])

        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

    def predict(self, appointment: Appointment) -> Prediction:
        """Return the probability and threshold-based call decision."""
        if not isinstance(appointment, Appointment):
            raise TypeError("appointment must be an Appointment instance")

        frame = appointment.to_frame()
        probability = float(self.model.predict_proba(frame)[0, 1])
        should_call = probability >= self.threshold

        return Prediction(probability=probability, call=bool(should_call))

    def __repr__(self) -> str:
        return (
            f"NoShowPredictor(model_path={str(self.model_path)!r}, "
            f"threshold={self.threshold:.2f})"
        )
