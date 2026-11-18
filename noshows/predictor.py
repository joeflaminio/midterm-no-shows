"""Load a trained model and predict appointment no-show risk."""

import json
from pathlib import Path

import joblib

from noshows.appointment import Appointment


class NoShowPredictor:
    """Predict no-show probability and recommend whether to call."""

    def __init__(
        self,
        model_path: str | Path = "models/no_show.joblib",
        info_path: str | Path = "models/model_info.json",
    ):
        self.model_path = Path(model_path)
        self.info_path = Path(info_path)

        # Load the trained model once when creating the predictor.
        self.model = joblib.load(self.model_path)

        # Read the threshold from saved training metadata.
        with self.info_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

        self.threshold = float(metadata["threshold"])

        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

    def predict(self, appointment: Appointment) -> tuple[float, bool]:
        """Return (no-show probability, should-call decision)."""
        if not isinstance(appointment, Appointment):
            raise TypeError("appointment must be an Appointment instance")

        frame = appointment.to_frame()
        probability = float(self.model.predict_proba(frame)[0, 1])
        should_call = probability >= self.threshold

        return probability, bool(should_call)

    def __repr__(self) -> str:
        return (
            f"NoShowPredictor(model_path={str(self.model_path)!r}, "
            f"threshold={self.threshold:.2f})"
        )
