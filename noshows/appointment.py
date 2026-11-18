"""Validated, immutable appointment data for no-show prediction."""

from dataclasses import dataclass
from numbers import Real

import pandas as pd

from noshows.pipeline import ALL_FEATURES

NUMBER_LIMITS = {
    "lead_days": (0, 120),
    "age": (12, 110),
    "prior_no_shows": (0, 30),
    "hour": (8, 17),
    "first_visit": (0, 1),
    "distance_km": (0.1, 150),
}

CATEGORY_CHOICES = {
    "weekday": ("mon", "tue", "wed", "thu", "fri"),
    "booking": ("online", "phone", "front_desk"),
    "insurance": ("private", "provincial", "self_pay"),
    "treatment": ("physio", "massage", "acupuncture"),
}


@dataclass(frozen=True)
class Appointment:
    """Validated appointment information for model predictions."""

    lead_days: int
    age: int
    prior_no_shows: int
    hour: int
    first_visit: int
    distance_km: float | None
    weekday: str
    booking: str
    insurance: str
    treatment: str

    def __post_init__(self) -> None:
        for field, (minimum, maximum) in NUMBER_LIMITS.items():
            value = getattr(self, field)

            if field == "distance_km" and value is None:
                continue

            if field == "distance_km":
                valid = (
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and minimum <= value <= maximum
                )
            else:
                valid = (
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and minimum <= value <= maximum
                    and float(value).is_integer()
                )

            if not valid:
                raise ValueError(
                    f"{field} must be from {minimum} to {maximum} (got {value!r})"
                )

            if field != "distance_km":
                object.__setattr__(self, field, int(value))

        for field, choices in CATEGORY_CHOICES.items():
            value = getattr(self, field)

            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"{field} must be one of {choices} (got {value!r})")

    def to_frame(self) -> pd.DataFrame:
        """Return a single-row DataFrame in training feature order."""
        values = {
            "lead_days": self.lead_days,
            "age": self.age,
            "prior_no_shows": self.prior_no_shows,
            "hour": self.hour,
            "first_visit": self.first_visit,
            "distance_km": self.distance_km,
            "weekday": self.weekday,
            "booking": self.booking,
            "insurance": self.insurance,
            "treatment": self.treatment,
        }

        return pd.DataFrame([values], columns=ALL_FEATURES)
