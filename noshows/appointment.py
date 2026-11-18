"""Validated, immutable appointment data for no-show prediction."""

from dataclasses import dataclass
from numbers import Integral, Real

import pandas as pd


@dataclass(frozen=True)
class Appointment:
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

    def __post_init__(self):
        integer_ranges = {
            "lead_days": (0, 120),
            "age": (12, 110),
            "prior_no_shows": (0, 30),
            "hour": (8, 17),
            "first_visit": (0, 1),
        }

        for field, (minimum, maximum) in integer_ranges.items():
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{field} must be an integer between {minimum} and {maximum}"
                )

        distance = self.distance_km
        if distance is not None:
            if (
                isinstance(distance, bool)
                or not isinstance(distance, Real)
                or not 0.1 <= distance <= 150
            ):
                raise ValueError("distance_km must be between 0.1 and 150, or None")

        allowed_categories = {
            "weekday": {"mon", "tue", "wed", "thu", "fri"},
            "booking": {"online", "phone", "front_desk"},
            "insurance": {"private", "provincial", "self_pay"},
            "treatment": {"physio", "massage", "acupuncture"},
        }

        for field, allowed in allowed_categories.items():
            if getattr(self, field) not in allowed:
                raise ValueError(
                    f"{field} must be one of: {', '.join(sorted(allowed))}"
                )

    def to_frame(self) -> pd.DataFrame:
        """Return one row with the exact model feature columns."""
        return pd.DataFrame(
            [
                {
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
            ]
        )
