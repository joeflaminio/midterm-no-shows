"""Tests for appointment validation and model input conversion."""

from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from noshows.appointment import Appointment
from noshows.pipeline import ALL_FEATURES


@pytest.fixture
def valid_appointment():
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


def test_valid_appointment(valid_appointment):
    assert valid_appointment.age == 35
    assert valid_appointment.distance_km == 5.5


def test_dataframe_features(valid_appointment):
    frame = valid_appointment.to_frame()

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (1, len(ALL_FEATURES))
    assert list(frame.columns) == ALL_FEATURES
    assert frame.iloc[0]["lead_days"] == 10


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("age", 11),
        ("age", 111),
        ("prior_no_shows", 31),
        ("hour", 18),
        ("first_visit", 2),
        ("distance_km", 0),
        ("distance_km", 151),
        ("weekday", "sunday"),
        ("booking", "email"),
        ("insurance", "unknown"),
        ("treatment", "dental"),
    ],
)
def test_invalid_values(valid_appointment, field, invalid_value):
    with pytest.raises(ValueError, match=field):
        replace(valid_appointment, **{field: invalid_value})


def test_missing_distance_is_allowed(valid_appointment):
    appointment = replace(valid_appointment, distance_km=None)

    assert appointment.distance_km is None
    assert pd.isna(appointment.to_frame().iloc[0]["distance_km"])


def test_appointment_is_immutable(valid_appointment):
    with pytest.raises(FrozenInstanceError):
        valid_appointment.age = 40


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("age", 35.5),
        ("lead_days", True),
        ("distance_km", float("nan")),
        ("distance_km", float("inf")),
    ],
)
def test_invalid_numeric_types(valid_appointment, field, invalid_value):
    with pytest.raises(ValueError, match=field):
        replace(valid_appointment, **{field: invalid_value})
