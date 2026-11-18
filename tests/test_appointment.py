"""Tests for appointment validation and model input conversion."""

from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from noshows.appointment import CATEGORY_CHOICES, NUMBER_LIMITS, Appointment
from noshows.pipeline import ALL_FEATURES


@pytest.fixture
def valid_appointment():
    """Return a valid appointment for testing."""
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
    assert valid_appointment.lead_days == 10
    assert valid_appointment.age == 35
    assert valid_appointment.distance_km == 5.5


def test_appointment_is_immutable(valid_appointment):
    with pytest.raises(FrozenInstanceError):
        valid_appointment.age = 40


def test_to_frame_has_one_row(valid_appointment):
    frame = valid_appointment.to_frame()

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (1, len(ALL_FEATURES))


def test_to_frame_feature_order(valid_appointment):
    frame = valid_appointment.to_frame()

    assert list(frame.columns) == list(ALL_FEATURES)


def test_to_frame_preserves_values(valid_appointment):
    frame = valid_appointment.to_frame()

    assert frame.loc[0, "lead_days"] == 10
    assert frame.loc[0, "age"] == 35
    assert frame.loc[0, "booking"] == "online"
    assert frame.loc[0, "treatment"] == "physio"


def test_missing_distance_allowed(valid_appointment):
    updated = replace(valid_appointment, distance_km=None)

    assert updated.distance_km is None
    assert pd.isna(updated.to_frame().loc[0, "distance_km"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("age", 11),
        ("age", 111),
        ("prior_no_shows", -1),
        ("prior_no_shows", 31),
        ("hour", 7),
        ("hour", 18),
        ("first_visit", -1),
        ("first_visit", 2),
        ("distance_km", 0),
        ("distance_km", 151),
    ],
)
def test_numeric_out_of_range_rejected(valid_appointment, field, value):
    with pytest.raises(ValueError, match=f"^{field}"):
        replace(valid_appointment, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", "10"),
        ("age", True),
        ("prior_no_shows", 1.5),
        ("hour", "10"),
        ("first_visit", 0.5),
        ("distance_km", "5.5"),
    ],
)
def test_invalid_numeric_types_rejected(valid_appointment, field, value):
    with pytest.raises(ValueError, match=f"^{field}"):
        replace(valid_appointment, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weekday", "saturday"),
        ("booking", "email"),
        ("insurance", "unknown"),
        ("treatment", "surgery"),
    ],
)
def test_invalid_categories_rejected(valid_appointment, field, value):
    with pytest.raises(ValueError, match=f"^{field}"):
        replace(valid_appointment, **{field: value})


def test_public_constants_for_gradio():
    assert NUMBER_LIMITS["lead_days"] == (0, 120)
    assert NUMBER_LIMITS["age"] == (12, 110)
    assert NUMBER_LIMITS["distance_km"] == (0.1, 150)

    assert CATEGORY_CHOICES["weekday"] == (
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
    )
    assert CATEGORY_CHOICES["booking"] == (
        "online",
        "phone",
        "front_desk",
    )


def test_gradio_whole_number_floats(valid_appointment):
    updated = replace(
        valid_appointment,
        lead_days=12.0,
        age=35.0,
        hour=10.0,
    )

    assert updated.lead_days == 12
    assert updated.age == 35
    assert updated.hour == 10


def test_gradio_decimal_integer_field_rejected(valid_appointment):
    with pytest.raises(ValueError, match="^lead_days"):
        replace(valid_appointment, lead_days=12.5)


def test_gradio_missing_distance_becomes_nan(valid_appointment):
    updated = replace(valid_appointment, distance_km=None)

    assert pd.isna(updated.to_frame().loc[0, "distance_km"])


def test_gradio_missing_required_number_rejected(valid_appointment):
    with pytest.raises(ValueError, match="^lead_days"):
        replace(valid_appointment, lead_days=None)
