"""Page checks: the Gradio page uses the classes, explains errors and warns."""

import gradio as gr
import pytest

import app
from noshows.appointment import CATEGORY_CHOICES, NUMBER_LIMITS

VALID = {
    "lead_days": 10,
    "age": 40,
    "prior_no_shows": 0,
    "hour": 11,
    "first_visit": 0,
    "distance_km": 5.0,
    "weekday": "wed",
    "booking": "phone",
    "insurance": "private",
    "treatment": "physio",
}


def page_values(**changes):
    """Return the page inputs, in page order, for VALID with some fields changed."""
    booking = {**VALID, **changes}
    return [booking[name] for name in app.FIELD_NAMES]


def test_page_is_built_from_the_class_lists():
    assert isinstance(app.demo, gr.Blocks)
    assert app.FIELD_NAMES == list(NUMBER_LIMITS) + list(CATEGORY_CHOICES)
    for name, component in zip(app.FIELD_NAMES, app.inputs, strict=True):
        if name in CATEGORY_CHOICES:
            choices = [value for _, value in component.choices]
            assert choices == list(CATEGORY_CHOICES[name])


def test_valid_booking_shows_chance_and_action():
    chances, action = app.check_booking(*page_values())
    assert 0 <= chances["No-show"] <= 1
    assert chances["No-show"] + chances["Comes"] == pytest.approx(1)
    assert action.startswith(("Call the patient", "No call needed"))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("lead_days", 500),
        ("age", 5),
        ("prior_no_shows", -1),
        ("hour", 20),
        ("distance_km", 0),
        ("age", None),
        ("booking", "onlne"),
        ("weekday", "sat"),
    ],
)
def test_bad_input_shows_a_clear_error_naming_the_field(field, bad_value):
    with pytest.raises(gr.Error, match=field):
        app.check_booking(*page_values(**{field: bad_value}))


def test_blank_distance_is_scored_with_a_warning():
    with pytest.warns(UserWarning, match="Distance was left blank"):
        chances, _ = app.check_booking(*page_values(distance_km=None))
    assert 0 <= chances["No-show"] <= 1


@pytest.mark.filterwarnings("ignore:Distance was left blank")
def test_action_follows_the_saved_threshold():
    for booking in app.EXAMPLE_BOOKINGS:
        values = [booking[name] for name in app.FIELD_NAMES]
        chances, action = app.check_booking(*values)
        should_call = chances["No-show"] >= app.PREDICTOR.threshold
        assert action.startswith("Call the patient") == should_call
