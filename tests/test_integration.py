"""Integration checks: the saved model, the classes and the app agree."""

import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor

SAFE = {
    "lead_days": 2,
    "age": 65,
    "prior_no_shows": 0,
    "hour": 11,
    "first_visit": 0,
    "distance_km": 3.0,
    "weekday": "wed",
    "booking": "front_desk",
    "insurance": "provincial",
    "treatment": "acupuncture",
}
RISKY = {
    "lead_days": 45,
    "age": 22,
    "prior_no_shows": 4,
    "hour": 17,
    "first_visit": 1,
    "distance_km": 40.0,
    "weekday": "mon",
    "booking": "online",
    "insurance": "self_pay",
    "treatment": "physio",
}


@pytest.fixture(scope="module")
def predictor():
    return NoShowPredictor()


def test_riskier_booking_scores_higher(predictor):
    safe = predictor.predict(Appointment(**SAFE))
    risky = predictor.predict(Appointment(**RISKY))
    assert risky.probability > safe.probability
    assert risky.call and not safe.call


@pytest.mark.parametrize(
    ("field", "safer", "riskier"),
    [
        ("prior_no_shows", 0, 5),
        ("lead_days", 1, 60),
        ("booking", "front_desk", "online"),
        ("insurance", "provincial", "self_pay"),
    ],
)
def test_one_risk_factor_raises_the_score(predictor, field, safer, riskier):
    low = predictor.predict(Appointment(**{**SAFE, field: safer}))
    high = predictor.predict(Appointment(**{**SAFE, field: riskier}))
    assert high.probability > low.probability
