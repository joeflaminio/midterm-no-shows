"""Maple Physio no-show check: a Gradio Blocks page that uses only our classes.

Appointment checks the booking and NoShowPredictor turns it into a chance and an
action. The field limits, the category lists and the threshold all come from
those classes and from models/model_info.json, never from numbers typed here.
"""

import gradio as gr

from noshows.appointment import CATEGORY_CHOICES, NUMBER_LIMITS, Appointment
from noshows.predictor import NoShowPredictor

try:  # present only on Hugging Face ZeroGPU hardware
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU
    def keep_zerogpu_happy() -> None:
        """ZeroGPU will not start without one GPU function. This one does nothing."""


PREDICTOR = NoShowPredictor()  # loads the pipeline and model_info.json once

# Page order follows the class lists, so the page and the class cannot disagree.
FIELD_NAMES = list(NUMBER_LIMITS) + list(CATEGORY_CHOICES)

LABELS = {
    "lead_days": "Days between booking and visit",
    "age": "Patient age (years)",
    "prior_no_shows": "Earlier missed appointments",
    "hour": "Start hour (24h clock)",
    "first_visit": "First visit to the clinic?",
    "distance_km": "Distance from home (km), leave blank if unknown",
    "weekday": "Day of the visit",
    "booking": "How it was booked",
    "insurance": "Insurance",
    "treatment": "Treatment",
}

EXAMPLE_BOOKINGS = [
    {  # a low-risk booking
        "lead_days": 3,
        "age": 62,
        "prior_no_shows": 0,
        "hour": 11,
        "first_visit": 0,
        "distance_km": 4.0,
        "weekday": "wed",
        "booking": "front_desk",
        "insurance": "provincial",
        "treatment": "acupuncture",
    },
    {  # a high-risk booking
        "lead_days": 35,
        "age": 24,
        "prior_no_shows": 3,
        "hour": 17,
        "first_visit": 1,
        "distance_km": 22.5,
        "weekday": "mon",
        "booking": "online",
        "insurance": "self_pay",
        "treatment": "physio",
    },
    {  # distance unknown: scored with a warning
        "lead_days": 14,
        "age": 45,
        "prior_no_shows": 1,
        "hour": 9,
        "first_visit": 0,
        "distance_km": None,
        "weekday": "fri",
        "booking": "phone",
        "insurance": "private",
        "treatment": "massage",
    },
]
EXAMPLES = [[booking[name] for name in FIELD_NAMES] for booking in EXAMPLE_BOOKINGS]


def check_booking(*values):
    """Check one booking with Appointment, then score it with NoShowPredictor."""
    booking = dict(zip(FIELD_NAMES, values, strict=True))
    try:
        appointment = Appointment(**booking)
    except ValueError as error:
        raise gr.Error(str(error), title="Please fix this booking") from error

    if appointment.distance_km is None:
        gr.Warning(
            "Distance was left blank. The model used the typical distance from "
            "the training data, so treat this chance as less certain."
        )

    result = PREDICTOR.predict(appointment)
    chances = {"No-show": result.probability, "Comes": 1 - result.probability}
    if result.call:
        action = (
            f"Call the patient the day before. The chance of a no-show "
            f"({result.probability:.0%}) is at or above the clinic's call "
            f"threshold of {PREDICTOR.threshold:.0%}."
        )
    else:
        action = (
            f"No call needed. The chance of a no-show ({result.probability:.0%}) "
            f"is below the clinic's call threshold of {PREDICTOR.threshold:.0%}."
        )
    return chances, action


def make_input(name):
    """Build one input from the class lists: the allowed range goes in the label."""
    if name in CATEGORY_CHOICES:
        return gr.Dropdown(list(CATEGORY_CHOICES[name]), label=LABELS[name])
    if name == "first_visit":
        return gr.Radio([("No", 0), ("Yes", 1)], value=0, label=LABELS[name])
    low, high = NUMBER_LIMITS[name]
    return gr.Number(label=f"{LABELS[name]} [{low} to {high}]")


with gr.Blocks(title="Maple Physio no-show check") as demo:
    gr.Markdown(
        "# Maple Physio no-show check\n"
        "Enter a booking to see the chance that the patient will not come, and "
        "whether the front desk should phone them the day before. A call costs "
        "$8; a no-show nobody phoned about costs $35. The data behind the model "
        "is invented, so treat the result as a prompt, not a verdict."
    )
    with gr.Row():
        with gr.Column():
            inputs = [make_input(name) for name in FIELD_NAMES]
            check_button = gr.Button("Check booking", variant="primary")
        with gr.Column():
            chance_out = gr.Label(label="Chance of a no-show")
            action_out = gr.Textbox(label="What to do", lines=3)
    gr.Examples(EXAMPLES, inputs=inputs, label="Example bookings")
    check_button.click(check_booking, inputs=inputs, outputs=[chance_out, action_out])


if __name__ == "__main__":
    demo.launch()
