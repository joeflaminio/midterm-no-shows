---
title: Maple Physio No-Show Check
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Maple Physio no-show check

> Development draft: app/class/threshold integration and final model/test results
> remain pending. The run and deployment sections below describe the intended
> completed workflow, not a verified working deployment. For current training-only
> results and safe reproduction, see reports/training_progress.md and
> reports/REPRODUCIBILITY.md.

## What it does

Missed appointments cost Maple Physio money. This page estimates the chance that a
booked patient will not show up and recommends whether the front desk should call
them. A call costs $8; an uncalled no-show costs $35. We assume a call makes the
patient attend.

Public Space: **TODO (Bibhash): https://huggingface.co/spaces/USERNAME/midterm-no-shows**

Demo recording (if recorded): **TODO**

## How to run it locally

```bash
conda env create -f environment.yml
conda activate midterm-no-shows
python -m noshows.train     # recreates models/ and reports/
gradio app.py               # opens the page locally
```

## Team

| Member  | Parts owned |
|---------|-------------|
| Joe     | Data and pipeline: `noshows/data.py` (unchanged), `noshows/pipeline.py`, shared feature lists, stratified split, data/pipeline tests |
| Sahar   | Model comparison and training: `noshows/train.py`, cross-validation, GridSearchCV, leakage experiment, final evaluation, saved model and reports |
| Vedant  | Cost rule and classes: `noshows/evaluation.py`, `noshows/appointment.py`, `noshows/predictor.py`, threshold selection, class tests |
| Bibhash | App and Space: `app.py` (Gradio Blocks), public Hugging Face Space, page checks |
| Sonam   | Integration, tests and README: environment and config files, deployment tests, fresh-clone check, README, demo coordination |

Each member must write code, test their work and review others' pull requests.
At least two reviewers are required per merge; verify contribution/review records
before claiming this requirement is complete.

## Decisions

**Columns used (known when the booking is made):**

| Column | Type | Why it is used |
|--------|------|----------------|
| lead_days | number | TODO (Joe) |
| age | number | TODO (Joe) |
| prior_no_shows | number | TODO (Joe) |
| hour | number | TODO (Joe) |
| first_visit | number (0/1) | TODO (Joe) |
| distance_km | number, may be blank | TODO (Joe); blanks filled by the median imputer |
| weekday | category | TODO (Joe) |
| booking | category | TODO (Joe) |
| insurance | category | TODO (Joe) |
| treatment | category | TODO (Joe) |

**Columns excluded:**

- `appointment_id` — an identifier with no meaning for attendance.
- `fee_charged` — recorded after the visit, so it is not available at booking time
  (leakage; see Result).
- `no_show` — the target the model predicts, never an input.

**Input limits (enforced by `Appointment`, shown by the page):**

| Field | Allowed | Reason |
|-------|---------|--------|
| lead_days | 0 to 120 | TODO (Vedant) |
| age | 12 to 110 | TODO (Vedant) |
| prior_no_shows | 0 to 30 | TODO (Vedant) |
| hour | 8 to 17 | TODO (Vedant) |
| first_visit | 0 or 1 | TODO (Vedant) |
| distance_km | 0.1 to 150, or blank with a warning | TODO (Vedant) |
| weekday | mon, tue, wed, thu, fri | TODO (Vedant) |
| booking | online, phone, front_desk | TODO (Vedant) |
| insurance | private, provincial, self_pay | TODO (Vedant) |
| treatment | physio, massage, acupuncture | TODO (Vedant) |
| Feature Selection (Role 1) | Selected numeric (`lead_days`, `age`, `prior_no_shows`, `hour`, `first_visit`, `distance_km`) and categorical (`weekday`, `booking`, `insurance`, `treatment`) features available at booking time. | TODO (Joe) |
| Excluded Columns (Role 1) | Excluded `appointment_id` (no predictive signal) and `fee_charged` (to prevent data leakage, as fees are only incurred after a missed visit). | TODO (Joe) |
Unknown categories (for example `onlne`) are rejected by `Appointment` with the
field named. The encoder also uses `handle_unknown="ignore"` so it can never crash.
TODO (Vedant): note which numeric fields accept decimals.

Data split: 75% train (3,000 rows) / 25% test (1,000 rows), stratified on `no_show`,
seed 11 (Joe). The final test set has not been evaluated. Reserve its evaluation until
every choice is fixed.

## Model comparison

Five-fold stratified cross-validation on the training data only
(mean ± standard deviation). Source: `reports/cv_results.csv`.

| Model | Precision | Recall | F1 | ROC-AUC | Accuracy |
|-------|-----------|--------|----|---------|----------|
| DummyClassifier | TODO | TODO | TODO | TODO | TODO |
| TODO (Sahar) | TODO | TODO | TODO | TODO | TODO |
| TODO (Sahar, class_weight="balanced") | TODO | TODO | TODO | TODO | TODO |
| TODO (Sahar) | TODO | TODO | TODO | TODO | TODO |

Why accuracy alone misleads: TODO (Sahar).

Tuning: GridSearchCV over TODO parameters (TODO combinations), scored on TODO.
Did tuning beat the fold-to-fold variation? TODO (Sahar).

Selected model: TODO (Sahar) — reason.

Threshold: TODO (Vedant), chosen from 0.05–0.95 using out-of-fold training
probabilities to minimise cost. Stored in `models/model_info.json`; the app reads
it from there.

## Result

Final held-out test set (1,000 appointments). Source: `reports/metrics.json`.

| Metric | Value |
|--------|-------|
| Precision | TODO |
| Recall | TODO |
| F1 | TODO |
| ROC-AUC | TODO |
| Accuracy | TODO |

**Cost on the test set** (Total = $8 × called + $35 × uncalled no-shows):

| Strategy | Cost |
|----------|------|
| Call nobody | $TODO |
| Call everybody | $TODO |
| Use the model | $TODO |

TODO (Sahar): one sentence on whether the model beats both alternatives.

**Leakage check:** honest cross-validated ROC-AUC = TODO; with `fee_charged`
added = TODO. The larger number cannot be trusted because `fee_charged` is only
known after the visit, so it effectively reveals the answer.

## Tests

`tests/` contains TODO test functions (at least 12, with at least 2 parametrized):

- Data and pipeline (Joe): reproducible data, excluded columns absent, stratified
  75/25 split, missing `distance_km` handled.
- Cost and threshold (Vedant): hand-worked cost example, call-nobody and
  call-everybody cheapest cases, threshold at both extremes.
- Classes (Vedant): valid booking and blank distance accepted, each impossible
  number and unseen category rejected with the field named, probability in [0, 1],
  riskier booking scores higher, decision at/below/above threshold, model loaded once.
- Deployment (Sonam, `tests/test_deployment.py`): scikit-learn version matches
  installed / `model_info.json` / `requirements.txt`; gradio and spaces absent from
  `requirements.txt`; README `sdk_version` matches installed Gradio;
  `python_version` is 3.12; `app_file` exists; `models/` not git-ignored.

From a fresh clone with the environment active:

```bash
python -m noshows.train
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Limitations

- The data is synthetic (4,000 generated appointments), so results may not match a
  real clinic.
- The model only sees booking-time information; it cannot know about illness,
  weather, transport problems or anything the patient hasn't told the clinic.
- It gives a probability, not a certainty — staff should treat it as a prompt,
  not a verdict.
- The cost rule assumes every call makes the patient attend, which overstates savings.
- With only 1,000 test appointments, the cost difference has real uncertainty;
  TODO (Sahar): how much to trust the savings.

## Data & Pipeline Decisions (Role 1)

### Feature Selection & Exclusions
* **Target Variable**: `no_show` (binary indicator for missed visits).
* **Selected Features**:
  * **Numeric (`lead_days`, `age`, `prior_no_shows`, `hour`, `first_visit`, `distance_km`)**: Captures scheduling lead time, patient demographics, historical attendance, time of day, and travel distance available at booking.
  * **Categorical (`weekday`, `booking`, `insurance`, `treatment`)**: Captures day-of-week patterns, booking channels, payment structures, and appointment types.
* **Excluded Columns**:
  * `appointment_id`: Arbitrary identifier with no predictive signal.
  * `fee_charged`: Excluded from the normal prediction features to prevent **data leakage**, as fees are only incurred after a missed visit (retained separately for the leakage experiment).

### Data Splitting & Reproducibility
* **Random Seed (`SEED = 11`)**: Used across both the data generator (`make_appointments(n=4000, seed=11)`) and the train/test split (`random_state=11`) to ensure fully deterministic, reproducible splits across the entire team's environment.
* **Split Strategy**: Stratified 75/25 train/test split preserving the original class distribution.