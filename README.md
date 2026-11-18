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

## What it does

Missed appointments cost Maple Physio money. This page estimates the chance that a
booked patient will not show up and recommends whether the front desk should call
them. A call costs $8; an uncalled no-show costs $35. We assume a call makes the
patient attend.

Public Space: https://huggingface.co/spaces/bibhashdas/midterm-no-shows

Enter a booking and the page shows the chance of a no-show and a plain action
(call or no call). Invalid entries get a message naming the field to fix; a blank
distance is accepted with a warning.

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
| Sonam   | Integration, tests and README: environment and config files, deployment tests, integration tests, integration reviews, fresh-clone check, README, demo coordination |

Every member committed from their own account. Most changes reached `main` through
pull requests reviewed by two members. PR #1 (restoring the supplied age line in
`data.py`) had one review before merging and was reviewed by a second member after
the merge. A few small README edits were committed directly to `main`. The pull
request pages on GitHub are the record of who reviewed what.

## Decisions

**Columns used** (all known when the booking is made). Seed 11 is used for the
generator and the split, so every member gets the same rows.

| Column | Type | Why it is used |
|--------|------|----------------|
| lead_days | number | Bookings made far ahead are easier to forget or replace |
| age | number | Attendance patterns differ by age |
| prior_no_shows | number | Past missed visits are a strong sign of future ones |
| hour | number | Very early and late slots are harder to keep |
| first_visit | number (0/1) | New patients have no attendance habit with the clinic yet |
| distance_km | number, may be blank | Longer trips make attending harder; blanks are filled with the training median |
| weekday | category | Mondays and Fridays behave differently from midweek |
| booking | category | Online, phone and front-desk bookings show different commitment |
| insurance | category | Who pays affects how costly missing is for the patient |
| treatment | category | Different treatments have different attendance patterns |

Numbers are scaled after the median imputer; categories are one-hot encoded with
`handle_unknown="ignore"`. All preprocessing is inside one pipeline, so each
cross-validation fold is prepared from its own training part only.

**Columns excluded:**

- `appointment_id`: an identifier with no meaning for attendance.
- `fee_charged`: recorded after the visit, so it is not known at booking time.
  Using it is leakage (see Result); it appears only in the leakage experiment.
- `no_show`: the target the model predicts, never an input.

**Input limits** (enforced by `Appointment`; the page builds its fields from the
same lists, so the two cannot disagree):

| Field | Allowed | Reason |
|-------|---------|--------|
| lead_days | 0 to 120 | Same-day up to about four months ahead |
| age | 12 to 110 | Youngest and oldest patients the clinic would plausibly see |
| prior_no_shows | 0 to 30 | A count; anything higher is almost certainly a typo |
| hour | 8 to 17 | Clinic opening hours (last start at 17:00) |
| first_visit | 0 or 1 | Yes or no |
| distance_km | 0.1 to 150, or blank with a warning | Local travel range; blank allowed because it is often unknown |
| weekday | mon, tue, wed, thu, fri | Clinic is open weekdays |
| booking | online, phone, front_desk | The three booking channels |
| insurance | private, provincial, self_pay | The three payment types |
| treatment | physio, massage, acupuncture | The three services |

Only `distance_km` accepts decimals (for example 7.5); the other numbers must be
whole numbers. Unknown categories such as `onlne` are rejected with the field
named, rather than silently ignored by the encoder.

Data split: 75% train (3,000 rows) / 25% test (1,000 rows), stratified on
`no_show`, seed 11. The test set was scored once, after every choice was frozen.

## Model comparison

Five-fold stratified cross-validation on the 3,000 training rows only (seed 11),
mean ± standard deviation across folds. Source: `reports/cv_results.csv`.

| Model | Precision | Recall | F1 | ROC-AUC | Accuracy |
|-------|-----------|--------|----|---------|----------|
| DummyClassifier (most frequent) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.738 ± 0.000 |
| Logistic regression (class_weight="balanced") | 0.422 ± 0.011 | 0.660 ± 0.031 | 0.515 ± 0.015 | 0.727 ± 0.012 | 0.675 ± 0.009 |
| Decision tree | 0.344 ± 0.023 | 0.387 ± 0.018 | 0.364 ± 0.015 | 0.562 ± 0.014 | 0.645 ± 0.021 |
| Random forest (100 trees) | 0.564 ± 0.051 | 0.206 ± 0.031 | 0.302 ± 0.040 | 0.696 ± 0.008 | 0.751 ± 0.010 |

**Why accuracy alone misleads:** about 26% of appointments are no-shows, so the
baseline that predicts "attends" for everyone reaches 73.8% accuracy while finding
zero no-shows. The random forest has the highest accuracy (75.1%) but finds only
about one in five no-shows. Missing a no-show is the expensive mistake here, so
we compared models on F1, chosen before seeing results.

**Tuning:** GridSearchCV on balanced logistic regression over two settings and six
combinations: `C` in {0.1, 1, 10} and `solver` in {liblinear, lbfgs}, scored on
F1. The best combination (`C=0.1`, `lbfgs`) reached F1 0.517 ± 0.013, a gain of
0.002 over the untuned model. That is much smaller than the fold-to-fold standard
deviation (0.015), so tuning did not meaningfully improve the model.

**Selected model:** balanced logistic regression (`C=0.1`, `lbfgs`). It had the best
F1 and the best ROC-AUC, finds about two thirds of no-shows, and was stable across
folds.

**Threshold: 0.45.** We tried every threshold from 0.05 to 0.95 in steps of 0.05 on
out-of-fold training probabilities and kept the cheapest. On the training data it
cost $18,921, against $27,475 for calling nobody and $24,000 for calling everybody
(`reports/training_threshold_costs.csv`). It is below 0.5 because a missed visit
($35) costs much more than a call ($8). The threshold is stored in
`models/model_info.json`, and the app reads it from there.

## Result

Final held-out test set (1,000 appointments), scored once with the frozen model
and threshold. Source: `reports/metrics.json`. The settings were frozen before the
test set was used. Later runs of `python -m noshows.train` (to check the command
works from a fresh clone) reuse the same frozen model settings and threshold with
no retuning, and reproduce exactly the same results.

| Metric | Value |
|--------|-------|
| Precision | 0.388 |
| Recall | 0.718 |
| F1 | 0.503 |
| ROC-AUC | 0.728 |
| Accuracy | 0.629 |

The model calls 485 patients and catches 188 of the 262 no-shows (74 missed).

**Cost on the test set** (total = $8 × called + $35 × uncalled no-shows):

| Strategy | Cost |
|----------|------|
| Call nobody | $9,170 |
| Call everybody | $8,000 |
| **Use the model** | **$6,470** |

The model beats both alternatives: $2,700 cheaper than calling nobody and $1,530
cheaper than calling everybody, so it is worth shipping under this cost rule.

**Leakage check:** honest cross-validated ROC-AUC = 0.727; with `fee_charged`
added = 0.923. The larger number cannot be trusted because `fee_charged` is only
known after the visit, so it effectively gives away the answer.

## Tests

`tests/` contains 79 test functions, 12 of them parametrized, and the full suite
passes (124 test cases):

- Data and pipeline (Joe): reproducible data, excluded columns absent, stratified
  75/25 split, missing `distance_km` handled.
- Cost and threshold (Vedant): hand-worked cost example, call-nobody and
  call-everybody cheapest cases, threshold at both extremes.
- Classes (Vedant): valid booking and blank distance accepted, each impossible
  number and unseen category rejected with the field named, probability in [0, 1],
  decision at/below/above threshold, model loaded once (monkeypatched
  `joblib.load`).
- Training (Sahar): model comparison, tuning, frozen settings and reproducible
  training outputs.
- Page (Bibhash, `tests/test_app.py`): field errors, blank-distance warning, and the
  displayed probability and action.
- Deployment (Sonam, `tests/test_deployment.py`): scikit-learn version matches
  installed / `model_info.json` / `requirements.txt`; gradio and spaces absent from
  `requirements.txt`; README `sdk_version` matches installed Gradio;
  `python_version` is 3.12; `app_file` exists; `models/` not git-ignored.
- Integration (Sonam, `tests/test_integration.py`): with the real saved model, a
  clearly riskier booking scores higher than a safer one and gets a call while the
  safer one does not; raising any single risk factor (prior no-shows, lead time,
  online booking, self-pay) raises the score.

From a fresh clone with the environment active:

```bash
python -m noshows.train
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Limitations

- The model only sees booking-time information. It cannot know about illness,
  weather, transport problems or anything the patient hasn't told the clinic.
- It gives a probability, not a certainty: staff should treat it as a prompt,
  not a verdict. It is wrong in both directions (297 unneeded calls and 74 missed
  no-shows on the test set).
- The data is synthetic (4,000 generated appointments), so results may not match a
  real clinic.
- The cost rule assumes every call makes the patient attend, which overstates the
  savings.
- The page accepts some values the model never saw in training (for example ages
  12–15 or 93–110, lead times over 60 days), so predictions there are less reliable.
- **How much to trust the savings:** with only 1,000 test appointments the exact
  figures would shift on another sample. Resampling the test predictions (5,000
  bootstrap samples) gives savings of roughly $1,950–$3,480 against calling nobody
  and $950–$2,080 against calling everybody (95% ranges). Both ranges stay above
  zero, so the model being cheaper is fairly solid, but the exact dollar amount is
  not.
