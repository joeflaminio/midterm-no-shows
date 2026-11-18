# Training contribution: draft review
This is Sahar's course-aligned training orchestration, not a finished submission.

## Dependencies before merge / final execution
- Joe's merged PR #2 provides `noshows.pipeline`. Reviewed head: 473e58f98651582b1fbdc1fdd08e2fb3e436a992.
- Vedant's evaluation module is available on a separate feature branch but has not yet been integrated or verified here. The final entrypoint currently expects `noshows.evaluation.total_cost(y_true, calls)` and `choose_threshold(y_true, probabilities, thresholds)`. These are provisional integration signatures, not instructor requirements.
- Tests use clearly labelled toy cost/threshold functions, not substitutes presented as Vedant's work.
- Coordinate model metadata/output filenames with prediction and deployment code.
- Do not merge as a finished application or claim generated model/metrics exist.

## Course alignment
Majority-class DummyClassifier baseline; balanced logistic regression, decision tree and random forest candidates; five-fold stratified CV. All five required metrics have means/std. Candidate selection and GridSearchCV use F1, fixed before seeing results. Grids cover at least two parameters and six combinations. Preprocessing remains inside Joe's pipeline. Threshold selection uses training OOF probabilities and the agreed cost rule. Leakage comparison adds fee_charged only for that experiment.

## Test safety
Tests use invented toy rows. No course-generator data or actual final holdout is used.
`python -m noshows.train` is the eventual final-output command and evaluates the holdout: DO NOT run it while developing or deciding models.
Runtime verification used existing Conda Python 3.11, matching environment.yml. README Space settings and deployment tests specify Python 3.12; actual cross-runtime/deployment parity is not yet verified.
The small integration test imports the shared pipeline normally and has no machine-specific paths.

## Training-only reproduction
See [reports/REPRODUCIBILITY.md](reports/REPRODUCIBILITY.md) for the portable, non-overwriting development command and its verified numerical comparison. No final holdout evaluation has been run.
