# Training contribution: draft review
This is Sahar's course-aligned training orchestration, not a finished submission.

## Dependencies before merge / final execution
- Joe's merged PR #2 provides `noshows.pipeline`. Reviewed head: 473e58f98651582b1fbdc1fdd08e2fb3e436a992.
- Vedant's PR #4 evaluation functions are integrated from main commit `75bc422ff64281ed25cfd97520583973d3e393d0`. `project_interfaces()` binds the actual shared functions for both the eventual final entrypoint and toy integration checks. No real final threshold has been chosen.
- Generic unit fixtures remain toy-only. The Joe-pipeline integration check now uses Vedant's actual evaluation functions with invented toy rows; it does not use the course generator or final holdout.
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

## Evaluation integration verification (2026-11-18)
- Integrated main through merged PR #4 commit `75bc422ff64281ed25cfd97520583973d3e393d0`, preserving Joe's README additions and teammate commit ancestry.
- The final entrypoint and integration test now share `project_interfaces()`, binding Joe's real pipeline and Vedant's real `total_cost`/`choose_threshold` functions.
- Windows Conda Python 3.11.18, NumPy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1 and pytest 9.2.0: 35 focused tests passed (`test_train_draft`, `test_joe_integration_local`, `test_reproduce_training`, `test_evaluation`, `test_evaluation_integration`).
- Ruff 0.17.5 from an existing Conda environment: repository lint and format checks passed. No packages were installed or changed.
- Tests used toy rows/hand-worked values only. They check the actual evaluation contract, lower-threshold ties and the >= call boundary. No actual final holdout was evaluated and no real final threshold/model was selected or saved.
- This is not a full deployment/fresh-clone pass. Pipeline tests involving the course split and deployment tests requiring final artifacts were not run in this step. Class/predictor integration, final outputs and Python 3.12 deployment parity remain pending.
- Existing PR approvals predate these changes; the updated integration needs team review before merge.
