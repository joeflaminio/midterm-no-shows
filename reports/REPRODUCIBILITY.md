# Reproducing training-only development evidence

This is not the final training/evaluation command. Activate the existing project
Conda environment (Python 3.11 per environment.yml), then from the repository root:

```bash
git fetch origin main
python -m noshows.reproduce_training --output-dir ../midterm-cv-reproduction --verify-existing
```

Choose a NEW output directory; existing directories are refused without overwrite.
Git must be available. On Windows, if Git is not on PATH, add:
`--git-executable "C:\Program Files\Git\cmd\git.exe"`.

The command reads the corrected supplied generator from commit
`d061c41861006984fbe51760234ab3fc566b4688` using read-only `git show`, verifies its
SHA256, and never changes the checkout, PR base or main. The pin preserves exact provenance; the draft now also includes the corrected
generator from main after Joe's PR2 merged.
Missing commit objects or a hash mismatch stop the run; fetch main as above.
No network download or code installation is performed by the Python command.

It uses Joe's fixed 75/25 split and seed11, discards the holdout values and labels,
compares the four candidates using five-fold stratified training CV, then runs
the F1 grid for the leading candidate. It saves training-only means/std, raw fold
scores, tuning grid, index manifest and version/source metadata. It does not call
prepare_training, evaluate_final or the final CLI, choose a cost threshold, save
a final model, or produce holdout metrics/OOF row probabilities.

With --verify-existing, all five CV metrics and tuning fold scores/parameters must
match the committed development CSVs within atol=1e-12, rtol=1e-10. Timing fields
are intentionally excluded. A version/platform change may cause a mismatch; do
not overwrite the reviewed CSVs merely to pass. Original evidence used Python
3.11.18, NumPy2.4.6, pandas3.1.1, sklearn1.9.1. The reported fold SD uses ddof=0.

Original archived run_initial_cv.py was recovered and SHA256-verified against
reports/training_evidence_metadata.json:
ed82994a26ab901651443c14b369f52357c09c5e787a69cd732469d50ca2b220.
The new module replaces its machine-specific reference path with the pinned
corrected generator. Original evidence and provenance hashes remain unchanged.

## Verification and remaining work
A Windows Conda reproduction matched the committed four-model CV and six-row
logistic tuning grid, without evaluating the final holdout. Dedicated safety
tests check output protection, generator hash validation, training-only return
values and population-standard-deviation reporting.

Local Python3.11 matches environment.yml. README Space settings and deployment
tests specify Python3.12; actual deployment/runtime parity remains unverified.
Joe PR2 has merged and main has been integrated into this draft. Vedant's evaluation functions are integrated from merged PR #4 and tested only
with toy data. Appointment and predictor code remain dependencies. Final threshold, leakage
results, model files, final holdout evaluation and deployment checks remain
pending. Model-selection CV estimates are not unbiased final performance.

For the subsequent training-only threshold/leakage evidence, use
`python -m noshows.training_diagnostics --output-dir ../new-training-diagnostics`.
See reports/training_progress.md for results, source-hash convention and limitations.
This separate safe command supersedes the pending threshold/leakage status above;
it does not change the original CV-only reproduction or its archived evidence.
Final model, final holdout evaluation and deployment remain pending.

Finalization has now been completed once with approved settings. See
reports/final_results.md. The final train entrypoint is guarded against repeats;
training-only reproduction remains safe. Do not remove final_run_lock.json or
final outputs to repeat evaluation without separate authorization.

## Portable frozen final reproduction (current behavior)

The repeat-blocking guard described above has been replaced. From a clean source
copy or existing checkout, in the course Conda environment, run:

```bash
python -m noshows.train
```

It imports the committed supplied generator normally, uses only approved frozen
balanced LR C=0.1/lbfgs and threshold0.45, and regenerates the fitted pipeline,
metadata, metrics and predictions. It does not redo selection. Training CV, OOF cost and leakage evidence are recomputed from training rows
on every invocation; no cached report inputs are needed. Outputs are staged before
replacement; original final evaluation audit records are preserved. No Git
executable or historical Git object is required for this final command.
The older training-only reproduction helper retains its explicit provenance pin.

Verified once in a separate clean source archive with no Git directory and existing
tracked model/metadata/lock. All model bytes and numerical results matched the
original exactly. 49 focused/toy/artifact tests and Ruff passed there. See
frozen_reproduction_verification.json. No second model/threshold selection or
change to original final artifacts occurred. A fresh environment install and
complete app/Space verification remain outside this check.

## Fresh training report regeneration

The current final command does not read training_cv_summary.csv or
training_diagnostics.json as required inputs. It recomputes four-model five-fold
training CV, frozen-model training OOF cost diagnostics and paired leakage folds.
The approved production model/settings/threshold remain fixed, regardless of the
recomputed diagnostic cost minimizer. It writes fresh cv_results.csv, metrics.json,
model artifacts, reproduced_threshold_costs.csv, reproduced_leakage_folds.csv and
reproduction_run.json; original evaluation audit records remain untouched.

44 focused tests passed, including a full toy reproduction with an initially empty
output directory, repeated existing outputs without Git, failure preservation and
training-index isolation. The current real-data training stage exactly matched
recorded CV values, all 19 cost rows and all five paired leakage folds, with final
evaluation and saving functions replaced by raising guards. No extra real holdout
prediction was performed for this change. See training_regeneration_verification.json.
The earlier clean-source real frozen reproduction remains valid evidence for the
unchanged downstream fit/evaluation stage, not a claim of another full real-data
run after this report-generation change. Full app/Space checks remain pending.
