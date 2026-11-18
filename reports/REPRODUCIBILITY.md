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
Joe PR2 has merged and main has been integrated into this draft. Vedant's evaluation,
Appointment and predictor code remain dependencies. Final threshold, leakage
results, model files, final holdout evaluation and deployment checks remain
pending. Model-selection CV estimates are not unbiased final performance.
