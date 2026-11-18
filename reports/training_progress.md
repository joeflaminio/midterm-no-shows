### Training-only model comparison and tuning

Using the corrected supplied generator and Joe’s shared pipeline, I compared a majority-class baseline, balanced logistic regression, decision tree and random forest. I used only the 3,000-row training partition, with five-fold stratified cross-validation and seed 11. Preprocessing stayed inside the pipeline. F1 was selected before comparing results.

Mean F1 ± fold standard deviation was:
- Majority baseline: **0.0000 ± 0.0000**
- Balanced logistic regression: **0.5150 ± 0.0151**
- Decision tree: **0.3637 ± 0.0153**
- Random forest: **0.3017 ± 0.0401**

Balanced logistic regression led on F1. I tuned two parameters across six combinations: `C` = 0.1, 1 or 10, and solver = `liblinear` or `lbfgs`. The best combination, `C=0.1` and `lbfgs`, achieved **0.5168 ± 0.0133** F1. The gain of **0.0018** is small relative to fold variation and does not establish a meaningful improvement.

I saved 3,000 training-only out-of-fold probabilities for later threshold selection. No cost threshold has been chosen, and the final holdout has not been evaluated. These are development results, not final performance claims.

## Training-only threshold and leakage follow-up

Using the previously reviewed balanced logistic regression settings (C=0.1,
solver=lbfgs), five-fold stratified OOF probabilities on the same 3,000 training
rows select threshold **0.45** from 0.05 to 0.95 in 0.05 steps. Calls use >=;
cost ties choose the lower threshold, using Vedant's existing functions unchanged.
Training OOF costs: **18,921** for the selected rule, **27,475** calling nobody,
and **24,000** calling everybody. These are training selection estimates, NOT
final holdout costs or promised savings. Model/settings were already selected
using these training folds, so these are not unbiased performance estimates.

The leakage diagnostic changes only the feature set by adding fee_charged to
Joe's numeric pipeline. With identical folds and estimator settings, honest
ROC-AUC is **0.72745 +/- 0.01175**; leaky ROC-AUC is **0.92316 +/- 0.00640**
(population fold SD). The inflated leaky score is not deployable: that field is
only known after the visit. Production features remain unchanged.

Evidence: training_threshold_costs.csv (all 19 thresholds),
training_leakage_folds.csv (all five paired folds), training_diagnostics.json
(source hashes, versions and scope). Source hashes normalize UTF-8 text to LF.
Generator bytes are separately checked against the pinned original hash.
No final fitted model was saved and the reserved holdout was not evaluated.
The above supersedes the earlier statement that threshold selection was pending.

Reproduce in the existing Conda environment, choosing a NEW output directory:

```bash
python -m noshows.training_diagnostics --output-dir ../new-training-diagnostics
```

If needed on Windows, add --git-executable "C:\Program Files\Git\cmd\git.exe".
This command does not call the final training CLI, evaluate_final or save_outputs.
It fits fold-specific models only and writes aggregate development evidence.
Independent arithmetic checks every cost and tie choice; a toy test compares
OOF costs against sklearn.cross_val_predict. Tests forbid final evaluation/output
and check training-row isolation plus refusal to overwrite existing output.

## Final authorized run completed

See reports/final_results.md for the frozen-settings, once-only holdout evaluation,
saved artifacts, verified metrics and limitations. It supersedes this report's
historical statements that final model/evaluation remained pending. Training-only
reproduction commands and development results above remain unchanged.

Portable frozen reproduction is now verified separately; the guard no longer
blocks the required command on fresh source copies. See reports/final_results.md
and reports/frozen_reproduction_verification.json. The original holdout metrics
and artifacts did not change, and no model/threshold was reselected.
