# Final frozen-model evaluation

The approved balanced logistic regression settings were frozen at C=0.1,
solver=lbfgs, class_weight=balanced, seed11 and call threshold0.45 before
examining the holdout. Code was committed as fac1477 before execution.
The whole preprocessing/model pipeline was fitted on all 3,000 training rows.
The 1,000-row holdout was predicted exactly once. No setting was changed afterward.

| Metric | Held-out value |
|---|---:|
| Precision | 0.387629 |
| Recall | 0.717557 |
| F1 | 0.503347 |
| ROC-AUC | 0.728335 |
| Accuracy | 0.629000 |

Confusion counts: TP188, FP297, FN74, TN441. There are485 calls.
At $8 per call plus $35 per uncalled no-show, model cost is **$6,470**,
versus **$9,170** calling nobody and **$8,000** calling everybody.
This is $2,700 below nobody and $1,530 below everybody on this synthetic holdout.
These results do not guarantee real-world savings: data are synthetic, only1,000
appointments were evaluated, and the cost assumption says every call prevents
a missed visit. Accuracy alone conceals the asymmetric costs and class imbalance.

The training-only leakage experiment (not a second holdout experiment) is recorded
in training_leakage_folds.csv: honest AUC0.72745 +/-0.01175 versus leaky
AUC0.92316 +/-0.00640. fee_charged is post-outcome information and is never a
production predictor. CV/model selection evidence remains development evidence.

## Artifacts and verification

- models/no_show.joblib: complete fitted pipeline, not just estimator.
- models/model_info.json: threshold, sklearn version, feature lists, seed,
  frozen settings, training leakage means and test metrics.
- reports/cv_results.csv: previously verified training CV summary, not rerun.
- reports/metrics.json and final_predictions.csv: saved one-time evaluation.
- reports/final_run_lock.json: completed-once marker and binary model hash.
- reports/final_verification.json: independent recorded-prediction arithmetic,
  confusion counts, runtime loader check and artifact hashes. Text artifact hashes
  describe pre-Git working-copy bytes; Git may normalize line endings. Binary
  model hash is unchanged by Git.

Before evaluation:40 focused tests passed using toy data/safety checks and prior
training tests. After evaluation:7 applicable artifact/dependency deployment
checks passed. Ruff0.17.5 check and format passed across21 files. Runtime was
Windows Conda Python3.11.18, sklearn1.9.1, pandas3.1.1, NumPy2.4.6.

Vedant's Appointment and NoShowPredictor at3591bf2 were read directly from Git
without changing his files. The saved pipeline/metadata loaded successfully and
predicted an invented valid appointment with blank distance (probability0.541465,
call=True). This was not a holdout row or a full class/app/deployment pass.

## Reproduction and remaining work

python -m noshows.train now reproduces the approved frozen model and threshold
using a normal import from noshows.data. It does not require Git, a historical
commit or a Windows path. Existing generated outputs are replaced only after a
successful staged computation; each file replacement is atomic, not the entire
multi-file operation. The original final_run_lock.json and final_verification.json
remain audit records and no longer block reproducibility. New invocations record
reports/reproduction_run.json. No model search or threshold reselection occurs.

One additional frozen reproduction was explicitly authorized and completed in a
separate clean source archive with no .git directory and all tracked artifacts
already present. Its model binary was identical; metadata, predictions, metrics
and CV values matched exactly. Original working-copy artifacts remained unchanged.
49 focused/synthetic/artifact tests passed in that archive and Ruff passed.
This supersedes the earlier repeat-blocking guard. See
reports/frozen_reproduction_verification.json. This verifies source portability
and frozen computation, not a newly installed environment or deployed application.
Team application/class integration, full fresh-clone/deployment tests, public Space
and latest-commit reviews remain pending. README still has team-owned draft/TODO
sections; this report is the authoritative new result, superseding earlier reports'
statements that final model and holdout evaluation were pending. No PR merge or
teammate implementation change was performed.

Current report-generation correction: the final command now recomputes training
CV, OOF cost and leakage evidence instead of reading cached training reports.
44 focused tests pass, including full toy execution with no cached reports.
The actual training-only stage matches all previously recorded values exactly;
no further real holdout prediction was done. See
reports/training_regeneration_verification.json for precise verification scope.
