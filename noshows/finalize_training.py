"""Reproduce frozen final settings without selecting anything on the holdout."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from noshows import pipeline, train, training_diagnostics
from noshows.data import make_appointments

FROZEN = {
    "model": "logistic_balanced",
    "C": 0.1,
    "solver": "lbfgs",
    "threshold": 0.45,
    "seed": 11,
}


def finalize(root, frame, comparison, diagnostics):
    root = Path(root)
    for name in [
        "models/no_show.joblib",
        "models/model_info.json",
        "reports/metrics.json",
        "reports/final_run_lock.json",
    ]:
        if (root / name).exists():
            raise FileExistsError(
                f"Final output/lock exists: {name}; do not reevaluate"
            )
    if diagnostics["threshold"] != FROZEN["threshold"] or diagnostics["settings"] != {
        "C": 0.1,
        "solver": "lbfgs",
        "class_weight": "balanced",
    }:
        raise ValueError("Training evidence differs from approved frozen settings")
    api = train.project_interfaces()
    training, holdout = train.restore_split_frames(frame, pipeline.split_data(frame))
    model = pipeline.build_pipeline(
        train.candidates(11)["logistic_balanced"].set_params(C=0.1, solver="lbfgs")
    )
    model.fit(training[api.features], training.no_show)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    lock = {
        "frozen_settings": FROZEN,
        "training_rows": len(training),
        "test_rows": len(holdout),
        "status": "evaluation_started_do_not_repeat",
    }
    with (root / "reports/final_run_lock.json").open("x", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
    # Persist the fitted pipeline before the one authorized evaluation.
    joblib.dump(model, root / "models/no_show.joblib")

    class Recorder:
        def predict_proba(self, x):
            probabilities = model.predict_proba(x)
            pd.DataFrame(
                {
                    "row_index": x.index,
                    "no_show": holdout.loc[x.index, "no_show"].to_numpy(),
                    "probability": probabilities[:, 1],
                }
            ).to_csv(root / "reports/final_predictions.csv", index=False)
            return probabilities

    metrics = train.evaluate_final(Recorder(), holdout, FROZEN["threshold"], api)
    (root / "reports/metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    details = {
        "selected_model": "logistic_balanced",
        "selection_scoring": "f1",
        "best_settings": {"model__C": 0.1, "model__solver": "lbfgs"},
        "threshold": 0.45,
        "honest_cv_roc_auc": diagnostics["honest_auc_mean"],
        "leaky_cv_roc_auc": diagnostics["leaky_auc_mean"],
        "settings_frozen_before_holdout": True,
    }
    train.save_outputs(root, model, comparison, details, metrics, api)
    lock.update(
        {
            "status": "completed_once",
            "model_sha256": hashlib.sha256(
                (root / "models/no_show.joblib").read_bytes()
            ).hexdigest(),
        }
    )
    (root / "reports/final_run_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def training_evidence(frame):
    """Recompute reports from training rows; never revise approved decisions."""
    x_train, x_test, y_train, y_test = pipeline.split_data(frame)
    training = frame.loc[x_train.index].copy()
    del x_test, y_test
    cv = StratifiedKFold(5, shuffle=True, random_state=FROZEN["seed"])
    comparison = train.compare_models(x_train, y_train, train.project_interfaces(), cv)
    costs, folds, diagnostics = training_diagnostics.analyze(training)
    # Recomputed training diagnostic minimum is evidence only, never a new choice.
    diagnostics["diagnostic_cost_minimizer"] = diagnostics["threshold"]
    diagnostics["threshold"] = FROZEN["threshold"]
    diagnostics["selection_frozen"] = True
    return comparison, diagnostics, costs, folds


def reproduce(root):
    """Regenerate outputs in staging; preserve original evaluation audit records.

    Repeating a frozen computation is reproducibility, not model selection.
    Each invocation predicts the holdout once; never tune using these results.
    """
    root = Path(root)
    frame = make_appointments(n=4000, seed=FROZEN["seed"])
    comparison, diagnostics, costs, folds = training_evidence(frame)
    with tempfile.TemporaryDirectory(prefix=".frozen-reproduction-", dir=root) as temp:
        stage = Path(temp)
        metrics = finalize(stage, frame, comparison, diagnostics)
        costs.to_csv(stage / "reports/reproduced_threshold_costs.csv", index=False)
        folds.to_csv(stage / "reports/reproduced_leakage_folds.csv", index=False)
        manifest = json.loads((stage / "reports/final_run_lock.json").read_text())
        manifest["status"] = "frozen_reproduction_completed"
        manifest["training_evidence_recomputed"] = True
        manifest["training_diagnostics"] = diagnostics
        manifest["purpose"] = "Reproduction only; no model or threshold reselection"
        (stage / "reports/reproduction_run.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        # Publish only after successful computation. Each file replacement is atomic;
        # the set of replacements is not a filesystem-wide transaction.
        for name in [
            "models/no_show.joblib",
            "models/model_info.json",
            "reports/cv_results.csv",
            "reports/metrics.json",
            "reports/final_predictions.csv",
            "reports/reproduction_run.json",
            "reports/reproduced_threshold_costs.csv",
            "reports/reproduced_leakage_folds.csv",
        ]:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, destination)
    return metrics


def main():
    print(json.dumps(reproduce(Path(__file__).resolve().parents[1]), indent=2))


if __name__ == "__main__":
    main()
