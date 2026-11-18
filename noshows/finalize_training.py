"""Frozen, once-only final run; never select settings using the holdout."""

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from noshows import pipeline, train
from noshows.reproduce_training import load_generator

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


def main():
    root = Path(__file__).resolve().parents[1]
    # Refuse before generating data if this checkout has already run.
    if (root / "reports/final_run_lock.json").exists():
        raise FileExistsError("Final run already started/completed; do not repeat")
    diagnostics = json.loads((root / "reports/training_diagnostics.json").read_text())
    comparison = pd.read_csv(root / "reports/training_cv_summary.csv")
    generator = load_generator(
        root, shutil.which("git") or "C:/Program Files/Git/cmd/git.exe"
    )
    print(
        json.dumps(
            finalize(root, generator(n=4000, seed=11), comparison, diagnostics),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
