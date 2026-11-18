"""Training-only threshold and leakage evidence; never fit/save a final model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from noshows import evaluation, pipeline, train
from noshows.reproduce_training import (
    GENERATOR_COMMIT,
    GENERATOR_SHA256,
    load_generator,
)


def training_frame(generator):
    frame = generator(n=4000, seed=pipeline.SEED)
    x_train, x_test, y_train, y_test = pipeline.split_data(frame)
    selected = frame.loc[
        x_train.index, pipeline.ALL_FEATURES + ["fee_charged", "no_show"]
    ].copy()
    del frame, x_test, y_test
    assert selected["no_show"].equals(y_train)
    return selected


def analyze(frame):
    """Use previously reviewed tuned LR settings, not a new model search."""
    model = train.candidates(pipeline.SEED)["logistic_balanced"]
    model.set_params(C=0.1, solver="lbfgs")
    honest = pipeline.build_pipeline(model)
    leaky = pipeline.build_pipeline(
        clone(model), numeric_features=pipeline.NUMERIC_FEATURES + ["fee_charged"]
    )
    x = frame[pipeline.ALL_FEATURES]
    leak_x = frame[pipeline.ALL_FEATURES + ["fee_charged"]]
    y = frame["no_show"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=pipeline.SEED)
    probabilities = np.full(len(y), np.nan)
    visits = np.zeros(len(y), dtype=int)
    folds = []
    for fold, (fit, valid) in enumerate(cv.split(x, y), 1):
        fitted = clone(honest).fit(x.iloc[fit], y[fit])
        p = fitted.predict_proba(x.iloc[valid])[:, 1]
        leak_p = (
            clone(leaky)
            .fit(leak_x.iloc[fit], y[fit])
            .predict_proba(leak_x.iloc[valid])[:, 1]
        )
        probabilities[valid] = p
        visits[valid] += 1
        folds.append(
            {
                "fold": fold,
                "honest_roc_auc": roc_auc_score(y[valid], p),
                "leaky_roc_auc": roc_auc_score(y[valid], leak_p),
            }
        )
    if not np.all(visits == 1) or not np.isfinite(probabilities).all():
        raise ValueError(
            "Every training row must receive exactly one finite OOF probability"
        )
    thresholds = np.arange(5, 96, 5) / 100
    rows = []
    for threshold in thresholds:
        calls = probabilities >= threshold
        called = int(calls.sum())
        missed = int(((y == 1) & ~calls).sum())
        independent_cost = 8 * called + 35 * missed
        if independent_cost != evaluation.total_cost(y, calls):
            raise ValueError("Independent cost check failed")
        rows.append(
            {
                "threshold": threshold,
                "called": called,
                "uncalled_no_shows": missed,
                "cost": independent_cost,
            }
        )
    costs = pd.DataFrame(rows)
    selected = evaluation.choose_threshold(y, probabilities, thresholds)
    independent_choice = float(
        costs.sort_values(["cost", "threshold"]).iloc[0]["threshold"]
    )
    if selected != independent_choice:
        raise ValueError("Independent threshold/tie check failed")
    fold_table = pd.DataFrame(folds)
    summary = {
        "training_rows": len(y),
        "threshold": selected,
        "oof_cost": int(costs.loc[costs.threshold == selected, "cost"].iloc[0]),
        "call_nobody_cost": int(35 * y.sum()),
        "call_everybody_cost": int(8 * len(y)),
        "honest_auc_mean": float(fold_table.honest_roc_auc.mean()),
        "honest_auc_std": float(fold_table.honest_roc_auc.std(ddof=0)),
        "leaky_auc_mean": float(fold_table.leaky_roc_auc.mean()),
        "leaky_auc_std": float(fold_table.leaky_roc_auc.std(ddof=0)),
        "folds": 5,
        "std_ddof": 0,
        "seed": pipeline.SEED,
        "selected_model": "logistic_balanced",
        "settings": {"C": 0.1, "solver": "lbfgs", "class_weight": "balanced"},
        "holdout_evaluated": False,
        "final_model_saved": False,
        "scope": (
            "training-only development; OOF cost is selection evidence, "
            "not unbiased final performance"
        ),
        "caution": (
            "Model/settings were selected with the same training folds. "
            "Leakage deliberately adds post-outcome fee_charged only "
            "to the diagnostic pipeline. Assumes every call prevents "
            "a no-show."
        ),
    }
    return costs, fold_table, summary


def reproduce(output_dir, git_executable="git"):
    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    costs, folds, summary = analyze(
        training_frame(load_generator(repo, git_executable))
    )
    summary.update(
        {
            "generator_commit": GENERATOR_COMMIT,
            "generator_sha256": GENERATOR_SHA256,
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "source_sha256": {
                name: hashlib.sha256(
                    (repo / name)
                    .read_text(encoding="utf-8-sig")
                    .replace("\r\n", "\n")
                    .encode("utf-8")
                ).hexdigest()
                for name in [
                    "noshows/pipeline.py",
                    "noshows/evaluation.py",
                    "noshows/train.py",
                    "noshows/training_diagnostics.py",
                ]
            },
        }
    )
    costs.to_csv(output_dir / "training_threshold_costs.csv", index=False)
    folds.to_csv(output_dir / "training_leakage_folds.csv", index=False)
    (output_dir / "training_diagnostics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-executable", default="git")
    args = parser.parse_args()
    print(json.dumps(reproduce(args.output_dir, args.git_executable), indent=2))


if __name__ == "__main__":
    main()
