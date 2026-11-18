"""Reproduce development CV/tuning only; never evaluate the final holdout."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from noshows import pipeline, train

GENERATOR_COMMIT = "d061c41861006984fbe51760234ab3fc566b4688"
GENERATOR_SHA256 = "44e53ed91977d764f16c7fc5e8a2204eee58de1b3e35dc47ae2bdebeae0e99f2"


def load_generator(repo, git_executable="git"):
    """Read the reviewed corrected generator without changing the checkout."""
    source = subprocess.check_output(
        [git_executable, "-C", str(repo), "show", f"{GENERATOR_COMMIT}:noshows/data.py"]
    )
    if hashlib.sha256(source).hexdigest() != GENERATOR_SHA256:
        raise ValueError("Corrected generator hash does not match recorded evidence")
    namespace = {"__name__": "reproduction_generator"}
    exec(compile(source, "pinned_corrected_data.py", "exec"), namespace)
    return namespace["make_appointments"]


def training_partition(make_appointments):
    frame = make_appointments(n=4000, seed=pipeline.SEED)
    x_train, x_test, y_train, y_test = pipeline.split_data(frame)
    manifest = {
        "train_indices": x_train.index.tolist(),
        "reserved_test_indices": x_test.index.tolist(),
    }
    # No test feature/label inspection, prediction, scoring or serialization.
    del frame, x_test, y_test
    return x_train, y_train, manifest


def summarize(name, scores):
    row = {"model": name}
    for metric in train.SCORING:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


def reproduce(output_dir, git_executable="git", verify_existing=False):
    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    # Fail before computation rather than overwrite previous evidence.
    output_dir.mkdir(parents=True, exist_ok=False)
    generator = load_generator(repo, git_executable)
    x, y, manifest = training_partition(generator)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=pipeline.SEED)
    raw, rows = [], []
    for name, model in train.candidates(pipeline.SEED).items():
        scores = cross_validate(
            pipeline.build_pipeline(model),
            x,
            y,
            scoring=train.SCORING,
            cv=cv,
            error_score="raise",
        )
        rows.append(summarize(name, scores))
        raw.append(
            {
                "model": name,
                "fold_values": {key: values.tolist() for key, values in scores.items()},
            }
        )
    comparison = pd.DataFrame(rows)
    winner = (
        comparison[comparison["model"] != "dummy"]
        .sort_values(["f1_mean", "model"], ascending=[False, True])
        .iloc[0]["model"]
    )
    search = GridSearchCV(
        pipeline.build_pipeline(train.candidates(pipeline.SEED)[winner]),
        train.tuning_grid(winner),
        scoring="f1",
        cv=cv,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(x, y)
    grid = pd.DataFrame(search.cv_results_)
    if verify_existing:
        expected = pd.read_csv(repo / "reports/training_cv_summary.csv")
        pd.testing.assert_frame_equal(comparison, expected, atol=1e-12, rtol=1e-10)
        expected_grid = pd.read_csv(repo / "reports/training_tuning_grid.csv")
        score_columns = [
            key
            for key in expected_grid.columns
            if key.endswith("_test_score") or key.startswith("param_")
        ]
        pd.testing.assert_frame_equal(
            grid[score_columns],
            expected_grid[score_columns],
            check_dtype=False,
            atol=1e-12,
            rtol=1e-10,
        )
    comparison.to_csv(output_dir / "training_cv_summary.csv", index=False)
    grid.to_csv(output_dir / "training_tuning_grid.csv", index=False)
    (output_dir / "cv_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (output_dir / "split_indices.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    metadata = {
        "scope": "training-only development reproduction",
        "generator_commit": GENERATOR_COMMIT,
        "generator_sha256": GENERATOR_SHA256,
        "source_sha256": {
            path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
            for path in [
                "noshows/pipeline.py",
                "noshows/train.py",
                "noshows/reproduce_training.py",
            ]
        },
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "seed": pipeline.SEED,
        "training_rows": len(y),
        "folds": 5,
        "std_ddof": 0,
        "selection_metric": "f1",
        "selected_model": winner,
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "verified_against_committed_results": verify_existing,
        "holdout_evaluated": False,
        "threshold_selected": False,
        "final_model_saved": False,
        "caution": "Model-selection CV is not unbiased final performance.",
    }
    (output_dir / "reproduction_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return comparison, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-executable", default="git")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    comparison, metadata = reproduce(
        args.output_dir, args.git_executable, args.verify_existing
    )
    print(comparison.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
