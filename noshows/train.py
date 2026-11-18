"""Sahar's draft orchestration; teammate interfaces need integration agreement.

All model selection uses training data. Importing this module does not train anything.
The final CLI uses Joe's pipeline and Vedant's merged evaluation functions.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier

SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}


@dataclass(frozen=True)
class Interfaces:
    """Injectable team interfaces; production binding is in project_interfaces()."""

    seed: int
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    build_pipeline: Callable
    total_cost: Callable
    choose_threshold: Callable

    @property
    def features(self):
        return list(self.numeric_features + self.categorical_features)


def candidates(seed):
    """Recommendations: the handout does not mandate these three algorithms."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_balanced": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed
        ),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed),
    }


def tuning_grid(name):
    """All candidate grids have two parameters and six combinations.

    Joe's shared estimator step is named 'model'.
    """
    grids = {
        "logistic_balanced": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"],
        },
        "decision_tree": {
            "model__max_depth": [3, 6, None],
            "model__min_samples_leaf": [5, 20],
        },
        "random_forest": {
            "model__max_depth": [4, 8, None],
            "model__min_samples_leaf": [2, 10],
        },
    }
    return grids[name]


def compare_models(x_train, y_train, api, cv):
    rows = []
    for name, estimator in candidates(api.seed).items():
        scores = cross_validate(
            api.build_pipeline(estimator),
            x_train,
            y_train,
            scoring=SCORING,
            cv=cv,
            error_score="raise",
        )
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_training(train_frame, api):
    """Select model/settings/threshold using ONLY a training frame.

    CV estimates used for selection are not an unbiased final performance estimate.
    The once-only holdout evaluation supplies that final estimate.
    """
    forbidden = {"appointment_id", "fee_charged", "no_show"}
    if forbidden.intersection(api.features):
        raise ValueError("Production feature lists contain an excluded column")
    x_train = train_frame.loc[:, api.features]
    y_train = train_frame["no_show"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=api.seed)
    comparison = compare_models(x_train, y_train, api, cv)
    eligible = comparison[comparison["model"] != "dummy"]
    selected = eligible.sort_values(["f1_mean", "model"], ascending=[False, True]).iloc[
        0
    ]
    name = selected["model"]
    untuned = api.build_pipeline(candidates(api.seed)[name])
    search = GridSearchCV(
        untuned, tuning_grid(name), scoring="f1", cv=cv, error_score="raise"
    )
    search.fit(x_train, y_train)
    # Keep the untuned model if the grid did not improve the training-CV mean.
    use_tuned = search.best_score_ > selected["f1_mean"]
    chosen = clone(search.best_estimator_ if use_tuned else untuned)
    oof = cross_val_predict(chosen, x_train, y_train, cv=cv, method="predict_proba")[
        :, 1
    ]
    threshold = float(api.choose_threshold(y_train, oof, np.arange(5, 96, 5) / 100))
    if not 0.05 <= threshold <= 0.95:
        raise ValueError("Threshold must be between 0.05 and 0.95")

    # Same builder and selected estimator; only this experiment adds leakage.
    leak_features = list(api.numeric_features) + ["fee_charged"]
    leaky = api.build_pipeline(
        clone(chosen.named_steps["model"]), numeric_features=leak_features
    )
    leak_auc = cross_validate(
        leaky,
        train_frame.loc[:, api.features + ["fee_charged"]],
        y_train,
        cv=cv,
        scoring="roc_auc",
        error_score="raise",
    )["test_score"]
    honest_auc = cross_validate(
        chosen, x_train, y_train, cv=cv, scoring="roc_auc", error_score="raise"
    )["test_score"]
    gain = float(search.best_score_ - selected["f1_mean"])
    details = {
        "selected_model": name,
        "selection_scoring": "f1",
        "tuning_best_params": search.best_params_,
        "tuning_best_cv_mean": float(search.best_score_),
        "untuned_cv_mean": float(selected["f1_mean"]),
        "untuned_cv_std": float(selected["f1_std"]),
        "tuning_gain": gain,
        "gain_exceeds_untuned_fold_std": bool(gain > selected["f1_std"]),
        "used_tuned_model": bool(use_tuned),
        "best_settings": {
            k: v for k, v in chosen.get_params().items() if k in tuning_grid(name)
        },
        "threshold": threshold,
        "threshold_grid": (np.arange(5, 96, 5) / 100).tolist(),
        "honest_cv_roc_auc": float(np.mean(honest_auc)),
        "leaky_cv_roc_auc": float(np.mean(leak_auc)),
        "cv_std_convention": "population standard deviation across five folds",
    }
    chosen.fit(x_train, y_train)
    return chosen, comparison, details


def evaluate_final(model, test_frame, threshold, api):
    """Call once after all choices are frozen; do not use to revise selection."""
    probabilities = model.predict_proba(test_frame.loc[:, api.features])[:, 1]
    y = test_frame["no_show"].to_numpy()
    calls = probabilities >= threshold
    costs = {
        "call_nobody": float(api.total_cost(y, np.zeros(len(y), dtype=bool))),
        "call_everybody": float(api.total_cost(y, np.ones(len(y), dtype=bool))),
        "model": float(api.total_cost(y, calls)),
    }
    return {
        "test_size": len(y),
        "decision_threshold": threshold,
        "precision": float(precision_score(y, calls, zero_division=0)),
        "recall": float(recall_score(y, calls, zero_division=0)),
        "f1": float(f1_score(y, calls, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, calls)),
        "costs": costs,
        "beats_both_alternatives": bool(
            costs["model"] < min(costs["call_nobody"], costs["call_everybody"])
        ),
        "caution": (
            "Synthetic data; finite test sample; assumes every call prevents a no-show."
        ),
    }


def save_outputs(root, model, comparison, details, metrics, api):
    root = Path(root)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    info = {
        "sklearn_version": sklearn.__version__,
        "seed": api.seed,
        "numeric_features": list(api.numeric_features),
        "categorical_features": list(api.categorical_features),
        **details,
        "test_results": metrics,
    }
    joblib.dump(model, root / "models/no_show.joblib")
    comparison.to_csv(root / "reports/cv_results.csv", index=False)
    for path, data in [
        ("models/model_info.json", info),
        ("reports/metrics.json", metrics),
    ]:
        (root / path).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def restore_split_frames(frame, split):
    """Adapt Joe's four outputs without resplitting or losing leakage-demo columns."""
    x_train, x_test, y_train, y_test = split
    if not frame.index.is_unique:
        raise ValueError("Source row indices must be unique")
    train_index, test_index = x_train.index, x_test.index
    if not train_index.is_unique or not test_index.is_unique:
        raise ValueError("Split row indices must be unique")
    if len(train_index.intersection(test_index)):
        raise ValueError("Training and test indices overlap")
    if set(train_index).union(test_index) != set(frame.index):
        raise ValueError("Split indices must preserve all original rows")
    for x, y in [(x_train, y_train), (x_test, y_test)]:
        if not y.index.equals(x.index) or not y.equals(frame.loc[x.index, "no_show"]):
            raise ValueError("Split labels must preserve original row alignment")
        if not x.equals(frame.loc[x.index, x.columns]):
            raise ValueError("Split features must preserve original row values")
    return frame.loc[train_index].copy(), frame.loc[test_index].copy()


def project_interfaces():
    """Bind the real shared pipeline and merged evaluation implementation."""
    from noshows.evaluation import choose_threshold, total_cost
    from noshows.pipeline import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        SEED,
        build_pipeline,
    )

    return Interfaces(
        SEED,
        tuple(NUMERIC_FEATURES),
        tuple(CATEGORICAL_FEATURES),
        build_pipeline,
        total_cost,
        choose_threshold,
    )


def main():
    # Reproduce approved frozen settings, never tune against the holdout.
    from noshows.finalize_training import main as finalize_main

    finalize_main()


if __name__ == "__main__":
    main()
