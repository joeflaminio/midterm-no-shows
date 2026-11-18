"""Shared feature definitions, data split, and preprocessing pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 11
TARGET_COL = "no_show"
EXCLUDED_COLS = ["appointment_id", "fee_charged"]

NUMERIC_FEATURES = [
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
]

CATEGORICAL_FEATURES = [
    "weekday",
    "booking",
    "insurance",
    "treatment",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a fixed-seed, stratified 75/25 train/test split."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_pipeline(
    model: BaseEstimator,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Build a preprocessing + estimator pipeline using a ColumnTransformer."""
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
