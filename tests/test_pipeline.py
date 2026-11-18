"""Tests for data reproducibility, feature lists, splitting, and preprocessing."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from noshows.data import make_appointments
from noshows.pipeline import (
    ALL_FEATURES,
    EXCLUDED_COLS,
    TARGET_COL,
    build_pipeline,
    split_data,
)


def test_data_reproducibility_and_excluded_columns():
    df1 = make_appointments()
    df2 = make_appointments()
    pd.testing.assert_frame_equal(df1, df2)

    for col in EXCLUDED_COLS + [TARGET_COL]:
        assert col not in ALL_FEATURES


def test_stratified_split_size_and_proportions():
    df = make_appointments()
    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) == 3000
    assert len(X_test) == 1000
    assert np.isclose(y_train.mean(), y_test.mean(), atol=0.01)


def test_pipeline_handles_missing_distance():
    df = make_appointments()
    X_train, X_test, y_train, _ = split_data(df)

    assert X_test["distance_km"].isna().any()

    pipe = build_pipeline(DummyClassifier(strategy="most_frequent"))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    assert len(preds) == len(X_test)
    assert not np.isnan(preds).any()
