# tests/test_modeling.py

import numpy as np
from src.modeling import train_random_forest, get_time_series_split
from sklearn.datasets import make_classification

def test_random_forest_training():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)

    model = train_random_forest(X, y)
    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == 50


def test_time_series_split():
    tscv = get_time_series_split(n_splits=3)
    assert tscv.n_splits == 3
