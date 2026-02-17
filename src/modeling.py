# src/modeling.py

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_time_series_split(n_splits=5):
    return TimeSeriesSplit(n_splits=n_splits)


def train_random_forest(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    return model


def train_xgboost():
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )


def cross_validate_model(model, X, y, tscv):
    scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")
    return np.mean(scores)
