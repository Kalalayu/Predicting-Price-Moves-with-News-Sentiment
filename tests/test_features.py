# tests/test_features.py

import pandas as pd
from src.features import compute_returns, create_target

def test_compute_returns():
    df = pd.DataFrame({
        "Stock": ["A", "A", "A"],
        "Close": [100, 110, 105]
    })

    df = compute_returns(df)

    assert "Returns" in df.columns
    assert df["Returns"].isna().sum() == 1


def test_create_target():
    df = pd.DataFrame({
        "Stock": ["A", "A", "A"],
        "Returns": [0.01, -0.02, 0.03]
    })

    df = create_target(df)

    assert "Target" in df.columns
    assert set(df["Target"].dropna().unique()).issubset({0, 1})
