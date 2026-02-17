# tests/test_sentiment.py

import pandas as pd
from src.sentiment import compute_sentiment

def test_compute_sentiment():
    df = pd.DataFrame({
        "headline": ["Stock is great", "Stock is terrible"]
    })

    df = compute_sentiment(df)

    assert "Sentiment" in df.columns
    assert df["Sentiment"].dtype == float
