# tests/test_data_loader.py

import pandas as pd
from src.data_loader import load_news

def test_load_news(tmp_path):
    # Create fake CSV
    file = tmp_path / "news.csv"
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "headline": ["Good news", "Bad news"]
    })
    df.to_csv(file, index=False)

    news = load_news(file)

    assert "date_only" in news.columns
    assert len(news) == 2
