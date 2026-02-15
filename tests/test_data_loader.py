import pandas as pd
from data_loader import load_stock

def test_load_stock():
    df = load_stock("../data/AAPL.csv", "AAPL")
    assert isinstance(df, pd.DataFrame)
    assert 'Stock' in df.columns
    assert 'Close' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df.index)
