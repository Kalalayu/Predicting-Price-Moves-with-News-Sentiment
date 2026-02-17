import pandas as pd
from src.indicators import add_sma, add_rsi


def test_indicators():
    df = pd.DataFrame({'Close': [10,11,12,13,14,15,16,17,18,19,20]})
    sma = add_sma(df, period=3)
    rsi = add_rsi(df, period=3)
    assert len(sma) == len(df)
    assert len(rsi) == len(df)
