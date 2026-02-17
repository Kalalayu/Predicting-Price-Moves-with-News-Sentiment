# src/features.py

import pandas as pd
import talib


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["Returns"] = df.groupby("Stock")["Close"].pct_change()
    return df


def compute_volatility(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df["Volatility"] = (
        df.groupby("Stock")["Returns"]
        .transform(lambda x: x.rolling(window).std())
    )
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI"] = df.groupby("Stock")["Close"].transform(
        lambda x: talib.RSI(x, timeperiod=14)
    )

    def macd_calc(series):
        macd, _, _ = talib.MACD(series)
        return pd.Series(macd, index=series.index)

    df["MACD"] = (
        df.groupby("Stock")["Close"]
        .apply(macd_calc)
        .reset_index(level=0, drop=True)
    )

    return df


def merge_sentiment(df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.merge(sentiment_df, on=["Stock", "Date"], how="left")
    df["Sentiment"] = df["Sentiment"].fillna(0)
    return df


def create_lag_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    for f in features:
        df[f"{f}_lag1"] = df.groupby("Stock")[f].shift(1)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["Target"] = df.groupby("Stock")["Returns"].shift(-1)
    df["Target"] = (df["Target"] > 0).astype(int)
    return df
