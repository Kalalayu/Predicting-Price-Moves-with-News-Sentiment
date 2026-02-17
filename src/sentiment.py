# src/sentiment.py

from textblob import TextBlob
import pandas as pd


def compute_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    news["Sentiment"] = news["headline"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    return news


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    daily = (
        news.groupby(["stock", "date_only"])["Sentiment"]
        .mean()
        .reset_index()
    )

    daily.rename(columns={"stock": "Stock", "date_only": "Date"}, inplace=True)
    daily["Date"] = pd.to_datetime(daily["Date"])

    return daily
