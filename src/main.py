# main.py

from src.data_loader import load_news, load_all_stocks
from src.sentiment import compute_sentiment, aggregate_daily_sentiment
from src.features import (
    compute_returns,
    compute_volatility,
    compute_technical_indicators,
    merge_sentiment,
    create_lag_features,
    create_target,
)
from src.modeling import train_xgboost, get_time_series_split, cross_validate_model
from src.evaluation import compute_sharpe_ratio

# Load data
news = load_news("../data/raw_analyst_ratings.csv")
stocks = load_all_stocks("../data", files, names)

# Sentiment
news = compute_sentiment(news)
daily_sentiment = aggregate_daily_sentiment(news)

# Feature Engineering
stocks = compute_returns(stocks)
stocks = compute_volatility(stocks)
stocks = compute_technical_indicators(stocks)
stocks = merge_sentiment(stocks, daily_sentiment)
stocks = create_lag_features(stocks, ["Returns", "Volatility", "RSI", "MACD", "Sentiment"])
stocks = create_target(stocks)
stocks.dropna(inplace=True)

# Modeling
X = stocks[[col for col in stocks.columns if "lag1" in col]]
y = stocks["Target"]

model = train_xgboost()
tscv = get_time_series_split()
accuracy = cross_validate_model(model, X, y, tscv)

print("CV Accuracy:", accuracy)
