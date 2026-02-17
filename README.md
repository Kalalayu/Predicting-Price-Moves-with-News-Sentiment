# Predicting-Price-Moves-with-News-Sentiment
Analyzes financial news headlines and stock prices to test whether sentiment predicts short-term returns. Uses NLP for sentiment scoring, TA-Lib technical indicators (MA, RSI, MACD), and statistical correlation analysis. Includes reproducible EDA, data pipelines, and financial modeling workflows.
# Task 3: Correlation Between News and Stock Movements

## Objective
Analyze the correlation between financial news sentiment and stock price movements for six major stocks: AAPL, AMZN, GOOG, META, MSFT, NVDA.

## Steps Performed
1. Loaded stock price CSV files and the raw news dataset.
2. Converted dates to datetime format and handled timezone issues.
3. Calculated daily stock returns.
4. Performed sentiment analysis on news headlines using TextBlob.
5. Aggregated daily sentiment per stock.
6. Merged stock returns with daily sentiment scores.
7. Calculated Pearson correlation between returns and sentiment for each stock.
8. Plotted rolling correlations (14-day window) to identify trends.
9. Visualized daily returns vs sentiment.
10. Plotted distribution histograms for returns and sentiment.
11. Created a correlation heatmap across all stocks.

## Insights
- Rolling correlation shows short-term variations in how news sentiment aligns with stock price movements.
- Correlation strength varies across stocks, with some showing stronger links to news sentiment.
- Visualization helps identify periods where sentiment has more influence on returns.


## Predicting Stock Price Movements with News Sentiment
Overview

This project predicts next-day stock price movements using a combination of technical indicators and news sentiment. It leverages machine learning models such as Random Forest, Gradient Boosting, and XGBoost and provides explainability using LIME, SHAP, and Partial Dependence Plots.

Dataset

Stock Price Data: Daily OHLCV (Open, High, Low, Close, Volume) for multiple tech stocks (AAPL, AMZN, GOOG, META, MSFT, NVDA).

News Headlines: Daily news headlines per stock, used to compute sentiment scores.

Features

Technical Indicators: Returns, Rolling Volatility, RSI, MACD

Sentiment: Daily average headline sentiment

Lagged Features: Lagged values of all features to prevent lookahead bias

Target

Binary classification:

1 → Price goes UP the next day

0 → Price goes DOWN the next day

Modeling

Random Forest – Hyperparameter tuning using GridSearchCV and TimeSeriesSplit

Gradient Boosting / XGBoost – Alternative boosting methods for improved accuracy

Evaluation: Accuracy per fold (Time-Series CV) and economic performance using Strategy Sharpe Ratio

Explainability

LIME – Local feature contributions for individual predictions

SHAP – Global and interaction feature importance

Partial Dependence Plots – Visualize marginal effects of features on predictions

How to Run

Install dependencies:

pip install pandas numpy scikit-learn xgboost shap lime talib matplotlib textblob


Place datasets in data/ directory:

raw_analyst_ratings.csv

AAPL.csv, AMZN.csv, GOOG.csv, META.csv, MSFT.csv, NVDA.csv

Run predictive_modeling.ipynb notebook step by step.

Results

XGBoost Accuracy: ~70% (using lagged features and sentiment)

Feature Importance: Sentiment, Returns, RSI, Volatility, MACD

Strategy Sharpe Ratio: Calculated from model-driven trading signals