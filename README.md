# Predicting-Price-Moves-with-News-Sentiment
Analyzes financial news headlines and stock prices to test whether sentiment predicts short-term returns. Uses NLP for sentiment scoring, TA-Lib technical indicators (MA, RSI, MACD), and statistical correlation analysis. Includes reproducible EDA, data pipelines, and financial modeling workflows.
# Task 2: Quantitative Analysis with TA-Lib and PyNance

## Overview
This task performs quantitative analysis on multiple stock datasets using **technical indicators** and **financial metrics**. The goal is to complement the news sentiment analysis from Task 1 with price-based signals.

We use six major stock datasets: **AAPL, AMZN, GOOG, META, MSFT, NVDA**.

## Features
- Load and prepare stock price data with columns: Open, High, Low, Close, Volume
- Calculate **Technical Indicators**:
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Daily Returns and Rolling Volatility
- Save processed datasets for future analysis
- Visualizations for each stock:
  - Price with SMA and EMA
  - RSI with overbought/oversold thresholds
  - MACD with signal and histogram
  - Returns & Volatility
- Modular design: separated functions for loading data, calculating indicators, and plotting results

## Requirements
- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- TA-Lib
- PyNance (optional for advanced metrics)


