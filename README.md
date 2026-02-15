# Predicting-Price-Moves-with-News-Sentiment
Analyzes financial news headlines and stock prices to test whether sentiment predicts short-term returns. Uses NLP for sentiment scoring, TA-Lib technical indicators (MA, RSI, MACD), and statistical correlation analysis. Includes reproducible EDA, data pipelines, and financial modeling workflows.
# Task 1: Financial News Dataset EDA & Text Analysis

## Overview
This notebook contains the Exploratory Data Analysis (EDA) and text analysis of the financial news dataset. The goal is to understand trends, headline patterns, publisher contributions, and key financial terms in the dataset.

## Contents
1. **Descriptive Statistics**
   - Headline lengths
   - Number of articles per publisher
   - Publication trends over time and by day of week

2. **Text Analysis**
   - Most common words
   - Financial event keywords (earnings, upgrades, price targets)
   - Publisher vs. keyword mapping
   - Top 2-word phrases (bigrams)

3. **Visualizations**
   - Histograms of headline lengths
   - Bar charts of top publishers
   - Day-of-week article distribution
   - Top bigrams with counts

4. **Key Insights**
   - Headlines are concise and event-driven
   - Most news comes from a few primary publishers
   - Financial events like earnings, price targets, upgrades, and 52-week highs dominate coverage
   - Variation in publisher focus and reporting style

## Usage
- Run the notebook in a Python environment with the following dependencies:
  - pandas
  - matplotlib
  - sklearn
