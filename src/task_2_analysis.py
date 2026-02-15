import pandas as pd
from data_loader import load_all_stocks
from indicators import add_sma, add_ema, add_rsi, add_macd, add_returns, add_rolling_volatility
from visualizations import plot_price_with_sma_ema, plot_rsi, plot_macd, plot_returns_volatility

# Map stock names to relative CSV paths
stock_files = {
    "AAPL": "../data/AAPL.csv",
    "AMZN": "../data/AMZN.csv",
    "GOOG": "../data/GOOG.csv",
    "META": "../data/META.csv",
    "MSFT": "../data/MSFT.csv",
    "NVDA": "../data/NVDA.csv"
}

# Load all stocks
all_stocks = load_all_stocks(stock_files)

# Process each stock separately
for stock_name, stock_df in all_stocks.groupby('Stock'):
    stock_df['SMA_20'] = add_sma(stock_df)
    stock_df['EMA_20'] = add_ema(stock_df)
    stock_df['RSI_14'] = add_rsi(stock_df)
    stock_df['MACD'], stock_df['MACD_signal'], stock_df['MACD_hist'] = add_macd(stock_df)
    stock_df['Returns'] = add_returns(stock_df)
    stock_df['Volatility'] = add_rolling_volatility(stock_df)
    
    # Save processed DataFrame
    stock_df.to_csv(f'outputs/{stock_name}_processed.csv')
    
    # Create visualizations
    plot_price_with_sma_ema(stock_df, stock_name)
    plot_rsi(stock_df, stock_name)
    plot_macd(stock_df, stock_name)
    plot_returns_volatility(stock_df, stock_name)

print("Task 2 Analysis Completed ✅ All indicators and plots generated in 'outputs/' folder.")
