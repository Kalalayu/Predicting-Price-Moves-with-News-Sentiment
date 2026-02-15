import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_price_with_sma_ema(df, stock, output_dir='outputs'):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Close Price')
    if 'SMA_20' in df.columns:
        plt.plot(df.index, df['SMA_20'], label='SMA 20')
    if 'EMA_20' in df.columns:
        plt.plot(df.index, df['EMA_20'], label='EMA 20')
    plt.title(f'{stock} Price with SMA/EMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{stock}_price_sma_ema.png")
    plt.close()

def plot_rsi(df, stock, output_dir='outputs'):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df['RSI_14'], label='RSI 14')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title(f'{stock} RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{stock}_RSI.png")
    plt.close()

def plot_macd(df, stock, output_dir='outputs'):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['MACD_signal'], label='Signal')
    plt.bar(df.index, df['MACD_hist'], label='Histogram', alpha=0.3)
    plt.title(f'{stock} MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{stock}_MACD.png")
    plt.close()

def plot_returns_volatility(df, stock, output_dir='outputs'):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df['Returns'], label='Daily Returns')
    plt.plot(df.index, df['Volatility'], label='Rolling Volatility')
    plt.title(f'{stock} Returns & Volatility')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{stock}_returns_volatility.png")
    plt.close()
