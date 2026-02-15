import talib
import pandas as pd

def add_sma(df: pd.DataFrame, period: int = 20, col: str = 'Close') -> pd.Series:
    return talib.SMA(df[col], timeperiod=period)

def add_ema(df: pd.DataFrame, period: int = 20, col: str = 'Close') -> pd.Series:
    return talib.EMA(df[col], timeperiod=period)

def add_rsi(df: pd.DataFrame, period: int = 14, col: str = 'Close') -> pd.Series:
    return talib.RSI(df[col], timeperiod=period)

def add_macd(df: pd.DataFrame, col: str = 'Close'):
    macd, signal, hist = talib.MACD(df[col], fastperiod=12, slowperiod=26, signalperiod=9)
    return macd, signal, hist

def add_returns(df: pd.DataFrame, col: str = 'Close') -> pd.Series:
    return df[col].pct_change()

def add_rolling_volatility(df: pd.DataFrame, window: int = 20, col: str = 'Close') -> pd.Series:
    return df[col].pct_change().rolling(window).std()
