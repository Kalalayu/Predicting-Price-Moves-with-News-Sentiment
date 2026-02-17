# src/data_loader.py

import os
import pandas as pd


def load_news(path: str) -> pd.DataFrame:
    news = pd.read_csv(path)
    news.columns = news.columns.str.strip()
    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    news = news.dropna(subset=["date"])
    news["date_only"] = news["date"].dt.date
    return news


def load_stock(file_path: str, stock_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df["Stock"] = stock_name
    return df


def load_all_stocks(data_dir: str, files: list, names: list) -> pd.DataFrame:
    stocks = {
        name: load_stock(os.path.join(data_dir, file), name)
        for file, name in zip(files, names)
    }
    return pd.concat(stocks.values(), ignore_index=True)
