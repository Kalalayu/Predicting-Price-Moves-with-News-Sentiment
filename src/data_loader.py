import pandas as pd

def load_stock(file_path: str, stock_name: str) -> pd.DataFrame:
    """
    Load a single stock CSV, convert Date to datetime, and add Stock column.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Stock'] = stock_name
    return df

def load_all_stocks(stock_files: dict) -> pd.DataFrame:
    """
    Load all stocks from a dictionary of file paths.
    Returns a single concatenated DataFrame.
    """
    stocks = [load_stock(fp, name) for name, fp in stock_files.items()]
    all_stocks = pd.concat(stocks)
    all_stocks.sort_index(inplace=True)
    return all_stocks
