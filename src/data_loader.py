import os
import pandas as pd


def load_stock(file_path: str, stock_name: str) -> pd.DataFrame:
    """
    Load a single stock CSV file.

    - Validates file existence
    - Converts Date column to datetime
    - Sets Date as index
    - Adds Stock column
    """

    # Ensure file exists (prevents silent CI failures)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stock file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Validate required column
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")

    # Convert Date to datetime safely
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows where Date failed conversion
    df = df.dropna(subset=["Date"])

    df.set_index("Date", inplace=True)

    # Add stock identifier column
    df["Stock"] = stock_name

    return df


def load_all_stocks(stock_files: dict) -> pd.DataFrame:
    """
    Load multiple stocks from a dictionary:
        { "AAPL": "path/to/AAPL.csv", ... }

    Returns:
        Concatenated DataFrame sorted by date.
    """

    if not stock_files:
        raise ValueError("stock_files dictionary cannot be empty.")

    stocks = []

    for name, fp in stock_files.items():
        stock_df = load_stock(fp, name)
        stocks.append(stock_df)

    all_stocks = pd.concat(stocks)

    # Sort by date index
    all_stocks.sort_index(inplace=True)

    return all_stocks
