import os
import pytest
from src.data_loader import load_stock

def test_load_stock():
    # Make path relative to this test file
    file_path = os.path.join(os.path.dirname(__file__), "../data/AAPL.csv")
    
    df = load_stock(file_path, "AAPL")

    # Basic assertions
    assert not df.empty
    assert "Stock" in df.columns
    assert df["Stock"].iloc[0] == "AAPL"
    assert df.index.name == "Date"
