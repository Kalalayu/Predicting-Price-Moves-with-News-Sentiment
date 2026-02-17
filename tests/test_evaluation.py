# tests/test_evaluation.py

import numpy as np
from src.evaluation import compute_sharpe_ratio

def test_sharpe_ratio():
    returns = np.array([0.01, 0.02, -0.01, 0.015])
    sharpe = compute_sharpe_ratio(returns)

    assert isinstance(sharpe, float)
