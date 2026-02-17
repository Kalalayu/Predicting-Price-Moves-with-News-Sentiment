# src/evaluation.py

import numpy as np


def compute_sharpe_ratio(returns: np.ndarray) -> float:
    return (returns.mean() / returns.std()) * np.sqrt(252)
