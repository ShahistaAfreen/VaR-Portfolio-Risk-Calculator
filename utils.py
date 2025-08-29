import pandas as pd
import numpy as np
from typing import Tuple


def load_prices(csv_file) -> pd.DataFrame:
    """Load CSV with Date column. Returns DataFrame of prices indexed by Date (datetime)."""
    df = pd.read_csv(csv_file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df


def prices_to_returns(prices: pd.DataFrame, kind: str = 'log') -> pd.DataFrame:
    if kind == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return prices.pct_change().dropna()


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.asarray(weights)
    assert returns.shape[1] == w.size, "Weight vector length must equal number of assets"
    return returns.dot(w)


def ensure_weights(weights: list, n_assets: int) -> np.ndarray:
    if weights is None:
        w = np.ones(n_assets) / n_assets
    else:
        w = np.array(weights, dtype=float)
        if w.size != n_assets:
            if w.size == 1:
                w = np.ones(n_assets) * w[0]
            else:
                padded = np.zeros(n_assets)
                padded[: w.size] = w
                w = padded
        s = w.sum()
        if np.isclose(s, 0.0):
            w = np.ones(n_assets) / n_assets
        else:
            w = w / s
    return w
