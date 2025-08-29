import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest
from typing import Tuple


def historical_var(portfolio_returns: pd.Series, alpha: float = 0.05) -> float:
    q = np.quantile(portfolio_returns, alpha)
    return -q


def parametric_var(portfolio_mean: float, portfolio_std: float, alpha: float = 0.05, days: int = 1) -> float:
    z = norm.ppf(alpha)
    mu = portfolio_mean * days
    sigma = portfolio_std * np.sqrt(days)
    q = mu + z * sigma
    return -q


def parametric_var_from_assets(returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.05, days: int = 1) -> Tuple[float, float, float]:
    w = np.asarray(weights)
    mu = returns.mean().dot(w)
    cov = returns.cov()
    sigma = np.sqrt(w.T.dot(cov).dot(w))
    var = parametric_var(mu, sigma, alpha=alpha, days=days)
    return float(var), float(mu), float(sigma)


def monte_carlo_var(returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.05, n_sims: int = 10000, days: int = 1, seed: int = 42) -> float:
    np.random.seed(seed)
    w = np.asarray(weights)
    mu = returns.mean().values
    cov = returns.cov().values

    mu_h = mu * days
    cov_h = cov * days

    sims = np.random.multivariate_normal(mean=mu_h, cov=cov_h, size=n_sims)
    port_sims = sims.dot(w)
    q = np.quantile(port_sims, alpha)
    return float(-q)


def backtest_var(portfolio_returns: pd.Series, var_series: pd.Series) -> dict:
    exceptions = (portfolio_returns < -var_series).sum()
    n = len(portfolio_returns)
    ex_rate = exceptions / n
    try:
        expected_alpha = float(var_series.name)
        if expected_alpha <= 0 or expected_alpha >= 0.5:
            expected_alpha = 0.05
    except Exception:
        expected_alpha = 0.05

    p_val = binomtest(exceptions, n, expected_alpha, alternative='two-sided').pvalue

    return {
        'exceptions': int(exceptions),
        'n': int(n),
        'exception_rate': float(ex_rate),
        'expected_alpha': expected_alpha,
        'p_value': float(p_val),
    }
