# VaR Portfolio Risk Calculator


Python-based tool estimating portfolio risk using Historical, Parametric, and Monte Carlo VaR models with backtesting and stress-testing.


## Features
- Historical VaR (non-parametric)
- Parametric VaR (Gaussian - portfolio-level using mean & covariance)
- Monte Carlo VaR (multivariate normal simulations)
- Backtesting (count exceptions, simple p-value using binomial test)
- Streamlit UI for uploading price files, selecting models and parameters, and visualizing results
- Fetch live market data using `yfinance`


## Quickstart
1. Create a Python environment (recommended: venv)


```bash
python -m venv .venv
source .venv/bin/activate # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
