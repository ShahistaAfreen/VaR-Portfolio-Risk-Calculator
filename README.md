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
source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Fetch sample data

```bash
python fetch_data.py
```

This will create `sample_data/prices.csv` with prices for Apple (AAPL), Microsoft (MSFT), and Google (GOOG).

3. Run the app

```bash
streamlit run streamlit_app.py
```

## Input format

Expect a CSV of historical **adjusted close** prices with a Date column and one column per asset. Example:

```
Date,Asset_A,Asset_B,Asset_C
2020-01-01,100.1,50.3,200.2
2020-01-02,100.9,50.1,199.0
...
```

## Notes

* Monte Carlo uses multivariate normal sampling of log returns; you can change to heavier-tailed distributions in `var_models.py`.
* Parametric VaR uses portfolio mean and covariance; results assume elliptical returns (normality).
