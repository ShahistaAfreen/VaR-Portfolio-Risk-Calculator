import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_prices, prices_to_returns, portfolio_returns, ensure_weights
from var_models import historical_var, parametric_var_from_assets, monte_carlo_var, backtest_var

st.set_page_config(page_title='VaR Portfolio Risk Calculator', layout='wide')

st.title('Value-at-Risk (VaR) Portfolio Risk Calculator')

uploaded = st.file_uploader('Upload CSV of historical prices (or run fetch_data.py first)', type=['csv'])

if uploaded is not None:
    prices = load_prices(uploaded)
    st.write('Preview:')
    st.dataframe(prices.head())

    returns = prices_to_returns(prices, kind='log')

    cols = returns.columns.tolist()
    st.sidebar.header('Portfolio Weights')
    weights_input = []
    for c in cols:
        val = st.sidebar.text_input(f'Weight for {c}', value='')
        weights_input.append(val)

    def parse_weights(inputs):
        parsed = []
        for v in inputs:
            try:
                parsed.append(float(v))
            except:
                parsed.append(0.0)
        return parsed

    weights = ensure_weights(parse_weights(weights_input), len(cols))

    alpha = st.sidebar.selectbox('Alpha (1-confidence)', [0.01, 0.025, 0.05], index=2)
    days = st.sidebar.number_input('Horizon (days)', min_value=1, value=1)
    n_sims = st.sidebar.number_input('Monte Carlo simulations', min_value=1000, value=10000, step=1000)

    port_ret = portfolio_returns(returns, weights)

    st.header('VaR Estimates')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Historical VaR', f"{historical_var(port_ret, alpha):.6f}")
    with col2:
        p_var, p_mu, p_sigma = parametric_var_from_assets(returns, weights, alpha=alpha, days=days)
        st.metric('Parametric VaR', f"{p_var:.6f}")
    with col3:
        mc = monte_carlo_var(returns, weights, alpha=alpha, n_sims=n_sims, days=days)
        st.metric('Monte Carlo VaR', f"{mc:.6f}")

    st.header('Backtesting (Rolling Historical VaR)')
    window = st.number_input('Rolling window (days)', min_value=10, value=250)
    rolling_var = -port_ret.rolling(window=window).quantile(alpha)
    rolling_var.name = str(alpha)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(port_ret.index, port_ret, label='Portfolio Returns')
    ax.plot(rolling_var.index, -rolling_var, label='VaR threshold')
    ax.legend()
    st.pyplot(fig)

    bt = backtest_var(port_ret.dropna(), rolling_var.dropna())
    st.write('Backtest summary:')
    st.json(bt)
else:
    st.info('Upload a price CSV or run fetch_data.py to generate one.')
