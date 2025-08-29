import yfinance as yf
import pandas as pd

# Example tickers (edit as needed)
tickers = ["AAPL", "MSFT", "GOOG"]

print(f"Fetching data for {tickers}...")
data = yf.download(tickers, start="2020-01-01", end="2023-12-31")["Adj Close"]

# Save to CSV
data.to_csv("sample_data/prices.csv")
print("Data saved to sample_data/prices.csv")
