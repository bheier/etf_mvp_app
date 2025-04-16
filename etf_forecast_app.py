import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as plt
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# --- Helper Functions ---
def reliably_fetch_etf_data(symbol, attempt=1, max_attempts=5, delay=5):
    """Fetch ETF data with retry logic and debug logging."""
    try:
        etf = yf.Ticker(symbol)
        hist = etf.history(period="max")
        info = etf.info
        return hist, info
    except Exception as e:
        if attempt < max_attempts:
            time_to_wait = delay * (2 ** (attempt - 1))
            logging.error(f"Error fetching {symbol}: {str(e)}. Retrying in {time_to_wait} seconds...")
            time.sleep(time_to_wait)  # Exponential backoff
            return reliably_fetch_etf_data(symbol, attempt + 1, max_attempts, delay)
        else:
            raise e

# Use the rest of your code as previously defined...

# Example of implementation with retries
etf_symbols = st.text_input("Enter comma-separated ETF symbols:", value="VOO,VTI,SPY").upper().split(",")

results = []
for symbol in etf_symbols:
    try:
        hist, info = reliably_fetch_etf_data(symbol.strip())
        # Existing logic to process the data...
    except Exception as e:
        st.error(f"Persistent error processing {symbol}: {str(e)}")

# Your remaining code to display results continues here...
