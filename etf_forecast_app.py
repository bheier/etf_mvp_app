import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# --- Helper Functions ---
def reliably_fetch_etf_data(symbol, attempt=1, max_attempts=5, delay=5):
    """Fetch ETF data with retry logic."""
    try:
        etf = yf.Ticker(symbol)
        hist = etf.history(period="max")
        info = etf.info
        return hist, info
    except Exception as e:
        if attempt < max_attempts:
            time_to_wait = delay * (2 ** (attempt - 1))
            st.error(f"Error fetching {symbol}: {str(e)}. Retrying in {time_to_wait} seconds...")
            time.sleep(time_to_wait)
            return reliably_fetch_etf_data(symbol, attempt + 1, max_attempts, delay)
        else:
            raise e

def is_currently_undervalued(hist, pe_ratio=None, pe_benchmark=15):
    """Determine if an ETF is currently undervalued."""
    short_ma = hist['Close'].rolling(window=50).mean().iloc[-1]
    long_ma = hist['Close'].rolling(window=200).mean().iloc[-1]
    current_price = hist['Close'].iloc[-1]
    
    ma_undervaluation = current_price < short_ma and current_price < long_ma
    
    pe_undervaluation = False
    if pe_ratio is not None:
        pe_undervaluation = pe_ratio < pe_benchmark
    
    # Determine if undervalued
    return ma_undervaluation or pe_undervaluation

def calculate_annualized_return(prices, years):
    if len(prices) < 252 * years:
        return None
    start_price = prices[-252 * years]
    end_price = prices[-1]
    return ((end_price / start_price) ** (1 / years)) - 1

def get_historical_returns(hist):
    prices = hist["Close"].dropna().values
    returns = {
        "1Y": calculate_annualized_return(prices, 1),
        "3Y": calculate_annualized_return(prices, 3),
        "5Y": calculate_annualized_return(prices, 5),
        "10Y": calculate_annualized_return(prices, 10),
        "Since Inception": ((prices[-1] / prices[0]) ** (1 / (len(prices) / 252))) - 1
    }
    return returns

def compute_score(undervaluation, history):
    historical_score_5Y = history["5Y"] if history["5Y"] is not None else 0
    historical_score_10Y = history["10Y"] if history["10Y"] is not None else 0
    total_historical_score = (historical_score_5Y + historical_score_10Y) / 2 * 0.5
    
    undervaluation_score = int(undervaluation) * 0.5
    return total_historical_score + undervaluation_score

# --- Streamlit UI ---
st.set_page_config(page_title="ETF Undervaluation & Forecast App", layout="wide")
st.title("ETF Undervaluation & Forecast App")

etf_symbols = st.text_input("Enter comma-separated ETF symbols:", value="VOO,VTI,SPY").upper().split(",")

results = []
for symbol in etf_symbols:
    try:
        hist, info = reliably_fetch_etf_data(symbol.strip())
        pe_ratio = info.get("trailingPE")

        undervaluation = is_currently_undervalued(hist, pe_ratio)
        history_returns = get_historical_returns(hist)
        score = compute_score(undervaluation, history_returns)
        
        results.append({
            "Symbol": symbol,
            "P/E": pe_ratio if pe_ratio is not None else 0,
            "1Y": history_returns["1Y"] if history_returns["1Y"] is not None else 0,
            "3Y": history_returns["3Y"] if history_returns["3Y"] is not None else 0,
            "5Y": history_returns["5Y"] if history_returns["5Y"] is not None else 0,
            "10Y": history_returns["10Y"] if history_returns["10Y"] is not None else 0,
            "Since Inception": history_returns["Since Inception"] if history_returns["Since Inception"] is not None else 0,
            "Currently Undervalued": undervaluation,
            "Score": score if score is not None else 0
        })

        # Add a delay between symbols to further prevent rate limiting
        time.sleep(2)
    
    except Exception as e:
        st.error(f"Persistent error processing {symbol}: {str(e)}")

if results:
    df = pd.DataFrame(results)
    df_sorted = df.sort_values("Score", ascending=False).reset_index(drop=True)

    df_sorted = df_sorted.fillna(0)  # Fills NaN with 0 to avoid formatting issues

    st.subheader("ETF Rankings by Score")
    st.dataframe(
        df_sorted.style.format({
            "P/E": "{:.2f}",
            "1Y": "{:.2%}",
            "3Y": "{:.2%}",
            "5Y": "{:.2%}",
            "10Y": "{:.2%}",
            "Since Inception": "{:.2%}",
            "Currently Undervalued": lambda x: f"{'Yes' if x else 'No'}",
            "Score": "{:.2f}"
        }), 
        use_container_width=True
    )
