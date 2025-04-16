import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
            time_to_wait = delay * (2 ** (attempt - 1))  # Exponential backoff
            st.error(f"Error fetching {symbol}: {str(e)}. Retrying in {time_to_wait} seconds...")
            time.sleep(time_to_wait)
            return reliably_fetch_etf_data(symbol, attempt + 1, max_attempts, delay)
        else:
            raise e

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

def get_undervaluation(info):
    try:
        current_price = info.get("currentPrice")
        fair_value = info.get("targetMeanPrice")  # Assuming "targetMeanPrice" is a proxy for fair value
        if current_price and fair_value:
            undervaluation = ((fair_value - current_price) / fair_value) * 100
            return undervaluation
        else:
            return None
    except:
        return None

def ml_forecast(hist, years_ahead=5):
    df = hist["Close"].dropna().reset_index()
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days
    X = df[["Days"]]
    y = df["Close"]
    model = LinearRegression().fit(X, y)
    future_day = df["Days"].max() + (252 * years_ahead)
    pred_price = model.predict([[future_day]])[0]
    current_price = y.iloc[-1]
    annual_return = ((pred_price / current_price) ** (1 / years_ahead)) - 1
    return annual_return

def compute_score(valuation, history, future):
    historical_score = history["5Y"] * 0.5 if history["5Y"] is not None else 0
    undervaluation_score = get_undervaluation(valuation) * 0.25 if get_undervaluation(valuation) is not None else 0
    future_score = future * 0.25 if future is not None else 0
    return historical_score + undervaluation_score + future_score

# --- Streamlit UI ---
st.set_page_config(page_title="ETF Undervaluation & Forecast App", layout="wide")
st.title("ETF Undervaluation & Forecast App")

etf_symbols = st.text_input("Enter comma-separated ETF symbols:", value="VOO,VTI,SPY").upper().split(",")

results = []
for symbol in etf_symbols:
    try:
        hist, info = reliably_fetch_etf_data(symbol.strip())
        valuation = {
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook")
        }
        history_returns = get_historical_returns(hist)
        forecast_return = ml_forecast(hist)
        score = compute_score(valuation, history_returns, forecast_return)
        
        undervaluation = get_undervaluation(info)

        results.append({
            "Symbol": symbol,
            "P/E": valuation["pe_ratio"] if valuation["pe_ratio"] is not None else 0,
            "P/B": valuation["pb_ratio"] if valuation["pb_ratio"] is not None else 0,
            "1Y": history_returns["1Y"] if history_returns["1Y"] is not None else 0,
            "3Y": history_returns["3Y"] if history_returns["3Y"] is not None else 0,
            "5Y": history_returns["5Y"] if history_returns["5Y"] is not None else 0,
            "10Y": history_returns["10Y"] if history_returns["10Y"] is not None else 0,
            "Since Inception": history_returns["Since Inception"] if history_returns["Since Inception"] is not None else 0,
            "Undervaluation (%)": undervaluation if undervaluation is not None else 0,
            "Forecast 5Y": forecast_return if forecast_return is not None else 0,
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
            "P/B": "{:.2f}",
            "1Y": "{:.2%}",
            "3Y": "{:.2%}",
            "5Y": "{:.2%}",
            "10Y": "{:.2%}",
            "Since Inception": "{:.2%}",
            "Undervaluation (%)": "{:.2f}%",
            "Forecast 5Y": "{:.2%}",
            "Score": "{:.2f}"
        }), 
        use_container_width=True
    )
