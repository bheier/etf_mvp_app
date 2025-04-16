import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

# Alpha Vantage setup
api_key = 'your_alpha_vantage_api_key'  # Replace with your API key
ts = TimeSeries(key=api_key, output_format='pandas')

# --- Helper Functions ---
def fetch_etf_data(symbol):
    try:
        # Fetch daily historical data
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_annualized_return(prices, years):
    if len(prices) < 252 * years:
        return None
    start_price = prices[-252 * years]
    end_price = prices[-1]
    return ((end_price / start_price) ** (1 / years)) - 1

def get_historical_returns(data):
    data = data['4. close']
    prices = data.values[::-1]  # Reverse to simulate oldest to newest order
    returns = {
        "1Y": calculate_annualized_return(prices, 1),
        "3Y": calculate_annualized_return(prices, 3),
        "5Y": calculate_annualized_return(prices, 5),
        "10Y": calculate_annualized_return(prices, 10),
        "Since Inception": ((prices[-1] / prices[0]) ** (1 / (len(prices) / 252))) - 1
    }
    return returns

def ml_forecast(data, years_ahead=5):
    data = data['4. close'].dropna().reset_index()
    data['Days'] = (data['date'] - data['date'].min()).dt.days
    X = data[['Days']]
    y = data['4. close']
    model = LinearRegression().fit(X, y)
    future_day = data['Days'].max() + (252 * years_ahead)
    pred_price = model.predict([[future_day]])[0]
    current_price = y.iloc[-1]
    annual_return = ((pred_price / current_price) ** (1 / years_ahead)) - 1
    return annual_return

def compute_score(valuation, history, future):
    score = 0
    if valuation["pe_ratio"] is not None:
        score += 1 / valuation["pe_ratio"]
    if valuation["pb_ratio"] is not None:
        score += 1 / valuation["pb_ratio"]
    if history["5Y"] is not None:
        score += history["5Y"] * 100
    if future is not None:
        score += future * 100
    return score

# --- Streamlit UI ---
st.set_page_config(page_title='ETF Undervaluation & Forecast App', layout='wide')
st.title('ETF Undervaluation & Forecast App')

etf_symbols = st.text_input('Enter comma-separated ETF symbols:', value='MSFT,GOOGL').upper().split(',')

results = []
for symbol in etf_symbols:
    try:
        data = fetch_etf_data(symbol.strip())
        if data is not None:
            valuation = {
                "pe_ratio": None,  # Adjust logic for P/E and P/B ratios if using such metrics from another source
                "pb_ratio": None
            }
            history_returns = get_historical_returns(data)
            forecast_return = ml_forecast(data)
            score = compute_score(valuation, history_returns, forecast_return)

            results.append({
                "Symbol": symbol,
                "P/E": valuation["pe_ratio"],
                "P/B": valuation["pb_ratio"],
                "1Y": history_returns["1Y"],
                "3Y": history_returns["3Y"],
                "5Y": history_returns["5Y"],
                "10Y": history_returns["10Y"],
                "Since Inception": history_returns["Since Inception"],
                "Forecast 5Y": forecast_return,
                "Score": score
            })
    except Exception as e:
        st.error(f"Persistent error processing {symbol}: {str(e)}")

if results:
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('Score', ascending=False).reset_index(drop=True)
    st.subheader('ETF Rankings by Score')
    st.dataframe(df_sorted.style.format({
        "P/E": "{:.2f}",
        "P/B": "{:.2f}",
        "1Y": "{:.2%}",
        "3Y": "{:.2%}",
        "5Y": "{:.2%}",
        "10Y": "{:.2%}",
        "Since Inception": "{:.2%}",
        "Forecast 5Y": "{:.2%}",
        "Score": "{:.2f}"
    }), use_container_width=True)
