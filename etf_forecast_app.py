import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

# Sample sector P/E averages - adjust this with real data
sector_pe_averages = {
    "Technology": 25,
    "Healthcare": 20,
    "Financials": 15,
    # Add more sectors as needed
}

# Map ETF symbols to sectors - Example implementation
etf_sectors = {
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "JNJ": "Healthcare",
    # Add sector mappings for other ETFs
}

# --- Helper Functions ---
def fetch_etf_data(symbol):
    try:
        data = yf.download(symbol, period='max')
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_ytd_return(data):
    current_year = datetime.now().year
    first_of_year = data.loc[str(current_year)].iloc[0]['Close']
    latest_close = data['Close'].iloc[-1]
    return (latest_close / first_of_year) - 1

def calculate_annualized_return(prices, years):
    if len(prices) < 252 * years:
        return None
    start_price = prices[-252 * years]
    end_price = prices[-1]
    return ((end_price / start_price) ** (1 / years)) - 1

def get_historical_returns(data):
    prices = data['Close'].values[::-1]
    returns = {
        "YTD": calculate_ytd_return(data),
        "1Y": calculate_annualized_return(prices, 1),
        "3Y": calculate_annualized_return(prices, 3),
        "5Y": calculate_annualized_return(prices, 5),
        "10Y": calculate_annualized_return(prices, 10),
        "Since Inception": ((prices[-1] / prices[0]) ** (1 / (len(prices) / 252))) - 1
    }
    return returns

def ml_forecast(data, years_ahead=5):
    data = data.dropna().reset_index()
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']]
    y = data['Close']
    model = LinearRegression().fit(X, y)
    future_day = data['Days'].max() + (252 * years_ahead)
    pred_price = model.predict([[future_day]])[0]
    current_price = y.iloc[-1]
    annual_return = ((pred_price / current_price) ** (1 / years_ahead)) - 1
    return annual_return

def compute_undervaluation(symbol, pe_ratio):
    sector = etf_sectors.get(symbol, None)
    if sector is None or pe_ratio is None:
        st.warning(f"No sector or P/E ratio available for {symbol}")
        return None
    
    sector_avg_pe = sector_pe_averages.get(sector, None)
    if sector_avg_pe is None:
        st.warning(f"No average P/E data for sector: {sector}")
        return None
    
    return ((sector_avg_pe - pe_ratio) / sector_avg_pe) * 100

def compute_score(history, future, undervaluation):
    score = 0.0
    if undervaluation is not None:
        score += (undervaluation / 100) * 20
    if history["5Y"] is not None:
        score += history["5Y"] * 50
    if future is not None:
        score += future * 30
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
            # Replace with real P/E ratio fetching mechanism
            simulated_pe_ratio = 18  # Placeholder
            history_returns = get_historical_returns(data)
            forecast_return = ml_forecast(data)
            undervaluation = compute_undervaluation(symbol.strip(), simulated_pe_ratio)
            score = compute_score(history_returns, forecast_return, undervaluation)

            results.append({
                "Symbol": symbol,
                "YTD": history_returns["YTD"],
                "1Y": history_returns["1Y"],
                "3Y": history_returns["3Y"],
                "5Y": history_returns["5Y"],
                "10Y": history_returns["10Y"],
                "Since Inception": history_returns["Since Inception"],
                "Forecast 5Y": forecast_return,
                "Undervaluation %": undervaluation,
                "Score": score
            })
    except Exception as e:
        st.error(f"Persistent error processing {symbol}: {str(e)}")

if results:
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('Score', ascending=False).reset_index(drop=True)
    st.subheader('ETF Rankings by Score')
    st.dataframe(df_sorted.style.format({
        "YTD": "{:.2%}",
        "1Y": "{:.2%}",
        "3Y": "{:.2%}",
        "5Y": "{:.2%}",
        "10Y": "{:.2%}",
        "Since Inception": "{:.2%}",
        "Forecast 5Y": "{:.2%}",
        "Undervaluation %": "{:.2f}%",
        "Score": "{:.2f}"
    }), use_container_width=True)
