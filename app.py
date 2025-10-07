# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.greeks.analytical import delta, gamma, theta, vega

# --- App Configuration ---
st.set_page_config(page_title="Option Rolling Analyzer", layout="wide")
st.title("Option Rolling Analytics")

# --- Analysis Functions with Caching for Speed ---

@st.cache_data
def get_stock_price(ticker):
    """Fetches the current stock price."""
    stock = yf.Ticker(ticker)
    return stock.history(period='1d')['Close'].iloc[0]
    
# NEWLY ADDED FUNCTION FOR FUNDAMENTALS
@st.cache_data
def get_stock_fundamentals(ticker_str):
    """Fetches key fundamental metrics for a stock ticker."""
    stock = yf.Ticker(ticker_str)
    info = stock.info

    # Define the metrics we want to display and get them safely from the info dict
    metrics = {
        'Company Name': info.get('shortName', 'N/A'),
        'Sector': info.get('sector', 'N/A'),
        'Market Cap': info.get('marketCap', 'N/A'),
        'Enterprise Value': info.get('enterpriseValue', 'N/A'),
        'Trailing P/E Ratio': info.get('trailingPE', 'N/A'),
        'Forward P/E Ratio': info.get('forwardPE', 'N/A'),
        'Price to Book': info.get('priceToBook', 'N/A'),
        'Total Assets': stock.balance_sheet.loc['Total Assets'].iloc[0] if not stock.balance_sheet.empty else 'N/A',
        'Total Debt': info.get('totalDebt', 'N/A'),
        '52-Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52-Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
        'Beta': info.get('beta', 'N/A')
    }

    # Format large numbers for readability
    for key in ['Market Cap', 'Enterprise Value', 'Total Assets', 'Total Debt']:
        if isinstance(metrics[key], (int, float)):
            metrics[key] = f"${metrics[key]:,}"
    
    # Format ratios to 2 decimal places
    for key in ['Trailing P/E Ratio', 'Forward P/E Ratio', 'Price to Book', 'Beta']:
            if isinstance(metrics[key], (int, float)):
                metrics[key] = f"{metrics[key]:.2f}"

    # Create a DataFrame for clean display
    df_fundamentals = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    return df_fundamentals

@st.cache_data
def analyze_option(ticker, expiration_date, strike_price, stock_price, risk_free_rate, q):
    """Fetches option data and calculates its Greeks."""
    stock = yf.Ticker(ticker)
    opt_chain = stock.option_chain(expiration_date).calls
    option = opt_chain[opt_chain['strike'] == strike_price]
    if option.empty:
        st.warning(f"Option not found for {ticker} {strike_price}C {expiration_date}")
        return None
        
    iv = option['impliedVolatility'].iloc[0]
    price = option['lastPrice'].iloc[0]
    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now() 
    days_to_exp = (exp_date_dt - today).days
    t = days_to_exp / 365.0

    d = delta('c', stock_price, strike_price, t, risk_free_rate, iv, q)
    g = gamma('c', stock_price, strike_price, t, risk_free_rate, iv, q)
    th = theta('c', stock_price, strike_price, t, risk_free_rate, iv, q) / 365
    v = vega('c', stock_price, strike_price, t, risk_free_rate, iv, q) / 100

    return {
        'Expiration': expiration_date,
        'Strike': strike_price,
        'Price': price,
        'Days to Exp': days_to_exp,
        'Implied Vol': iv,
        'Delta': d,
        'Gamma': g,
        'Theta': th,
        'Vega': v,
        'Gamma/Theta Ratio': abs(g / th) if th != 0 else 0
    }

@st.cache_data
def plot_theta_decay(ticker, expiration_date, strike_price, stock_price, risk_free_rate, q):
    """Visualizes the acceleration of Theta as expiration approaches."""
    stock = yf.Ticker(ticker)
    opt_chain = stock.option_chain(expiration_date).calls
    option = opt_chain[opt_chain['strike'] == strike_price]
    if option.empty:
        return go.Figure().update_layout(title_text="Option data not found for Theta Decay plot.")
        
    iv = option['impliedVolatility'].iloc[0]
    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    initial_days = (exp_date_dt - today).days

    days_remaining = np.arange(max(1, initial_days), 1, -1)
    time_to_exp = days_remaining / 365.0
    
    thetas = [theta('c', stock_price, strike_price, t, risk_free_rate, iv, q) / 365 for t in time_to_exp]

    fig = go.Figure(data=go.Scatter(x=-days_remaining, y=thetas, mode='lines', line=dict(color='red')))
    fig.update_layout(
        title=f'<b>Theta Decay Curve for Current Option</b><br>(Assuming Price and IV Remain Constant)',
        xaxis_title='Days Until Expiration',
        yaxis_title='Daily Theta ($)',
        xaxis=dict(autorange="reversed"),
        template='plotly_white'
    )
    return fig


@st.cache_data
def simulate_option_value_with_drift(ticker, expiration_date, strike_price, risk_free_rate, q):
    """Simulates the option's value over time, factoring in historical price drift."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')
    daily_returns = hist['Close'].pct_change().dropna()
    daily_drift = daily_returns.mean()
    stock_price = hist['Close'][-1]
    
    opt_chain = stock.option_chain(expiration_date).calls
    option_data = opt_chain[opt_chain['strike'] == strike_price]
    iv = option_data['impliedVolatility'].iloc[0]

    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    initial_days = (exp_date_dt - today).days
    
    sim_days, sim_option_values = [], []
    
    for days_left in range(initial_days, 1, -1):
        t = days_left / 365.0
        sim_days.append(days_left)
        option_price = black_scholes_merton('c', stock_price, strike_price, t, risk_free_rate, iv, q)
        sim_option_values.append(option_price)
        stock_price *= (1 + daily_drift)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=-np.array(sim_days), y=sim_option_values, mode='lines', name='Projected Option Value', line=dict(color='blue')))
    fig.update_layout(
        title=f'<b>Projected Option Value for Current Option</b><br>(with Daily Price Drift of {daily_drift:.4%})',
        xaxis_title='Days Until Expiration', yaxis_title='Option Price ($)',
        xaxis=dict(autorange="reversed"), template='plotly_white'
    )
    return fig

@st.cache_data
def plot_projected_stock_price(ticker, expiration_date):
    """Plots historical stock price and projects its future path."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')
    daily_returns = hist['Close'].pct_change().dropna()
    daily_drift = daily_returns.mean()
    
    last_price = hist['Close'][-1]
    last_date = hist.index[-1]

    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    days_to_project = (exp_date_dt - today).days

    projected_prices = []
    current_price = last_price
    for _ in range(days_to_project):
        current_price *= (1 + daily_drift)
        projected_prices.append(current_price)

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_project)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Historical Price', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=projected_prices, mode='lines', name=f'Projected Trend ({daily_drift:.4%}/day)', line=dict(color='darkorange', dash='dash')))
    fig.update_layout(
        title=f'<b>{ticker} Historical Price and Projected Trend</b>',
        xaxis_title='Date', yaxis_title='Stock Price ($)',
        template='plotly_white', legend=dict(x=0.01, y=0.98)
    )
    return fig

# --- Sidebar for User Inputs ---
st.sidebar.header("Your Option Position")
TICKER = st.sidebar.text_input("Ticker", "BABA").upper()
CURRENT_EXPIRATION = st.sidebar.text_input("Current Expiration (YYYY-MM-DD)", "2025-12-19")
CURRENT_STRIKE = st.sidebar.number_input("Strike Price", value=220, step=1)
ROLL_TO_EXPIRATION = st.sidebar.text_input("Roll to Expiration (YYYY-MM-DD)", "2026-02-20")
Q = st.sidebar.number_input("Dividend Yield (e.g., 0.01 for 1%)", value=0.0, format="%.4f")
RISK_FREE_RATE = st.sidebar.number_input("Risk-Free Rate (e.g., 0.04 for 4%)", value=0.042, format="%.3f")

st.sidebar.button("Update and Analyze")

# --- Main App Logic ---
with st.spinner('Fetching data and running analysis...'):
    try:
        # --- 1. Display Price and Comparison Table ---
        S = get_stock_price(TICKER)
        st.metric(f"Current {TICKER} Price", f"${S:.2f}")

        st.subheader("Greeks Comparison")
        current_option_stats = analyze_option(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, S, RISK_FREE_RATE, Q)
        roll_to_option_stats = analyze_option(TICKER, ROLL_TO_EXPIRATION, CURRENT_STRIKE, S, RISK_FREE_RATE, Q)

        if current_option_stats and roll_to_option_stats:
            df_compare = pd.DataFrame([current_option_stats, roll_to_option_stats])
            df_compare.index = ['Current Position', 'Rolled Position']
            st.dataframe(df_compare.round(4))

        # --- 2. Display Fundamental Metrics ---
        st.subheader(f"Fundamental Metrics for {TICKER}")
        df_fundamentals = get_stock_fundamentals(TICKER)
        st.dataframe(df_fundamentals)

        # --- 3. Display Charts ---
        st.subheader("Visualizations")
        
        fig_theta = plot_theta_decay(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, S, RISK_FREE_RATE, Q)
        st.plotly_chart(fig_theta, use_container_width=True)
        
        fig_option_value = simulate_option_value_with_drift(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, RISK_FREE_RATE, Q)
        st.plotly_chart(fig_option_value, use_container_width=True)
        
        fig_stock_price = plot_projected_stock_price(TICKER, CURRENT_EXPIRATION)
        st.plotly_chart(fig_stock_price, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred. Please check your inputs. Error: {e}")