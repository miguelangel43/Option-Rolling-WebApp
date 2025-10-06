# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from py_vollib.black_scholes_merton.greeks.analytical import delta, gamma, theta, vega

# --- App Configuration ---
st.set_page_config(page_title="Option Rolling Analyzer", layout="wide")
st.title("Option Rolling Analytics")

# --- Your Analysis Functions (Copied from the notebook) ---
# (Copy and paste all your functions here: get_stock_price, analyze_option, 
#  simulate_option_value_with_drift, plot_projected_stock_price, etc.)
def get_stock_price(ticker):
    """Fetches the current stock price."""
    stock = yf.Ticker(ticker)
    return stock.history(period='1d')['Close'].iloc[0]

def analyze_option(ticker, expiration_date, strike_price, stock_price, risk_free_rate):
    """Fetches option data and calculates its Greeks."""
    stock = yf.Ticker(ticker)
    opt_chain = stock.option_chain(expiration_date).calls
    
    # Find our specific option
    option = opt_chain[opt_chain['strike'] == strike_price]
    if option.empty:
        print(f"Option not found for {ticker} {strike_price}C {expiration_date}")
        return None
        
    # Extract data
    iv = option['impliedVolatility'].iloc[0]
    price = option['lastPrice'].iloc[0]
    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    days_to_exp = (exp_date_dt - today).days
    t = days_to_exp / 365.0 # Time to expiration in years
    q = 0.0 # Dividend yield, assumed 0 for simplicity

    # Calculate Greeks using py_vollib
    d = delta('c', stock_price, strike_price, t, risk_free_rate, iv, q)
    g = gamma('c', stock_price, strike_price, t, risk_free_rate, iv, q)
    th = theta('c', stock_price, strike_price, t, risk_free_rate, iv, q) / 365 # Daily theta
    v = vega('c', stock_price, strike_price, t, risk_free_rate, iv, q) / 100 # Vega per 1% change

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
        'Gamma/Theta Ratio': abs(g / th) if th !=0 else 0
    }

def simulate_option_value_with_drift(ticker, expiration_date, strike_price, risk_free_rate, q):
    """
    Simulates the option's value over time, factoring in historical price drift
    to project the underlying stock price.
    """
    # --- 1. Calculate Historical Drift ---
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')
    daily_returns = hist['Close'].pct_change().dropna()
    daily_drift = daily_returns.mean()
    stock_price = hist['Close'][-1]
    
    print(f"Using initial {ticker} price: ${stock_price:.2f}")
    print(f"Calculated average daily drift (last 6 months): {daily_drift:.4%}")

    # --- 2. Set up Simulation ---
    opt_chain = stock.option_chain(expiration_date).calls
    option_data = opt_chain[opt_chain['strike'] == strike_price]
    iv = option_data['impliedVolatility'].iloc[0]

    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    initial_days = (exp_date_dt - today).days
    
    sim_days = []
    sim_stock_prices = []
    sim_option_values = []
    
    # --- 3. Run Simulation Loop ---
    for days_left in range(initial_days, 1, -1):
        t = days_left / 365.0
        
        # Store current values
        sim_days.append(days_left)
        sim_stock_prices.append(stock_price)
        
        # Calculate option price for this day
        option_price = black_scholes_merton('c', stock_price, strike_price, t, risk_free_rate, iv, q)
        sim_option_values.append(option_price)
        
        # Increment stock price by the daily drift for the next day
        stock_price *= (1 + daily_drift)
        
    # --- 4. Plot Results ---
    fig = go.Figure()

    # Add Option Value trace
    fig.add_trace(go.Scatter(
        x=-np.array(sim_days), 
        y=sim_option_values, 
        mode='lines', 
        name='Projected Option Value',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title=f'<b>Projected Option Value for {ticker} {strike_price}C</b><br>(with Daily Price Drift of {daily_drift:.4%})',
        xaxis_title='Days Until Expiration',
        yaxis_title='Option Price ($)',
        xaxis=dict(autorange="reversed"),
        template='plotly_white'
    )
    fig.show()

def plot_projected_stock_price(ticker, expiration_date, q):
    """
    Plots the historical stock price and projects its future path based on
    the 6-month historical daily drift.
    """
    # --- 1. Get Historical Data & Calculate Drift ---
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')
    daily_returns = hist['Close'].pct_change().dropna()
    daily_drift = daily_returns.mean()
    
    last_price = hist['Close'][-1]
    last_date = hist.index[-1]

    # --- 2. Project Future Prices ---
    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    days_to_project = (exp_date_dt - today).days

    projected_prices = []
    current_price = last_price
    for _ in range(days_to_project):
        current_price *= (1 + daily_drift)
        projected_prices.append(current_price)

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_project)

    # --- 3. Plot Results ---
    fig = go.Figure()

    # Historical Price Trace
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='royalblue', width=2)
    ))

    # Projected Price Trace
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=projected_prices,
        mode='lines',
        name=f'Projected Trend ({daily_drift:.4%}/day)',
        line=dict(color='darkorange', dash='dash')
    ))

    fig.update_layout(
        title=f'<b>{ticker} Historical Price and Projected Trend</b>',
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        template='plotly_white',
        legend=dict(x=0.01, y=0.98)
    )

    fig.show()

# --- Sidebar for User Inputs ---
st.sidebar.header("Your Option Position")
TICKER = st.sidebar.text_input("Ticker", "AAPL").upper()
CURRENT_EXPIRATION = st.sidebar.text_input("Current Expiration (YYYY-MM-DD)", "2025-11-21")
CURRENT_STRIKE = st.sidebar.number_input("Strike Price", value=175)
ROLL_TO_EXPIRATION = st.sidebar.text_input("Roll to Expiration (YYYY-MM-DD)", "2026-01-16")
Q = st.sidebar.number_input("Dividend Yield (as decimal)", value=0.0)
RISK_FREE_RATE = st.sidebar.number_input("Risk-Free Rate (as decimal)", value=0.04)

# --- Main App Logic ---
if st.sidebar.button("Update and Analyze"):
    try:
        # --- 1. Display Comparison Table ---
        st.subheader("Greeks Comparison")
        S = get_stock_price(TICKER)
        st.metric(f"Current {TICKER} Price", f"${S:.2f}")

        current_option_stats = analyze_option(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, S, RISK_FREE_RATE)
        roll_to_option_stats = analyze_option(TICKER, ROLL_TO_EXPIRATION, CURRENT_STRIKE, S, RISK_FREE_RATE)

        if current_option_stats and roll_to_option_stats:
            df_compare = pd.DataFrame([current_option_stats, roll_to_option_stats])
            df_compare.index = ['Current Position', 'Rolled Position']
            st.dataframe(df_compare.round(4))

        # --- 2. Display Charts ---
        st.subheader("Visualizations")
        
        # Option Value Projection
        fig_option_value = simulate_option_value_with_drift(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, RISK_FREE_RATE)
        st.plotly_chart(fig_option_value, use_container_width=True)
        
        # Stock Price Projection
        fig_stock_price = plot_projected_stock_price(TICKER, CURRENT_EXPIRATION)
        st.plotly_chart(fig_stock_price, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Enter your option details in the sidebar and click 'Update and Analyze'.")

