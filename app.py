import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.greeks.analytical import delta, gamma, theta, vega
from statsmodels.tsa.api import Holt
from arch import arch_model

# --- App Configuration ---
st.set_page_config(page_title="Option Rolling Analyzer", layout="wide")
st.title("Option Rolling Analytics")

# --- Analysis Functions with Caching for Speed ---

@st.cache_data
def get_stock_price(ticker):
    """Fetches the current stock price."""
    stock = yf.Ticker(ticker)
    return stock.history(period='1d')['Close'].iloc[0]
    
# MODIFIED FUNCTION FOR HORIZONTAL FUNDAMENTALS
@st.cache_data
def get_stock_fundamentals(ticker_str):
    """Fetches key fundamental metrics for a stock ticker."""
    stock = yf.Ticker(ticker_str)
    info = stock.info
    
    # Safely get dates and format them
    earnings_date_ts = info.get('earningsTimestamp')
    ex_dividend_date_ts = info.get('exDividendDate')
    
    earnings_date = pd.to_datetime(earnings_date_ts, unit='s').strftime('%Y-%m-%d') if earnings_date_ts else 'N/A'
    ex_dividend_date = pd.to_datetime(ex_dividend_date_ts, unit='s').strftime('%Y-%m-%d') if ex_dividend_date_ts else 'N/A'

    metrics = {
        'Company Name': info.get('shortName', 'N/A'),
        'Market Cap': info.get('marketCap', 'N/A'),
        'Enterprise Value': info.get('enterpriseValue', 'N/A'),
        'Trailing P/E': info.get('trailingPE', 'N/A'),
        'Forward P/E': info.get('forwardPE', 'N/A'),
        'Price to Book': info.get('priceToBook', 'N/A'),
        'Dividend Yield': info.get('dividendYield', 'N/A'),
        'Beta': info.get('beta', 'N/A'),
        'Earnings Date': earnings_date,
        'Ex-Dividend Date': ex_dividend_date,
    }
    # Format large numbers into billions for readability
    for key in ['Market Cap', 'Enterprise Value']:
        if isinstance(metrics[key], (int, float)):
            value_in_billions = metrics[key] / 1_000_000_000
            metrics[key] = f"${value_in_billions:.2f}B"
    # Format ratios
    for key in ['Trailing P/E', 'Forward P/E', 'Price to Book', 'Beta']:
        if isinstance(metrics[key], (int, float)):
            metrics[key] = f"{metrics[key]:.2f}"
    # Format dividend yield as percentage
    if isinstance(metrics['Dividend Yield'], (int, float)):
        metrics['Dividend Yield'] = f"{metrics['Dividend Yield'] * 100:.2f}%"
        
    # Create a horizontal DataFrame
    df_fundamentals = pd.DataFrame([metrics])
    return df_fundamentals.set_index('Company Name')


@st.cache_data
def forecast_stock_price(ticker, days_to_project):
    """Forecasts stock price trend with Holt's model and volatility with GARCH."""
    hist = yf.Ticker(ticker).history(period='2y')['Close']
    holt_model = Holt(hist, initialization_method="estimated").fit()
    forecast = holt_model.forecast(days_to_project)
    returns = hist.pct_change().dropna() * 100
    garch_model = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
    garch_forecasts = garch_model.forecast(horizon=days_to_project)
    cond_vol = np.sqrt(garch_forecasts.variance.values[-1, :]) / 100
    forecast_std_dev = cond_vol * np.sqrt(np.arange(1, days_to_project + 1))
    upper_bound = forecast * (1 + forecast_std_dev)
    lower_bound = forecast * (1 - forecast_std_dev)
    future_dates = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=days_to_project)
    return hist, forecast, upper_bound, lower_bound, future_dates

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
    return {'Expiration': expiration_date, 'Strike': strike_price, 'Price': price, 'Days to Exp': days_to_exp,
            'Implied Vol': iv, 'Delta': d, 'Gamma': g, 'Theta': th, 'Vega': v,
            'Gamma/Theta Ratio': abs(g / th) if th != 0 else 0}

@st.cache_data
def plot_theta_decay(ticker, expiration_date, strike_price, stock_price, risk_free_rate, q):
    """Visualizes the static acceleration of Theta as expiration approaches."""
    stock = yf.Ticker(ticker)
    opt_chain = stock.option_chain(expiration_date).calls
    option = opt_chain[opt_chain['strike'] == strike_price]
    if option.empty: return go.Figure().update_layout(title_text="Option data not found for static Theta Decay plot.")
    iv = option['impliedVolatility'].iloc[0]
    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    initial_days = (exp_date_dt - today).days
    days_remaining = np.arange(max(1, initial_days), 1, -1)
    time_to_exp = days_remaining / 365.0
    thetas = [theta('c', stock_price, strike_price, t, risk_free_rate, iv, q) / 365 for t in time_to_exp]
    fig = go.Figure(data=go.Scatter(x=-days_remaining, y=thetas, mode='lines', line=dict(color='red')))
    fig.update_layout(title=f'<b>Static Theta Decay Curve for Current Option</b><br>(Assuming Constant Price and IV)',
                      xaxis_title='Days Until Expiration', yaxis_title='Daily Theta ($)',
                      xaxis=dict(autorange="reversed"), template='plotly_white')
    return fig

# NEW DYNAMIC THETA PLOT
@st.cache_data
def plot_dynamic_theta_decay(ticker, expiration_date, strike_price, risk_free_rate, q, forecast_path):
    """Simulates and plots theta decay along the forecasted price path."""
    stock = yf.Ticker(ticker)
    opt_chain = stock.option_chain(expiration_date).calls
    option_data = opt_chain[opt_chain['strike'] == strike_price]
    if option_data.empty: return go.Figure().update_layout(title_text="Option data not found for dynamic Theta Decay plot.")
    iv = option_data['impliedVolatility'].iloc[0]
    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    initial_days = (exp_date_dt - today).days
    
    sim_days, sim_thetas = [], []
    
    for i, days_left in enumerate(range(initial_days, 1, -1)):
        if i >= len(forecast_path): break
        t = days_left / 365.0
        stock_price = forecast_path.iloc[i]
        sim_days.append(days_left)
        daily_theta = theta('c', stock_price, strike_price, t, risk_free_rate, iv, q) / 365
        sim_thetas.append(daily_theta)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=-np.array(sim_days), y=sim_thetas, mode='lines', name='Projected Theta', line=dict(color='purple')))
    fig.update_layout(title=f'<b>Dynamic Theta Decay along Forecasted Path</b><br>(For Current Option)',
                      xaxis_title='Days Until Expiration', yaxis_title='Daily Theta ($)',
                      xaxis=dict(autorange="reversed"), template='plotly_white')
    return fig


@st.cache_data
def simulate_option_value_with_forecast(ticker, expiration_date, strike_price, risk_free_rate, q, forecast_path):
    """Simulates the option's value over the forecast path."""
    stock = yf.Ticker(ticker)
    opt_chain = stock.option_chain(expiration_date).calls
    option_data = opt_chain[opt_chain['strike'] == strike_price]
    iv = option_data['impliedVolatility'].iloc[0]
    exp_date_dt = pd.to_datetime(expiration_date)
    today = pd.Timestamp.now()
    initial_days = (exp_date_dt - today).days
    sim_days, sim_option_values = [], []
    for i, days_left in enumerate(range(initial_days, 1, -1)):
        if i >= len(forecast_path): break
        t = days_left / 365.0
        stock_price = forecast_path.iloc[i]
        sim_days.append(days_left)
        option_price = black_scholes_merton('c', stock_price, strike_price, t, risk_free_rate, iv, q)
        sim_option_values.append(option_price)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=-np.array(sim_days), y=sim_option_values, mode='lines', name='Projected Option Value', line=dict(color='blue')))
    fig.update_layout(title=f'<b>Projected Option Value based on Trend Forecast</b>',
                      xaxis_title='Days Until Expiration', yaxis_title='Option Price ($)',
                      xaxis=dict(autorange="reversed"), template='plotly_white')
    return fig

# MODIFIED FUNCTION TO ADD LINES
@st.cache_data
def plot_projected_stock_price(ticker, current_expiration, roll_to_expiration, strike_price):
    """Plots historical price and the Holt/GARCH forecast with key levels."""
    today = pd.Timestamp.now()
    days_to_project = (pd.to_datetime(current_expiration) - today).days
    hist, forecast, upper_bound, lower_bound, future_dates = forecast_stock_price(ticker, days_to_project)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, mode='lines', name='Historical Price', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name="Holt's Trend Forecast", line=dict(color='darkorange')))
    fig.add_trace(go.Scatter(x=future_dates, y=upper_bound, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_dates, y=lower_bound, mode='lines', line=dict(width=0), name='GARCH Volatility Cone', fill='tonexty', fillcolor='rgba(255,165,0,0.2)'))
    
    # Add vertical lines for expiration dates
    fig.add_vline(x=pd.to_datetime(current_expiration), line_width=2, line_dash="dash", line_color="green",
                  annotation_text="Current Exp", annotation_position="top right")
    fig.add_vline(x=pd.to_datetime(roll_to_expiration), line_width=2, line_dash="dash", line_color="red",
                  annotation_text="Roll-to Exp", annotation_position="top right")

    # Add horizontal line for strike price
    fig.add_hline(y=strike_price, line_width=2, line_dash="dash", line_color="purple",
                  annotation_text=f"Strike ${strike_price}", annotation_position="bottom right")

    fig.update_layout(title=f'<b>{ticker} Price Forecast with Holt Trend and GARCH Volatility</b>',
                      xaxis_title='Date', yaxis_title='Stock Price ($)',
                      template='plotly_white', legend=dict(x=0.01, y=0.98))
    return fig, forecast

# --- Sidebar for User Inputs ---
st.sidebar.header("Your Option Position")
TICKER = st.sidebar.text_input("Ticker", "BABA").upper()

# --- Fetch default dividend yield for the sidebar ---
try:
    ticker_info = yf.Ticker(TICKER).info
    # yfinance provides yield as a float (e.g., 0.02 for 2%)
    default_div_yield = ticker_info.get('dividendYield', 0.0)
    if default_div_yield is None:  # Handle case where yield is None
        default_div_yield = 0.0
except Exception:
    default_div_yield = 0.0 # Default if ticker is invalid or API fails

CURRENT_EXPIRATION = st.sidebar.text_input("Current Expiration (YYYY-MM-DD)", "2025-12-19")
CURRENT_STRIKE = st.sidebar.number_input("Strike Price", value=220, step=1)
ROLL_TO_EXPIRATION = st.sidebar.text_input("Roll to Expiration (YYYY-MM-DD)", "2026-02-20")
Q = st.sidebar.number_input("Dividend Yield (e.g., 0.01 for 1%)", value=float(default_div_yield), format="%.4f")
RISK_FREE_RATE = st.sidebar.number_input("Risk-Free Rate (e.g., 0.04 for 4%)", value=0.042, format="%.3f")

st.sidebar.button("Update and Analyze")

# --- Main App Logic ---
with st.spinner('Fetching data and running advanced models... This may take a moment.'):
    try:
        S = get_stock_price(TICKER)
        st.metric(f"Current {TICKER} Price", f"${S:.2f}")

        st.subheader("Greeks Comparison")
        current_option_stats = analyze_option(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, S, RISK_FREE_RATE, Q)
        roll_to_option_stats = analyze_option(TICKER, ROLL_TO_EXPIRATION, CURRENT_STRIKE, S, RISK_FREE_RATE, Q)
        if current_option_stats and roll_to_option_stats:
            df_compare = pd.DataFrame([current_option_stats, roll_to_option_stats])
            df_compare.index = ['Current Position', 'Rolled Position']
            st.dataframe(df_compare.round(4))

        st.subheader(f"Fundamental Metrics for {TICKER}")
        df_fundamentals = get_stock_fundamentals(TICKER)
        st.dataframe(df_fundamentals)

        st.subheader("Visualizations")
        # Forecast now extends to the rolled position's maturity and includes lines
        fig_stock_price, forecast_path = plot_projected_stock_price(TICKER, CURRENT_EXPIRATION, ROLL_TO_EXPIRATION, CURRENT_STRIKE)
        st.plotly_chart(fig_stock_price, use_container_width=True)

        # Slice the forecast path for simulations of the current option
        days_current_exp = (pd.to_datetime(CURRENT_EXPIRATION) - pd.Timestamp.now()).days
        forecast_path_current = forecast_path[:days_current_exp]
        
        fig_option_value = simulate_option_value_with_forecast(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, RISK_FREE_RATE, Q, forecast_path_current)
        st.plotly_chart(fig_option_value, use_container_width=True)
        
        # Display the new dynamic theta plot
        fig_dynamic_theta = plot_dynamic_theta_decay(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, RISK_FREE_RATE, Q, forecast_path_current)
        st.plotly_chart(fig_dynamic_theta, use_container_width=True)
        
        fig_static_theta = plot_theta_decay(TICKER, CURRENT_EXPIRATION, CURRENT_STRIKE, S, RISK_FREE_RATE, Q)
        st.plotly_chart(fig_static_theta, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred. Please check your inputs. Error: {e}")
