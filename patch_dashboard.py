import re
import os

app_path = "dashboard/app.py"

with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Replace column names globally
content = content.replace("q10_5d_return", "q10_return")
content = content.replace("q50_5d_return", "q50_return")
content = content.replace("q90_5d_return", "q90_return")

# 2. Add Sidebar controls
sidebar_old = """    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:10px;">Data Controls</div>', unsafe_allow_html=True)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    lookback = st.slider("History (days)", 30, 365, 90)

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.65rem;color:#4a5568;text-align:center;line-height:1.6;">TFT Model v1.0<br>NIFTY 500 Universe<br>5-Day Quantile Forecast</div>', unsafe_allow_html=True)"""

sidebar_new = """    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:10px;">Data Controls</div>', unsafe_allow_html=True)
    
    selected_horizon = st.selectbox("Time Horizon", ["5d", "10d", "21d", "126d"], index=0)
    
    if st.button("⚡ Sync Live Market Data", use_container_width=True):
        with st.spinner("Fetching live data & predicting..."):
            import subprocess
            subprocess.run(["python", "../Cleaner/macro_fetcher.py"], cwd="dashboard")
            subprocess.run(["python", "../Cleaner/nse_fii_scraper.py"], cwd="dashboard")
            subprocess.run(["python", "../Cleaner/predict.py"], cwd="dashboard")
            st.cache_data.clear()
            st.rerun()
            
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    lookback = st.slider("History (days)", 30, 365, 90)

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.65rem;color:#4a5568;text-align:center;line-height:1.6;">AI Forecasting Engine<br>NIFTY 500 Universe<br>Multi-Horizon Quantile Forecast</div>', unsafe_allow_html=True)"""

content = content.replace(sidebar_old, sidebar_new)

# 3. Update SQL queries in cache functions
query1_old = """@st.cache_data(ttl=60)
def load_latest_predictions():
    return query(\"\"\"
        WITH latest_prices AS (
            SELECT DISTINCT ON (ticker) ticker, close_price AS current_price
            FROM market_data_daily
            ORDER BY ticker, date DESC
        )
        SELECT p.ticker, c.company_name, c.sector,
               p.q10_return, p.q50_return, p.q90_return,
               p.spread, p.signal_strength, p.generated_at,
               COALESCE(m.current_price, 0.0) AS current_price
        FROM predictions p
        LEFT JOIN companies c ON p.ticker = c.ticker
        LEFT JOIN latest_prices m ON p.ticker = m.ticker
        WHERE p.generated_at = (SELECT MAX(generated_at) FROM predictions)
        ORDER BY p.q50_return DESC
    \"\"\")"""

query1_new = """@st.cache_data(ttl=60)
def load_latest_predictions(horizon="5d"):
    return query(\"\"\"
        WITH latest_prices AS (
            SELECT DISTINCT ON (ticker) ticker, close_price AS current_price
            FROM market_data_daily
            ORDER BY ticker, date DESC
        )
        SELECT p.ticker, c.company_name, c.sector,
               p.q10_return, p.q50_return, p.q90_return,
               p.spread, p.signal_strength, p.generated_at,
               COALESCE(m.current_price, 0.0) AS current_price
        FROM predictions p
        LEFT JOIN companies c ON p.ticker = c.ticker
        LEFT JOIN latest_prices m ON p.ticker = m.ticker
        WHERE p.generated_at = (SELECT MAX(generated_at) FROM predictions)
          AND p.horizon = %s
        ORDER BY p.q50_return DESC
    \"\"\", (horizon,))"""
content = content.replace(query1_old, query1_new)

query2_old = """@st.cache_data(ttl=60)
def load_prediction_history(ticker: str, days: int = 30):
    return query(\"\"\"
        SELECT generated_at, q10_return, q50_return, q90_return, spread, signal_strength
        FROM predictions
        WHERE ticker = %s AND generated_at >= NOW() - INTERVAL '%s days'
        ORDER BY generated_at ASC
    \"\"\", (ticker, days))"""

query2_new = """@st.cache_data(ttl=60)
def load_prediction_history(ticker: str, horizon: str = "5d", days: int = 30):
    return query(\"\"\"
        SELECT generated_at, q10_return, q50_return, q90_return, spread, signal_strength
        FROM predictions
        WHERE ticker = %s AND horizon = %s AND generated_at >= NOW() - INTERVAL '%s days'
        ORDER BY generated_at ASC
    \"\"\", (ticker, horizon, days))"""
content = content.replace(query2_old, query2_new)

query3_old = """@st.cache_data(ttl=120)
def load_sector_performance():
    return query(\"\"\"
        SELECT c.sector,
               AVG(p.q50_return) AS avg_return,
               AVG(p.signal_strength) AS avg_signal,
               COUNT(DISTINCT p.ticker) AS stock_count,
               MAX(p.q50_return) AS best_return
        FROM predictions p
        JOIN companies c ON p.ticker = c.ticker
        WHERE p.generated_at = (SELECT MAX(generated_at) FROM predictions)
          AND c.sector IS NOT NULL AND c.sector != ''
        GROUP BY c.sector
        ORDER BY avg_return DESC
    \"\"\")"""

query3_new = """@st.cache_data(ttl=120)
def load_sector_performance(horizon="5d"):
    return query(\"\"\"
        SELECT c.sector,
               AVG(p.q50_return) AS avg_return,
               AVG(p.signal_strength) AS avg_signal,
               COUNT(DISTINCT p.ticker) AS stock_count,
               MAX(p.q50_return) AS best_return
        FROM predictions p
        JOIN companies c ON p.ticker = c.ticker
        WHERE p.generated_at = (SELECT MAX(generated_at) FROM predictions)
          AND p.horizon = %s
          AND c.sector IS NOT NULL AND c.sector != ''
        GROUP BY c.sector
        ORDER BY avg_return DESC
    \"\"\", (horizon,))"""
content = content.replace(query3_old, query3_new)

# 4. Update function calls to pass selected_horizon
content = content.replace("preds_df = load_latest_predictions()", "preds_df = load_latest_predictions(selected_horizon)")
content = content.replace("load_sector_performance()", "load_sector_performance(selected_horizon)")
content = content.replace("load_prediction_history(ticker_sel, lookback)", "load_prediction_history(ticker_sel, selected_horizon, lookback)")

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Dashboard app.py patched successfully!")
