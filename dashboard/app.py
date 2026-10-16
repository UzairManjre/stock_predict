"""
AlphaEdge Dashboard — TFT Stock Prediction Platform
A premium Streamlit UI for monitoring AI-driven NIFTY stock predictions.
"""

import os
import sys
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import time

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaEdge — AI Stock Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load .env ────────────────────────────────────────────────────────────────
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
if os.path.exists(_env_path):
    with open(_env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

DB_CFG = dict(
    host=os.environ.get("DB_HOST", "localhost"),
    port=os.environ.get("DB_PORT", "5433"),
    database=os.environ.get("DB_NAME", "stock_predict"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "postgres"),
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #0f1629;
    --bg-card: #141d35;
    --bg-card-hover: #1a2540;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --text-primary: #f0f4ff;
    --text-secondary: #8896b3;
    --text-muted: #4a5568;
    --border: #1e2d4a;
    --glow-blue: rgba(59, 130, 246, 0.15);
    --glow-green: rgba(16, 185, 129, 0.15);
    --glow-red: rgba(239, 68, 68, 0.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0f1629 100%) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}
.metric-card:hover {
    border-color: var(--accent-blue);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px var(--glow-blue);
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-delta {
    font-size: 0.75rem;
    margin-top: 6px;
    font-weight: 500;
}
.metric-green { color: var(--accent-green); }
.metric-red { color: var(--accent-red); }
.metric-blue { color: var(--accent-blue); }
.metric-amber { color: var(--accent-amber); }
.metric-cyan { color: var(--accent-cyan); }
.metric-purple { color: var(--accent-purple); }

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.section-header h2 {
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-primary);
    margin: 0;
}
.section-badge {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Signal badges */
.signal-strong-buy {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.signal-buy {
    background: linear-gradient(135deg, #065f46, #059669);
    color: #6ee7b7;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.signal-sell {
    background: linear-gradient(135deg, #7f1d1d, #dc2626);
    color: #fca5a5;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.signal-neutral {
    background: rgba(100,116,139,0.2);
    color: #94a3b8;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Top-nav logo */
.logo-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 32px;
    padding: 16px 20px;
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.05));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
}
.logo-icon {
    font-size: 2rem;
    line-height: 1;
}
.logo-text {
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.logo-sub {
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}

/* Data tables */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse 2s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Ticker pill */
.ticker-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    color: #93c5fd;
    padding: 2px 8px;
    border-radius: 5px;
}

/* Hide streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Input widgets */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── DB Connection ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=60)
def get_connection():
    try:
        return psycopg2.connect(**DB_CFG)
    except Exception as e:
        return None


def query(sql, params=None, ttl=30):
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        conn.rollback()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"DB Error: {e}")
        return pd.DataFrame()


# ─── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_latest_predictions(horizon="5d"):
    return query("""
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
    """, (horizon,))


@st.cache_data(ttl=60)
def load_prediction_history(ticker: str, horizon: str = "5d", days: int = 30):
    return query("""
        SELECT generated_at, q10_return, q50_return, q90_return, spread, signal_strength
        FROM predictions
        WHERE ticker = %s AND horizon = %s AND generated_at >= NOW() - INTERVAL '%s days'
        ORDER BY generated_at ASC
    """, (ticker, horizon, days))


@st.cache_data(ttl=60)
def load_price_history(ticker: str, days: int = 90):
    return query("""
        SELECT date, open_price, high_price, low_price, close_price, adj_close, volume
        FROM market_data_daily
        WHERE ticker = %s AND date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY date ASC
    """, (ticker, days))


@st.cache_data(ttl=60)
def load_sentiment_history(ticker: str, days: int = 60):
    return query("""
        SELECT date, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed
        FROM daily_sentiment_aggregates
        WHERE ticker = %s AND date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY date ASC
    """, (ticker, days))


@st.cache_data(ttl=120)
def load_sector_performance(horizon="5d"):
    return query("""
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
    """, (horizon,))


@st.cache_data(ttl=60)
def load_recent_reports(limit: int = 20):
    return query("""
        SELECT fr.ticker, c.company_name, fr.filing_date, fr.headline,
               fr.report_type, fr.received_time
        FROM financial_reports fr
        LEFT JOIN companies c ON fr.ticker = c.ticker
        ORDER BY fr.received_time DESC NULLS LAST
        LIMIT %s
    """, (limit,))


@st.cache_data(ttl=60)
def load_db_stats():
    return query("""
        SELECT
            (SELECT COUNT(*) FROM companies WHERE is_active) AS active_stocks,
            (SELECT COUNT(*) FROM predictions WHERE generated_at = (SELECT MAX(generated_at) FROM predictions)) AS latest_preds,
            (SELECT MAX(generated_at) FROM predictions) AS last_prediction_time,
            (SELECT MAX(date) FROM market_data_daily) AS last_price_date,
            (SELECT COUNT(*) FROM financial_reports) AS total_reports,
            (SELECT COUNT(*) FROM sentiment_chunks) AS total_chunks
    """)


# ─── Technical Indicator calculations ─────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    e1 = series.ewm(span=fast, adjust=False).mean()
    e2 = series.ewm(span=slow, adjust=False).mean()
    macd = e1 - e2
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def compute_bbands(series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std


def signal_label(q50, signal_strength):
    if q50 > 0.02 and signal_strength > 1.5:
        return "STRONG BUY", "signal-strong-buy"
    elif q50 > 0.005:
        return "BUY", "signal-buy"
    elif q50 < -0.01:
        return "SELL", "signal-sell"
    else:
        return "NEUTRAL", "signal-neutral"


def calculate_target_metrics(current_price, q50_return, spread):
    current_price = float(current_price or 0.0)
    q50_return = float(q50_return or 0.0)
    spread = float(spread or 0.0)

    # 1. Target Alpha (Expected Outperformance)
    alpha_edge = q50_return * 100
    safe_alpha = alpha_edge * 0.8 if alpha_edge > 0 else alpha_edge

    # 3. Calibrated confidence score using piecewise linear interpolation
    anchors = [
        (0.060, 95.0),  # P0
        (0.094, 80.0),  # P25
        (0.113, 65.0),  # P50
        (0.131, 55.0),  # P75
        (0.158, 47.0),  # P90
        (0.315, 35.0),  # P100
    ]

    # Clamp spread to the anchor range
    if spread <= anchors[0][0]:
        confidence_score = anchors[0][1]
    elif spread >= anchors[-1][0]:
        confidence_score = anchors[-1][1]
    else:
        for i in range(len(anchors) - 1):
            s0, c0 = anchors[i]
            s1, c1 = anchors[i + 1]
            if s0 <= spread <= s1:
                t = (spread - s0) / (s1 - s0)
                confidence_score = c0 + t * (c1 - c0)
                break

    # 4. Confidence rating and label
    if confidence_score >= 80.0:
        conf_label = "High"
        conf_class = "metric-green"
    elif confidence_score >= 60.0:
        conf_label = "Moderate"
        conf_class = "metric-amber"
    else:
        conf_label = "Low"
        conf_class = "metric-red"

    return {
        "current_price": current_price,
        "raw_target": alpha_edge,
        "safe_target": safe_alpha,
        "confidence_score": confidence_score,
        "confidence_label": conf_label,
        "confidence_class": conf_class
    }


# ─── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(14,20,40,0.6)',
    font=dict(family='Inter', color='#8896b3', size=11),
    margin=dict(l=10, r=10, t=35, b=10),
    legend=dict(
        bgcolor='rgba(10,14,26,0.8)',
        bordercolor='rgba(59,130,246,0.2)',
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        linecolor='rgba(255,255,255,0.08)',
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        linecolor='rgba(255,255,255,0.08)',
        tickfont=dict(size=10),
    ),
)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-bar">
        <div class="logo-icon">⚡</div>
        <div>
            <div class="logo-text">AlphaEdge</div>
            <div class="logo-sub">AI Stock Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["🏠  Overview", "📈  Top Picks", "🔍  Stock Deep Dive", "🌐  Sector Analysis", "📰  News & Sentiment", "🎯  Accuracy", "⚙️  Pipeline Status"],
        label_visibility="collapsed",
    )

    st.markdown("---")
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
    st.markdown(f'<div style="font-size:0.65rem;color:#4a5568;text-align:center;line-height:1.6;">AI Forecasting Engine<br>NIFTY 500 Universe<br>Multi-Horizon Quantile Forecast</div>', unsafe_allow_html=True)

if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()


# ─── Page: Overview ───────────────────────────────────────────────────────────
if nav == "🏠  Overview":
    st.markdown('<div style="font-size:1.6rem;font-weight:800;margin-bottom:4px;">Market Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8896b3;font-size:0.85rem;margin-bottom:28px;">Real-time AI predictions powered by Temporal Fusion Transformer</div>', unsafe_allow_html=True)

    # Stats row
    stats_df = load_db_stats()
    preds_df = load_latest_predictions(selected_horizon)

    bullish = len(preds_df[preds_df['q50_return'] > 0.005]) if not preds_df.empty else 0
    bearish = len(preds_df[preds_df['q50_return'] < -0.005]) if not preds_df.empty else 0
    avg_return = preds_df['q50_return'].mean() * 100 if not preds_df.empty else 0
    top_return = preds_df['q50_return'].max() * 100 if not preds_df.empty else 0

    last_pred = "N/A"
    if not stats_df.empty and stats_df['last_prediction_time'].iloc[0]:
        lp = pd.to_datetime(stats_df['last_prediction_time'].iloc[0])
        last_pred = lp.strftime("%d %b, %H:%M")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        active = int(stats_df['active_stocks'].iloc[0]) if not stats_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Stocks</div>
            <div class="metric-value metric-blue">{active}</div>
            <div class="metric-delta" style="color:#8896b3;">NIFTY Universe</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Bullish Signals</div>
            <div class="metric-value metric-green">{bullish}</div>
            <div class="metric-delta metric-green">+Q50 &gt; 0.5%</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Bearish Signals</div>
            <div class="metric-value metric-red">{bearish}</div>
            <div class="metric-delta metric-red">-Q50 &lt; -0.5%</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        color = "metric-green" if avg_return > 0 else "metric-red"
        sign = "+" if avg_return > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg 5D Forecast</div>
            <div class="metric-value {color}">{sign}{avg_return:.2f}%</div>
            <div class="metric-delta" style="color:#8896b3;">Median (Q50)</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Opportunity</div>
            <div class="metric-value metric-cyan">+{top_return:.2f}%</div>
            <div class="metric-delta" style="color:#8896b3;">Last: {last_pred}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two columns: prediction distribution + sector heatmap
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header"><h2>Return Distribution</h2><span class="section-badge">Live</span></div>', unsafe_allow_html=True)
        if not preds_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=preds_df['q50_return'] * 100,
                nbinsx=40,
                name='Q50 Return',
                marker=dict(
                    color=preds_df['q50_return'].apply(
                        lambda x: '#10b981' if x > 0 else '#ef4444'
                    ),
                    line=dict(width=0),
                ),
                opacity=0.85,
            ))
            fig.add_vline(x=0, line_color='rgba(255,255,255,0.3)', line_dash='dash', line_width=1)
            fig.add_vline(x=avg_return, line_color='#3b82f6', line_dash='dot', line_width=1.5,
                         annotation_text=f"Avg {avg_return:.2f}%", annotation_font_color='#3b82f6')
            fig.update_layout(
                **PLOT_LAYOUT,
                height=280,
                showlegend=False,
                xaxis_title='Predicted Return (%)',
                yaxis_title='Stocks',
                title=dict(text='All Tickers — Forecast Distribution', font=dict(size=12, color='#8896b3')),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No predictions available. Run the pipeline first.")

    with col_b:
        st.markdown('<div class="section-header"><h2>Sector Performance</h2><span class="section-badge">Live</span></div>', unsafe_allow_html=True)
        sector_df = load_sector_performance(selected_horizon)
        if not sector_df.empty:
            sector_df['avg_return_pct'] = sector_df['avg_return'] * 100
            sector_df['color'] = sector_df['avg_return_pct'].apply(
                lambda x: '#10b981' if x > 0 else '#ef4444'
            )
            fig = go.Figure(go.Bar(
                x=sector_df['avg_return_pct'],
                y=sector_df['sector'],
                orientation='h',
                marker=dict(
                    color=sector_df['color'],
                    line=dict(width=0),
                ),
                text=sector_df['avg_return_pct'].apply(lambda x: f"{x:+.2f}%"),
                textposition='outside',
                textfont=dict(size=10, color='#8896b3'),
            ))
            fig.add_vline(x=0, line_color='rgba(255,255,255,0.15)', line_width=1)
            fig.update_layout(
                **PLOT_LAYOUT,
                height=280,
                showlegend=False,
                xaxis_title='Avg Predicted Return (%)',
                title=dict(text='Sector-wise Avg Predicted Return', font=dict(size=12, color='#8896b3')),
            )
            fig.update_yaxes(tickfont=dict(size=9), gridcolor='rgba(255,255,255,0.04)', title_text=None)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No sector data available.")

    # Signal summary table
    st.markdown('<div class="section-header"><h2>Quick Signal Table</h2><span class="section-badge">Top 15</span></div>', unsafe_allow_html=True)
    if not preds_df.empty:
        top15 = preds_df.head(15).copy()
        
        # Calculate metrics for each row
        metrics = top15.apply(lambda r: calculate_target_metrics(r['current_price'], r['q50_return'], r['spread']), axis=1)
        top15['Current Price'] = [f"₹{m['current_price']:,.2f}" for m in metrics]
        top15['Safe Alpha'] = [f"{m['safe_target']:+.2f}%" for m in metrics]
        top15['Confidence'] = [f"{m['confidence_score']:.0f}% ({m['confidence_label']})" for m in metrics]
        
        top15['Signal'] = top15.apply(lambda r: signal_label(r['q50_return'], r['signal_strength'])[0], axis=1)
        top15['Return (Q50)'] = (top15['q50_return'] * 100).map(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
        top15['Signal Str.'] = top15['signal_strength'].map(lambda x: f"{x:.2f}")

        disp = top15[['ticker', 'company_name', 'sector', 'Current Price', 'Safe Alpha', 'Signal', 'Return (Q50)', 'Confidence', 'Signal Str.']].rename(
            columns={'ticker': 'Ticker', 'company_name': 'Company', 'sector': 'Sector'}
        )
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn(width=100),
                "Company": st.column_config.TextColumn(width=200),
                "Current Price": st.column_config.TextColumn(help="Latest closing price from market data"),
                "Safe Alpha": st.column_config.TextColumn(help="Under-promise target: positive expected relative return is discounted by 20% for safety"),
                "Confidence": st.column_config.TextColumn(help="Uncertainty-mapped score: based on spread between Q10 and Q90"),
                "Return (Q50)": st.column_config.TextColumn(help="AI's raw expected 5-day return"),
            }
        )


# ─── Page: Top Picks ──────────────────────────────────────────────────────────
elif nav == "📈  Top Picks":
    st.markdown('<div style="font-size:1.6rem;font-weight:800;margin-bottom:4px;">Top Alpha Opportunities</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8896b3;font-size:0.85rem;margin-bottom:28px;">Ranked by TFT Q50 predicted return with confidence metrics</div>', unsafe_allow_html=True)

    preds_df = load_latest_predictions(selected_horizon)

    if preds_df.empty:
        st.warning("No predictions in database. Run `run_daily_update.py` first.")
    else:
        # Filters
        col_f1, col_f2, col_f3 = st.columns([1.5, 2, 1])
        with col_f1:
            signal_filter = st.selectbox("Signal Filter", ["All", "Strong Buy", "Buy", "Neutral", "Sell"])
        with col_f2:
            sectors = ["All"] + sorted(preds_df['sector'].dropna().unique().tolist())
            sector_filter = st.selectbox("Sector", sectors)
        with col_f3:
            top_n = st.number_input("Show Top N", min_value=5, max_value=200, value=30)

        df = preds_df.copy()
        df['signal_label'] = df.apply(lambda r: signal_label(r['q50_return'], r['signal_strength'])[0], axis=1)

        if signal_filter != "All":
            sig_map = {"Strong Buy": "STRONG BUY", "Buy": "BUY", "Neutral": "NEUTRAL", "Sell": "SELL"}
            df = df[df['signal_label'] == sig_map[signal_filter]]
        if sector_filter != "All":
            df = df[df['sector'] == sector_filter]

        df = df.head(int(top_n))

        # Bubble chart
        st.markdown('<div class="section-header"><h2>Signal Map</h2><span class="section-badge">Q50 vs Confidence</span></div>', unsafe_allow_html=True)

        if not df.empty:
            df['color_val'] = df['q50_return']
            fig = px.scatter(
                df,
                x='q50_return',
                y='signal_strength',
                size='spread',
                color='q50_return',
                hover_name='ticker',
                hover_data={'company_name': True, 'sector': True, 'q50_return': ':.4f', 'signal_strength': ':.2f', 'spread': False, 'color_val': False},
                color_continuous_scale=[(0, '#ef4444'), (0.45, '#f59e0b'), (0.55, '#f59e0b'), (1, '#10b981')],
                size_max=30,
                labels={'q50_return': 'Predicted Return', 'signal_strength': 'Signal Strength (Sharpe-like)'},
            )
            fig.add_vline(x=0, line_color='rgba(255,255,255,0.2)', line_dash='dash')
            fig.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_dash='dash')
            fig.update_layout(
                **PLOT_LAYOUT,
                height=380,
                coloraxis_showscale=False,
                title=dict(text='Bubble size = uncertainty spread', font=dict(size=11, color='#8896b3')),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Full ranked table
        st.markdown('<div class="section-header"><h2>Ranked Picks</h2></div>', unsafe_allow_html=True)
        disp = df.copy()
        
        # Calculate metrics for each row
        metrics = disp.apply(lambda r: calculate_target_metrics(r['current_price'], r['q50_return'], r['spread']), axis=1)
        disp['Current Price'] = [f"₹{m['current_price']:,.2f}" for m in metrics]
        disp['Safe Alpha'] = [f"{m['safe_target']:+.2f}%" for m in metrics]
        disp['Confidence'] = [f"{m['confidence_score']:.0f}% ({m['confidence_label']})" for m in metrics]
        
        disp['Q10'] = (disp['q10_return'] * 100).map(lambda x: f"{x:+.3f}%")
        disp['Expected Return (Q50) ▲'] = (disp['q50_return'] * 100).map(lambda x: f"{x:+.3f}%")
        disp['Q90'] = (disp['q90_return'] * 100).map(lambda x: f"{x:+.3f}%")
        disp['Spread'] = (disp['spread'] * 100).map(lambda x: f"{x:.3f}%")
        disp['Signal Str.'] = disp['signal_strength'].map(lambda x: f"{x:.2f}")
        disp = disp.rename(columns={'ticker': 'Ticker', 'company_name': 'Company', 'sector': 'Sector', 'signal_label': 'Signal'})

        st.dataframe(
            disp[['Ticker', 'Company', 'Sector', 'Signal', 'Current Price', 'Safe Alpha', 'Confidence', 'Expected Return (Q50) ▲', 'Signal Str.']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Current Price": st.column_config.TextColumn(help="Latest closing price from market data"),
                "Safe Alpha": st.column_config.TextColumn(help="Under-promise target: positive expected relative return is discounted by 20% for safety"),
                "Confidence": st.column_config.TextColumn(help="Uncertainty-mapped score: based on spread between Q10 and Q90"),
                "Expected Return (Q50) ▲": st.column_config.TextColumn(help="Median predicted return"),
                "Signal Str.": st.column_config.TextColumn(help="Sharpe-like signal: Q50 / spread"),
            }
        )

        # Bar chart — top 20
        st.markdown('<div class="section-header"><h2>Return Forecast Bar</h2><span class="section-badge">Top 20</span></div>', unsafe_allow_html=True)
        top20 = df.head(20).sort_values('q50_return', ascending=True)
        if not top20.empty:
            fig = go.Figure()
            # Error bars (Q10→Q90)
            fig.add_trace(go.Bar(
                y=top20['ticker'],
                x=top20['q50_return'] * 100,
                orientation='h',
                marker=dict(
                    color=top20['q50_return'].apply(lambda x: '#10b981' if x > 0 else '#ef4444'),
                    opacity=0.9,
                    line=dict(width=0),
                ),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=(top20['q90_return'] - top20['q50_return']) * 100,
                    arrayminus=(top20['q50_return'] - top20['q10_return']) * 100,
                    color='rgba(255,255,255,0.3)',
                    thickness=1.5,
                ),
                name='Q50',
                hovertemplate='<b>%{y}</b><br>Q50: %{x:.3f}%<extra></extra>',
            ))
            fig.add_vline(x=0, line_color='rgba(255,255,255,0.2)', line_width=1)
            fig.update_layout(
                **PLOT_LAYOUT,
                height=420,
                showlegend=False,
                xaxis_title='Predicted Return (%)',
                title=dict(text='Error bars = Q10/Q90 confidence band', font=dict(size=11, color='#8896b3')),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ─── Page: Stock Deep Dive ────────────────────────────────────────────────────
elif nav == "🔍  Stock Deep Dive":
    st.markdown('<div style="font-size:1.6rem;font-weight:800;margin-bottom:4px;">Stock Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8896b3;font-size:0.85rem;margin-bottom:28px;">Technical analysis, prediction history, and sentiment for any stock</div>', unsafe_allow_html=True)

    preds_df = load_latest_predictions(selected_horizon)
    all_tickers = sorted(preds_df['ticker'].unique().tolist()) if not preds_df.empty else []

    col_s, col_p = st.columns([2, 1])
    with col_s:
        selected = st.selectbox("Select Stock", all_tickers, index=0 if all_tickers else None)
    with col_p:
        price_days = st.selectbox("Price History", [30, 60, 90, 180, 365], index=2)

    if not selected:
        st.info("No predictions available.")
        st.stop()

    # Load data
    price_df = load_price_history(selected, price_days)
    pred_hist = load_prediction_history(selected, 90)
    sent_df = load_sentiment_history(selected, 90)

    # Company info
    ticker_info = preds_df[preds_df['ticker'] == selected].iloc[0] if not preds_df.empty else None

    if ticker_info is not None:
        label, css_cls = signal_label(ticker_info['q50_return'], ticker_info['signal_strength'])
        company_name = ticker_info.get('company_name', selected) or selected
        sector = ticker_info.get('sector', '—') or '—'

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;padding:16px 20px;
                    background:linear-gradient(135deg,rgba(59,130,246,0.06),rgba(6,182,212,0.03));
                    border:1px solid rgba(59,130,246,0.15);border-radius:12px;margin-bottom:20px;">
            <div>
                <div style="font-size:0.65rem;color:#8896b3;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">{sector}</div>
                <div style="font-size:1.4rem;font-weight:800;color:#f0f4ff;">{company_name}</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#93c5fd;">{selected}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
                <span class="{css_cls}">{label}</span>
                <div style="margin-top:6px;font-size:1.1rem;font-weight:700;color:{'#10b981' if ticker_info['q50_return']>0 else '#ef4444'};">
                    {ticker_info['q50_return']*100:+.3f}% <span style="font-size:0.7rem;color:#8896b3;">Q50</span>
                </div>
                <div style="font-size:0.75rem;color:#8896b3;">
                    Range: {ticker_info['q10_return']*100:+.2f}% → {ticker_info['q90_return']*100:+.2f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Calculate metrics
        m = calculate_target_metrics(ticker_info['current_price'], ticker_info['q50_return'], ticker_info['spread'])

        # Display AI Market-Adjusted Target Calculator Grid
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:24px;">
            <div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:12px;text-align:center;position:relative;overflow:hidden;">
                <div style="font-size:0.65rem;color:#8896b3;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Current Price</div>
                <div style="font-size:1.6rem;font-weight:800;color:#f0f4ff;font-family:'JetBrains Mono',monospace;">₹{m['current_price']:,.2f}</div>
                <div style="font-size:0.7rem;color:#4a5568;margin-top:4px;">Latest market close price</div>
            </div>
            <div style="padding:16px;background:linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,182,212,0.05));border:1px solid var(--accent-green);border-radius:12px;text-align:center;position:relative;overflow:hidden;box-shadow: 0 4px 20px rgba(16,185,129,0.08);">
                <div style="font-size:0.65rem;color:#10b981;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;font-weight:700;">Safe Alpha (Under-Promise)</div>
                <div style="font-size:1.6rem;font-weight:800;color:#10b981;font-family:'JetBrains Mono',monospace;">{m['safe_target']:+.2f}%</div>
                <div style="font-size:0.7rem;color:#8896b3;margin-top:4px;">Expected outperformance vs NIFTY</div>
            </div>
            <div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:12px;text-align:center;position:relative;overflow:hidden;">
                <div style="font-size:0.65rem;color:#8896b3;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">AI Raw Alpha & Confidence</div>
                <div style="font-size:1.4rem;font-weight:800;color:#3b82f6;font-family:'JetBrains Mono',monospace;">{m['raw_target']:+.2f}% <span style="font-size:0.75rem;font-weight:500;color:#8896b3;">(raw)</span></div>
                <div style="font-size:0.75rem;margin-top:4px;font-weight:600;" class="{m['confidence_class']}">
                    Confidence: {m['confidence_score']:.0f}% ({m['confidence_label']})
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Candlestick + Indicators ──────────────────────────────────────────────
    st.markdown('<div class="section-header"><h2>Price Chart + Indicators</h2></div>', unsafe_allow_html=True)

    if not price_df.empty:
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.sort_values('date')
        closes = price_df['adj_close'].astype(float)

        rsi = compute_rsi(closes)
        macd_line, macd_sig, macd_hist = compute_macd(closes)
        bb_upper, bb_mid, bb_lower = compute_bbands(closes)
        price_df['sma20'] = closes.rolling(20).mean()
        price_df['sma50'] = closes.rolling(50).mean()

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.50, 0.18, 0.18, 0.14],
            vertical_spacing=0.025,
            subplot_titles=('Price + Bollinger Bands', 'Volume', 'RSI (14)', 'MACD'),
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=price_df['date'],
            open=price_df['open_price'].astype(float),
            high=price_df['high_price'].astype(float),
            low=price_df['low_price'].astype(float),
            close=price_df['close_price'].astype(float),
            name='OHLC',
            increasing=dict(fillcolor='#10b981', line=dict(color='#10b981', width=0.8)),
            decreasing=dict(fillcolor='#ef4444', line=dict(color='#ef4444', width=0.8)),
        ), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=price_df['date'], y=bb_upper, name='BB Upper',
            line=dict(color='rgba(59,130,246,0.4)', width=1, dash='dot'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df['date'], y=bb_lower, name='BB Lower',
            line=dict(color='rgba(59,130,246,0.4)', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(59,130,246,0.04)', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df['date'], y=bb_mid, name='BB Mid',
            line=dict(color='rgba(59,130,246,0.5)', width=1), showlegend=False), row=1, col=1)

        # SMAs
        fig.add_trace(go.Scatter(x=price_df['date'], y=price_df['sma20'], name='SMA20',
            line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df['date'], y=price_df['sma50'], name='SMA50',
            line=dict(color='#8b5cf6', width=1.5)), row=1, col=1)

        # Volume
        vol_colors = ['#10b981' if c >= o else '#ef4444'
                     for c, o in zip(price_df['close_price'].astype(float), price_df['open_price'].astype(float))]
        fig.add_trace(go.Bar(
            x=price_df['date'], y=price_df['volume'].astype(float),
            name='Volume', marker_color=vol_colors, opacity=0.7, showlegend=False,
        ), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=price_df['date'], y=rsi, name='RSI',
            line=dict(color='#06b6d4', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_color='rgba(239,68,68,0.4)', line_dash='dot', row=3, col=1)
        fig.add_hline(y=30, line_color='rgba(16,185,129,0.4)', line_dash='dot', row=3, col=1)
        fig.add_hline(y=50, line_color='rgba(255,255,255,0.1)', line_dash='dot', row=3, col=1)

        # MACD
        hist_colors = ['#10b981' if v >= 0 else '#ef4444' for v in macd_hist.fillna(0)]
        fig.add_trace(go.Bar(x=price_df['date'], y=macd_hist, name='MACD Hist',
            marker_color=hist_colors, opacity=0.6, showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=price_df['date'], y=macd_line, name='MACD',
            line=dict(color='#3b82f6', width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=price_df['date'], y=macd_sig, name='Signal',
            line=dict(color='#f59e0b', width=1.5, dash='dot')), row=4, col=1)

        fig.update_layout(
            **PLOT_LAYOUT,
            height=680,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', y=1.02, x=0, font=dict(size=10),
                       bgcolor='rgba(10,14,26,0.8)', bordercolor='rgba(59,130,246,0.2)', borderwidth=1),
        )
        fig.update_yaxes(row=3, col=1, range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info(f"No price data for {selected}. Fetch data first.")

    # ── Prediction History ────────────────────────────────────────────────────
    if not pred_hist.empty:
        st.markdown('<div class="section-header"><h2>Prediction History</h2></div>', unsafe_allow_html=True)
        pred_hist['generated_at'] = pd.to_datetime(pred_hist['generated_at'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_hist['generated_at'], y=pred_hist['q90_return'] * 100,
            name='Q90', fill='tonexty', fillcolor='rgba(16,185,129,0.06)',
            line=dict(color='rgba(16,185,129,0.3)', width=0.5),
        ))
        fig.add_trace(go.Scatter(
            x=pred_hist['generated_at'], y=pred_hist['q50_return'] * 100,
            name='Q50 (Median)', fill='tonexty', fillcolor='rgba(59,130,246,0.1)',
            line=dict(color='#3b82f6', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=pred_hist['generated_at'], y=pred_hist['q10_return'] * 100,
            name='Q10', fill='tonexty', fillcolor='rgba(239,68,68,0.05)',
            line=dict(color='rgba(239,68,68,0.3)', width=0.5),
        ))
        fig.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_dash='dash')
        fig.update_layout(**PLOT_LAYOUT, height=240, xaxis_title='Date')
        fig.update_yaxes(title_text='Predicted Return (%)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Sentiment ─────────────────────────────────────────────────────────────
    if not sent_df.empty:
        st.markdown('<div class="section-header"><h2>FinBERT Sentiment History</h2></div>', unsafe_allow_html=True)
        sent_df['date'] = pd.to_datetime(sent_df['date'])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(
            x=sent_df['date'], y=sent_df['net_sentiment_score'],
            name='Net Sentiment', fill='tozeroy',
            fillcolor='rgba(59,130,246,0.1)', line=dict(color='#3b82f6', width=2),
        ), row=1, col=1)
        fig.add_trace(go.Bar(x=sent_df['date'], y=sent_df['bullish_intensity'],
            name='Bullish', marker_color='#10b981', opacity=0.7), row=2, col=1)
        fig.add_trace(go.Bar(x=sent_df['date'], y=sent_df['bearish_intensity'].apply(lambda x: -x),
            name='Bearish', marker_color='#ef4444', opacity=0.7), row=2, col=1)
        fig.add_hline(y=0, row=1, col=1, line_color='rgba(255,255,255,0.1)')
        fig.update_layout(**PLOT_LAYOUT, height=300, barmode='relative')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown('<div style="color:#4a5568;font-size:0.8rem;padding:12px 0;">No sentiment data available for this stock.</div>', unsafe_allow_html=True)


# ─── Page: Sector Analysis ────────────────────────────────────────────────────
elif nav == "🌐  Sector Analysis":
    st.markdown('<div style="font-size:1.6rem;font-weight:800;margin-bottom:4px;">Sector Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8896b3;font-size:0.85rem;margin-bottom:28px;">AI model predictions aggregated by market sector</div>', unsafe_allow_html=True)

    preds_df = load_latest_predictions(selected_horizon)
    sector_df = load_sector_performance(selected_horizon)

    if preds_df.empty:
        st.warning("No predictions in database.")
        st.stop()

    # Treemap
    st.markdown('<div class="section-header"><h2>Sector Treemap</h2><span class="section-badge">By Return</span></div>', unsafe_allow_html=True)
    preds_sector = preds_df.dropna(subset=['sector'])
    if not preds_sector.empty:
        fig = px.treemap(
            preds_sector,
            path=['sector', 'ticker'],
            values='signal_strength',
            color='q50_return',
            color_continuous_scale=[(0, '#7f1d1d'), (0.35, '#ef4444'), (0.5, '#1e293b'), (0.65, '#10b981'), (1, '#064e3b')],
            color_continuous_midpoint=0,
            hover_data={'q50_return': ':.4f', 'signal_strength': ':.2f'},
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='white', size=11),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
            coloraxis_colorbar=dict(
                title='5D Return',
                tickformat='.2%',
                len=0.8,
                thickness=12,
            )
        )
        fig.update_traces(
            textfont=dict(size=11, color='white'),
            marker=dict(line=dict(width=1, color='rgba(0,0,0,0.4)')),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    if not sector_df.empty:
        # Sector table
        st.markdown('<div class="section-header"><h2>Sector Breakdown</h2></div>', unsafe_allow_html=True)
        disp = sector_df.copy()
        disp['Avg Return'] = (disp['avg_return'] * 100).map(lambda x: f"{x:+.3f}%")
        disp['Best Stock Return'] = (disp['best_return'] * 100).map(lambda x: f"{x:+.3f}%")
        disp['Avg Signal Str.'] = disp['avg_signal'].map(lambda x: f"{x:.2f}")
        disp = disp.rename(columns={'sector': 'Sector', 'stock_count': 'Stocks'})
        st.dataframe(
            disp[['Sector', 'Stocks', 'Avg Return', 'Best Stock Return', 'Avg Signal Str.']],
            use_container_width=True,
            hide_index=True,
        )

        # Scatter: signal vs return per sector
        st.markdown('<div class="section-header"><h2>Risk-Return by Sector</h2></div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            sector_df,
            x='avg_return',
            y='avg_signal',
            size='stock_count',
            color='avg_return',
            text='sector',
            color_continuous_scale=[(0, '#ef4444'), (0.5, '#f59e0b'), (1, '#10b981')],
            labels={'avg_return': 'Avg 5D Return', 'avg_signal': 'Avg Signal Strength'},
        )
        fig2.update_traces(textposition='top center', textfont=dict(size=9, color='#8896b3'))
        fig2.update_layout(
            **PLOT_LAYOUT,
            height=350,
            coloraxis_showscale=False,
            title=dict(text='Bubble size = number of stocks in sector', font=dict(size=11, color='#8896b3')),
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


# ─── Page: News & Sentiment ────────────────────────────────────────────────────
elif nav == "📰  News & Sentiment":
    st.markdown('<div style="font-size:1.6rem;font-weight:800;margin-bottom:4px;">News & Sentiment Feed</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8896b3;font-size:0.85rem;margin-bottom:28px;">FinBERT-analyzed corporate announcements from live document watcher</div>', unsafe_allow_html=True)

    reports_df = load_recent_reports(50)

    if reports_df.empty:
        st.info("No reports in the database. Drop PDFs into the `uploaded_documents/` folder.")
    else:
        # Overall sentiment metrics
        preds_df = load_latest_predictions(selected_horizon)
        sent_query = """
            SELECT ticker, AVG(net_sentiment_score) AS avg_sent,
                   AVG(bullish_intensity) AS avg_bull, AVG(bearish_intensity) AS avg_bear
            FROM daily_sentiment_aggregates
            WHERE date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY ticker
            ORDER BY avg_sent DESC
        """
        sent_overview = query(sent_query)

        if not sent_overview.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header"><h2>Top Bullish (7-day)</h2></div>', unsafe_allow_html=True)
                top_bull = sent_overview.head(10)
                fig = go.Figure(go.Bar(
                    x=top_bull['avg_sent'].astype(float),
                    y=top_bull['ticker'],
                    orientation='h',
                    marker=dict(color='#10b981', opacity=0.8),
                    text=top_bull['avg_sent'].astype(float).map(lambda x: f"{x:.3f}"),
                    textposition='outside',
                ))
                fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            with col2:
                st.markdown('<div class="section-header"><h2>Top Bearish (7-day)</h2></div>', unsafe_allow_html=True)
                top_bear = sent_overview.tail(10).sort_values('avg_sent')
                fig = go.Figure(go.Bar(
                    x=top_bear['avg_sent'].astype(float),
                    y=top_bear['ticker'],
                    orientation='h',
                    marker=dict(color='#ef4444', opacity=0.8),
                    text=top_bear['avg_sent'].astype(float).map(lambda x: f"{x:.3f}"),
                    textposition='outside',
                ))
                fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Recent filings table
        st.markdown('<div class="section-header"><h2>Recent Filings</h2><span class="section-badge">Live Feed</span></div>', unsafe_allow_html=True)
        for _, row in reports_df.head(20).iterrows():
            sentiment_badge = ""
            ts_str = str(row.get('received_time', ''))[:16] if row.get('received_time') else "—"
            ticker = row.get('ticker', '—')
            company = row.get('company_name', '—') or ticker
            headline = str(row.get('headline', 'No headline')) or "No headline"
            rtype = str(row.get('report_type', '')) or "Filing"

            st.markdown(f"""
            <div style="padding:12px 16px;margin-bottom:8px;background:var(--bg-card);
                        border:1px solid var(--border);border-radius:10px;
                        border-left:3px solid var(--accent-blue);
                        transition:all 0.2s;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span class="ticker-pill">{ticker}</span>
                        <span style="color:#f0f4ff;font-weight:600;font-size:0.85rem;">{company}</span>
                        <span style="font-size:0.65rem;background:rgba(139,92,246,0.15);color:#c4b5fd;
                                     padding:2px 8px;border-radius:5px;font-weight:600;">{rtype}</span>
                    </div>
                    <span style="font-size:0.7rem;color:#4a5568;font-family:JetBrains Mono,monospace;">{ts_str}</span>
                </div>
                <div style="font-size:0.8rem;color:#8896b3;line-height:1.4;">{headline[:200]}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── Page: Pipeline Status ────────────────────────────────────────────────────
elif nav == "⚙️  Pipeline Status":
    st.markdown('<div style="font-size:1.6rem;font-weight:800;margin-bottom:4px;">Pipeline & System Status</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8896b3;font-size:0.85rem;margin-bottom:28px;">Database health, data coverage, and pipeline monitoring</div>', unsafe_allow_html=True)

    stats_df = load_db_stats()

    if not stats_df.empty:
        row = stats_df.iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Active Tickers</div>
                <div class="metric-value metric-blue">{int(row['active_stocks']) if row['active_stocks'] else 0}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest Predictions</div>
                <div class="metric-value metric-green">{int(row['latest_preds']) if row['latest_preds'] else 0}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Reports</div>
                <div class="metric-value metric-purple">{int(row['total_reports']) if row['total_reports'] else 0}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col4, col5 = st.columns(2)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Last Price Date</div>
                <div class="metric-value" style="font-size:1.2rem;color:#06b6d4;">
                    {str(row['last_price_date'])[:10] if row['last_price_date'] else 'N/A'}
                </div>
            </div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Last Prediction Run</div>
                <div class="metric-value" style="font-size:1.2rem;color:#06b6d4;">
                    {str(row['last_prediction_time'])[:16] if row['last_prediction_time'] else 'N/A'}
                </div>
            </div>""", unsafe_allow_html=True)

    # Data coverage chart
    st.markdown('<div class="section-header"><h2>Data Coverage per Ticker</h2></div>', unsafe_allow_html=True)
    coverage_df = query("""
        SELECT ticker, COUNT(*) AS days, MIN(date) AS from_date, MAX(date) AS to_date
        FROM market_data_daily
        GROUP BY ticker
        ORDER BY days DESC
        LIMIT 50
    """)

    if not coverage_df.empty:
        fig = go.Figure(go.Bar(
            x=coverage_df['ticker'],
            y=coverage_df['days'].astype(int),
            marker=dict(
                color=coverage_df['days'].astype(int),
                colorscale=[[0, '#1e3a5f'], [1, '#3b82f6']],
                line=dict(width=0),
            ),
            hovertemplate='<b>%{x}</b><br>Days: %{y}<extra></extra>',
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            height=280,
            showlegend=False,
            xaxis_tickangle=-45,
            xaxis_tickfont=dict(size=8),
            yaxis_title='Trading Days in DB',
            title=dict(text='Top 50 tickers by data history length', font=dict(size=11, color='#8896b3')),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Run pipeline controls
    st.markdown('<div class="section-header"><h2>Manual Controls</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:10px;margin-bottom:12px;">
        <div style="font-size:0.75rem;color:#8896b3;margin-bottom:12px;">
            Run these commands in your terminal to manually trigger pipeline stages:
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:0.75rem;line-height:2;color:#93c5fd;">
            <div style="color:#4a5568;"># Full daily update (fetch + features + predict)</div>
            <div>.\\venv\\Scripts\\python.exe run_daily_update.py</div>
            <br>
            <div style="color:#4a5568;"># Fetch only latest prices</div>
            <div>.\\venv\\Scripts\\python.exe Cleaner/daily_fetch.py --days 15</div>
            <br>
            <div style="color:#4a5568;"># Run prediction only</div>
            <div>.\\venv\\Scripts\\python.exe Cleaner/predict.py --drift-check</div>
            <br>
            <div style="color:#4a5568;"># Start live document watcher</div>
            <div>.\\venv\\Scripts\\python.exe Cleaner/live_watcher.py</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sentiment chunks coverage
    st.markdown('<div class="section-header"><h2>Sentiment Chunk Analysis</h2></div>', unsafe_allow_html=True)
    chunks_df = query("""
        SELECT c.sector, COUNT(sc.chunk_id) AS chunk_count, COUNT(DISTINCT fr.ticker) AS tickers
        FROM sentiment_chunks sc
        JOIN financial_reports fr ON sc.report_id = fr.report_id
        JOIN companies c ON fr.ticker = c.ticker
        WHERE c.sector IS NOT NULL AND c.sector != ''
        GROUP BY c.sector
        ORDER BY chunk_count DESC
    """)

    if not chunks_df.empty:
        fig = px.bar(
            chunks_df, x='sector', y='chunk_count',
            color='tickers',
            color_continuous_scale=[[0, '#1e3a5f'], [1, '#3b82f6']],
            labels={'chunk_count': 'Sentiment Chunks', 'sector': 'Sector'},
        )
        fig.update_layout(
            **PLOT_LAYOUT, height=280,
            coloraxis_colorbar=dict(title='Tickers', thickness=12, len=0.7),
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("No sentiment chunk data yet. Upload documents via the live watcher.")

    # Live watcher status
    st.markdown('<div class="section-header"><h2>System Status</h2></div>', unsafe_allow_html=True)
    conn = get_connection()
    db_ok = conn is not None

    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploaded_documents'))
    pending = []
    if os.path.exists(upload_dir):
        pending = [f for f in os.listdir(upload_dir) if f.lower().endswith('.pdf')]

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div style="padding:14px;background:var(--bg-card);border:1px solid var(--border);border-radius:10px;">
            <div style="font-size:0.65rem;color:#8896b3;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Database</div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:10px;height:10px;border-radius:50%;background:{'#10b981' if db_ok else '#ef4444'};
                            box-shadow:0 0 8px {'#10b981' if db_ok else '#ef4444'};"></div>
                <div style="font-weight:600;color:{'#10b981' if db_ok else '#ef4444'};">
                    {'Connected' if db_ok else 'Disconnected'}
                </div>
            </div>
            <div style="font-size:0.7rem;color:#4a5568;margin-top:4px;">{DB_CFG['host']}:{DB_CFG['port']} / {DB_CFG['database']}</div>
        </div>
        <div style="padding:14px;background:var(--bg-card);border:1px solid var(--border);border-radius:10px;">
            <div style="font-size:0.65rem;color:#8896b3;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Pending Documents</div>
            <div style="font-size:1.5rem;font-weight:800;color:{'#f59e0b' if pending else '#10b981'};">{len(pending)}</div>
            <div style="font-size:0.7rem;color:#4a5568;margin-top:4px;">in uploaded_documents/</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Page: Accuracy ───────────────────────────────────────────────────────────
elif nav == "🎯  Accuracy":
    st.markdown('<div style="font-size:1.6rem;font-weight:800;margin-bottom:4px;">Was the AI Right?</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8896b3;font-size:0.85rem;margin-bottom:28px;">Every week we check: did the stocks the AI said would go up, actually go up? Here are the real results.</div>', unsafe_allow_html=True)

    # ── Overall aggregate KPIs ────────────────────────────────────────────────
    agg_df = query("""
        SELECT
            COUNT(*)                          AS total_evaluated,
            AVG(abs_error)                    AS overall_mae,
            SQRT(AVG(sq_error))               AS overall_rmse,
            AVG(direction_correct::float)     AS overall_dir_acc,
            AVG(in_quantile_band::float)      AS overall_q_coverage,
            AVG(q10_miss::float)              AS q10_miss_rate,
            AVG(q90_miss::float)              AS q90_miss_rate,
            SUM(CASE WHEN q50_predicted > 0.005 AND direction_correct=1
                     THEN 1 ELSE 0 END)::float /
                NULLIF(SUM(CASE WHEN q50_predicted > 0.005 THEN 1 ELSE 0 END),0) AS buy_hit_rate
        FROM prediction_accuracy
    """)

    has_data = not agg_df.empty and agg_df['total_evaluated'].iloc[0] and int(agg_df['total_evaluated'].iloc[0]) > 0

    if not has_data:
        # ── Pending notice when no evaluations yet ───────────────────────────
        st.markdown("""
        <div style="padding:28px;background:linear-gradient(135deg,rgba(245,158,11,0.06),rgba(59,130,246,0.04));
                    border:1px solid rgba(245,158,11,0.2);border-radius:14px;text-align:center;">
            <div style="font-size:2rem;margin-bottom:12px;">⏳</div>
            <div style="font-size:1rem;font-weight:700;color:#f0f4ff;margin-bottom:8px;">Accuracy data will appear in ~7 days</div>
            <div style="font-size:0.82rem;color:#8896b3;line-height:1.6;max-width:480px;margin:0 auto;">
                The model predicted returns for today's prices. We need to wait until the
                <strong style="color:#f0f4ff;">actual trading window</strong> has closed so we can
                compare predictions against real market prices.<br><br>
                The <code style="background:rgba(59,130,246,0.12);padding:2px 6px;border-radius:4px;
                color:#93c5fd;">accuracy_checker.py</code> runs automatically every day as part of
                <code style="background:rgba(59,130,246,0.12);padding:2px 6px;border-radius:4px;
                color:#93c5fd;">run_daily_update.py</code>.
            </div>
            <div style="margin-top:20px;font-size:0.75rem;color:#4a5568;font-family:JetBrains Mono,monospace;">
                You can also force it now: .\\venv\\Scripts\\python.exe Cleaner/accuracy_checker.py
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show what WILL be displayed once data arrives
        st.markdown('<div class="section-header" style="margin-top:32px;"><h2>What You\'ll See Here</h2></div>', unsafe_allow_html=True)
        preview_items = [
            ("📏", "How far off was the AI?", "On average, how many % points did the AI's price prediction miss by. Lower = more accurate."),
            ("🧭", "Did it call the direction right?", "When the AI said a stock would go UP, did it actually go up? Flipping a coin gives 50% — anything above that means the AI is adding value."),
            ("🎯", "Did the confidence range make sense?", "The AI gives a best-case and worst-case range. This checks if the real outcome usually landed inside that range — it should about 80% of the time."),
            ("💰", "When it said BUY, did you make money?", "Out of all the stocks the AI flagged as BUY, how many actually went up in the target horizon?"),
            ("🏆", "Top 10 picks — real profit", "If you had put equal money into the AI's top 10 recommended stocks each week, how much would you have made?"),
            ("📊", "Did it beat the market?", "Did the AI's top picks do better than just buying NIFTY 50? If yes, the AI is earning its keep."),
            ("🔬", "Which stocks is it best at?", "Some stocks are easier to predict than others. See where the AI shines and where it struggles."),
            ("📈", "Is the confidence range honest?", "Checks if the AI is too optimistic or too pessimistic about how extreme price moves can be."),
        ]
        for icon, title, desc_text in preview_items:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;padding:10px 14px;
                        background:var(--bg-card);border:1px solid var(--border);border-radius:8px;margin-bottom:6px;">
                <span style="font-size:1.1rem;">{icon}</span>
                <div>
                    <div style="font-weight:600;font-size:0.82rem;color:#f0f4ff;">{title}</div>
                    <div style="font-size:0.75rem;color:#8896b3;margin-top:2px;">{desc_text}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.stop()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    a = agg_df.iloc[0]
    mae_pct = float(a['overall_mae']) * 100
    rmse_pct = float(a['overall_rmse']) * 100
    dir_acc = float(a['overall_dir_acc'])
    q_cov = float(a['overall_q_coverage'])
    buy_hit = float(a['buy_hit_rate']) if a['buy_hit_rate'] is not None else None
    total_ev = int(a['total_evaluated'])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stocks Checked</div>
            <div class="metric-value metric-blue">{total_ev:,}</div>
            <div class="metric-delta" style="color:#8896b3;">predictions verified so far</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        mae_label = 'Great' if mae_pct < 1 else ('Good' if mae_pct < 2 else 'OK')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Miss</div>
            <div class="metric-value {'metric-green' if mae_pct < 2 else 'metric-amber'}">{mae_pct:.2f}%</div>
            <div class="metric-delta" style="color:#8896b3;">how far off per stock ({mae_label})</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        dir_color = 'metric-green' if dir_acc >= 0.55 else ('metric-amber' if dir_acc >= 0.50 else 'metric-red')
        dir_label = 'Better than random!' if dir_acc >= 0.55 else ('About average' if dir_acc >= 0.50 else 'Needs improvement')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Called Up/Down Correctly</div>
            <div class="metric-value {dir_color}">{dir_acc:.1%}</div>
            <div class="metric-delta" style="color:#8896b3;">{dir_label} (coin flip = 50%)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        q_color = 'metric-green' if 0.70 <= q_cov <= 0.90 else 'metric-amber'
        q_label = 'Well calibrated' if 0.70 <= q_cov <= 0.90 else ('Too confident' if q_cov < 0.70 else 'Too cautious')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Landed in Safe Zone</div>
            <div class="metric-value {q_color}">{q_cov:.1%}</div>
            <div class="metric-delta" style="color:#8896b3;">{q_label} (target = 80%)</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        if buy_hit is not None:
            bh_color = 'metric-green' if buy_hit >= 0.55 else ('metric-amber' if buy_hit >= 0.50 else 'metric-red')
            bh_label = 'Trustworthy BUY calls' if buy_hit >= 0.60 else ('Usually right' if buy_hit >= 0.50 else 'Be cautious')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">BUY Calls That Paid Off</div>
                <div class="metric-value {bh_color}">{buy_hit:.1%}</div>
                <div class="metric-delta" style="color:#8896b3;">{bh_label}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="metric-label">BUY Calls That Paid Off</div><div class="metric-value" style="color:#4a5568;">Pending</div><div class="metric-delta" style="color:#4a5568;">not enough data yet</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-batch summary table ────────────────────────────────────────────────
    summary_df = query("""
        SELECT batch_date, tickers_evaluated, tickers_missing_data,
               mae, rmse, direction_accuracy, quantile_coverage,
               hit_rate_buy, hit_rate_strong_buy,
               top10_long_return, long_only_return, nifty_return_same_period
        FROM prediction_accuracy_summary
        ORDER BY batch_date DESC
    """)

    if not summary_df.empty:
        col_a, col_b = st.columns([1.2, 1])

        with col_a:
            st.markdown('<div class="section-header"><h2>Batch Accuracy History</h2></div>', unsafe_allow_html=True)
            disp = summary_df.copy()
            disp['MAE'] = (disp['mae'].astype(float) * 100).map(lambda x: f"{x:.3f}%")
            disp['RMSE'] = (disp['rmse'].astype(float) * 100).map(lambda x: f"{x:.3f}%")
            disp['Direction'] = disp['direction_accuracy'].astype(float).map(lambda x: f"{x:.1%}")
            disp['Q-Coverage'] = disp['quantile_coverage'].astype(float).map(lambda x: f"{x:.1%}")
            disp['Top-10 Return'] = disp['top10_long_return'].apply(
                lambda x: f"{float(x)*100:+.2f}%" if x is not None else 'N/A')
            disp['NIFTY'] = disp['nifty_return_same_period'].apply(
                lambda x: f"{float(x)*100:+.2f}%" if x is not None else 'N/A')
            disp['Buy Hit'] = disp['hit_rate_buy'].apply(
                lambda x: f"{float(x):.1%}" if x is not None else 'N/A')
            disp = disp.rename(columns={'batch_date': 'Date', 'tickers_evaluated': 'N'})
            st.dataframe(
                disp[['Date', 'N', 'MAE', 'Direction', 'Q-Coverage', 'Buy Hit', 'Top-10 Return', 'NIFTY']],
                use_container_width=True, hide_index=True,
            )

        with col_b:
            st.markdown('<div class="section-header"><h2>Top-10 Alpha vs NIFTY</h2></div>', unsafe_allow_html=True)
            plot_df = summary_df.dropna(subset=['top10_long_return', 'nifty_return_same_period']).copy()
            if not plot_df.empty:
                plot_df['batch_date'] = pd.to_datetime(plot_df['batch_date'])
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=plot_df['batch_date'].dt.strftime('%d %b'),
                    y=plot_df['top10_long_return'].astype(float) * 100,
                    name='Top-10 Model',
                    marker_color='#3b82f6',
                    opacity=0.85,
                ))
                fig.add_trace(go.Bar(
                    x=plot_df['batch_date'].dt.strftime('%d %b'),
                    y=plot_df['nifty_return_same_period'].astype(float) * 100,
                    name='NIFTY 50',
                    marker_color='#8b5cf6',
                    opacity=0.65,
                ))
                fig.add_hline(y=0, line_color='rgba(255,255,255,0.15)')
                fig.update_layout(**PLOT_LAYOUT, height=240, barmode='group',
                                  xaxis_title='Batch', title=dict(text='Model top-10 vs benchmark', font=dict(size=11, color='#8896b3')))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Need NIFTY data (ticker='^NSEI') in the DB for comparison.")

    # ── Direction accuracy over time ───────────────────────────────────────────
    if not summary_df.empty:
        st.markdown('<div class="section-header"><h2>Model Accuracy Trend</h2></div>', unsafe_allow_html=True)
        trend_df = summary_df.sort_values('batch_date').copy()
        trend_df['batch_date'] = pd.to_datetime(trend_df['batch_date'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Direction Accuracy', 'MAE Over Time'))
        fig.add_trace(go.Scatter(
            x=trend_df['batch_date'], y=trend_df['direction_accuracy'].astype(float) * 100,
            name='Dir Acc %', line=dict(color='#3b82f6', width=2), mode='lines+markers',
        ), row=1, col=1)
        fig.add_hline(y=50, line_color='rgba(239,68,68,0.4)', line_dash='dash', row=1, col=1,
                      annotation_text='Random', annotation_font_color='#ef4444')
        fig.add_trace(go.Scatter(
            x=trend_df['batch_date'], y=trend_df['mae'].astype(float) * 100,
            name='MAE %', line=dict(color='#f59e0b', width=2), mode='lines+markers',
        ), row=1, col=2)
        fig.update_layout(**PLOT_LAYOUT, height=240, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Per-ticker error breakdown ─────────────────────────────────────────────
    st.markdown('<div class="section-header"><h2>Per-Ticker Accuracy</h2></div>', unsafe_allow_html=True)
    ticker_acc = query("""
        SELECT ticker,
               COUNT(*)                        AS batches,
               AVG(abs_error)                  AS mae,
               AVG(direction_correct::float)   AS dir_acc,
               AVG(in_quantile_band::float)    AS q_cov,
               AVG(actual_5d_return)           AS avg_actual_return,
               AVG(q50_predicted)              AS avg_predicted
        FROM prediction_accuracy
        GROUP BY ticker
        HAVING COUNT(*) >= 1
        ORDER BY mae ASC
        LIMIT 50
    """)

    if not ticker_acc.empty:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div style="font-size:0.75rem;color:#8896b3;margin-bottom:8px;font-weight:600;">BEST PREDICTED (Lowest MAE)</div>', unsafe_allow_html=True)
            best = ticker_acc.head(15).copy()
            best['MAE'] = (best['mae'].astype(float) * 100).map(lambda x: f"{x:.4f}%")
            best['Dir Acc'] = best['dir_acc'].astype(float).map(lambda x: f"{x:.1%}")
            best['Q Cov'] = best['q_cov'].astype(float).map(lambda x: f"{x:.1%}")
            st.dataframe(best[['ticker', 'batches', 'MAE', 'Dir Acc', 'Q Cov']].rename(columns={'ticker': 'Ticker', 'batches': 'N'}),
                        use_container_width=True, hide_index=True)

        with col_r:
            st.markdown('<div style="font-size:0.75rem;color:#8896b3;margin-bottom:8px;font-weight:600;">WORST PREDICTED (Highest MAE)</div>', unsafe_allow_html=True)
            worst = ticker_acc.sort_values('mae', ascending=False).head(15).copy()
            worst['MAE'] = (worst['mae'].astype(float) * 100).map(lambda x: f"{x:.4f}%")
            worst['Dir Acc'] = worst['dir_acc'].astype(float).map(lambda x: f"{x:.1%}")
            worst['Q Cov'] = worst['q_cov'].astype(float).map(lambda x: f"{x:.1%}")
            st.dataframe(worst[['ticker', 'batches', 'MAE', 'Dir Acc', 'Q Cov']].rename(columns={'ticker': 'Ticker', 'batches': 'N'}),
                        use_container_width=True, hide_index=True)

        # MAE scatter: predicted vs actual
        st.markdown('<div class="section-header"><h2>Predicted vs Actual Return</h2></div>', unsafe_allow_html=True)
        scatter_df = query("""
            SELECT ticker, q50_predicted, actual_5d_return, abs_error, sector
            FROM prediction_accuracy
            ORDER BY abs_error ASC
            LIMIT 500
        """)
        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x='q50_predicted', y='actual_5d_return',
                color='abs_error',
                color_continuous_scale=[(0, '#10b981'), (0.5, '#f59e0b'), (1, '#ef4444')],
                hover_name='ticker',
                hover_data={'sector': True, 'abs_error': ':.5f'},
                labels={'q50_predicted': 'Predicted Return (Q50)', 'actual_5d_return': 'Actual Return'},
                opacity=0.7,
            )
            # Perfect prediction diagonal
            min_v = min(scatter_df['q50_predicted'].min(), scatter_df['actual_5d_return'].min())
            max_v = max(scatter_df['q50_predicted'].max(), scatter_df['actual_5d_return'].max())
            fig.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                name='Perfect Fit', mode='lines',
                line=dict(color='rgba(255,255,255,0.2)', dash='dash', width=1),
                showlegend=True,
            ))
            fig.add_vline(x=0, line_color='rgba(255,255,255,0.1)')
            fig.add_hline(y=0, line_color='rgba(255,255,255,0.1)')
            fig.update_layout(
                **PLOT_LAYOUT, height=380,
                coloraxis_colorbar=dict(title='Abs Error', thickness=12, len=0.7),
                title=dict(text='Each dot = one ticker prediction vs realized return. Closer to diagonal = better.', font=dict(size=11, color='#8896b3')),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Calibration ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h2>Quantile Calibration</h2></div>', unsafe_allow_html=True)
    calib_df = query("""
        SELECT
            CASE
                WHEN actual_5d_return < -0.03 THEN '< -3%'
                WHEN actual_5d_return < -0.01 THEN '-3% to -1%'
                WHEN actual_5d_return < 0     THEN '-1% to 0%'
                WHEN actual_5d_return < 0.01  THEN '0% to 1%'
                WHEN actual_5d_return < 0.03  THEN '1% to 3%'
                ELSE '> 3%'
            END AS return_bucket,
            AVG(in_quantile_band::float) AS q_coverage,
            COUNT(*) AS n
        FROM prediction_accuracy
        GROUP BY return_bucket
        ORDER BY MIN(actual_5d_return)
    """)
    if not calib_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=calib_df['return_bucket'],
            y=calib_df['q_coverage'].astype(float) * 100,
            marker=dict(
                color=calib_df['q_coverage'].astype(float).apply(
                    lambda x: '#10b981' if 0.70 <= x <= 0.90 else ('#f59e0b' if x > 0.5 else '#ef4444')
                ),
                line=dict(width=0),
            ),
            text=calib_df['n'].astype(str) + ' stocks',
            textposition='outside',
            textfont=dict(size=9, color='#8896b3'),
        ))
        fig.add_hline(y=80, line_color='rgba(59,130,246,0.5)', line_dash='dash',
                      annotation_text='Ideal 80%', annotation_font_color='#3b82f6')
        fig.update_layout(
            **PLOT_LAYOUT, height=260, showlegend=False,
            xaxis_title='Actual Return Bucket', yaxis_title='Q10–Q90 Coverage %',
            title=dict(text='How often does the actual return fall inside the predicted Q10–Q90 range?', font=dict(size=11, color='#8896b3')),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
