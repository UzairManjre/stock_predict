import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np

# Load env variables from .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "stock_predict")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def decay_forward_fill(series, half_life=5.0):
    decay_factor = 0.5 ** (1.0 / half_life)
    values = series.values
    out = np.zeros_like(values, dtype=float)
    current = 0.0
    for i in range(len(values)):
        val = values[i]
        if not np.isnan(val):
            current = val
        else:
            current = current * decay_factor
        out[i] = current
    return out

def main():
    print(f"Connecting to database {DB_NAME}...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    # 1. Fetch market data (excl. NIFTY50) - Now including adj_close
    print("Loading daily market data from PostgreSQL...")
    market_df = pd.read_sql_query("""
        SELECT date, ticker, open_price, high_price, low_price, close_price, adj_close, volume 
        FROM market_data_daily 
        WHERE ticker != 'NIFTY50'
        ORDER BY ticker, date
    """, conn, parse_dates=['date'])
    print(f"Loaded {len(market_df)} market data rows.")

    # 2. Fetch NIFTY50 index price
    print("Loading NIFTY50 index data...")
    nifty_df = pd.read_sql_query("""
        SELECT date, close_price as nifty_close
        FROM market_data_daily
        WHERE ticker = 'NIFTY50'
        ORDER BY date
    """, conn, parse_dates=['date'])
    print(f"Loaded {len(nifty_df)} NIFTY50 rows.")

    # Calculate NIFTY50 log returns
    nifty_df['nifty_50_return'] = np.log(nifty_df['nifty_close'] / nifty_df['nifty_close'].shift(1))
    nifty_df['nifty_5d_return'] = np.log(nifty_df['nifty_close'] / nifty_df['nifty_close'].shift(5))
    nifty_df['nifty_10d_return'] = np.log(nifty_df['nifty_close'] / nifty_df['nifty_close'].shift(10))
    nifty_df['nifty_21d_return'] = np.log(nifty_df['nifty_close'] / nifty_df['nifty_close'].shift(21))
    nifty_df['nifty_126d_return'] = np.log(nifty_df['nifty_close'] / nifty_df['nifty_close'].shift(126))
    
    nifty_df['nifty_volatility_20d'] = nifty_df['nifty_50_return'].rolling(window=20).std().fillna(0.0)
    
    for col in ['nifty_50_return', 'nifty_5d_return', 'nifty_10d_return', 'nifty_21d_return', 'nifty_126d_return']:
        nifty_df[col] = nifty_df[col].fillna(0.0)
        
    nifty_df = nifty_df[['date', 'nifty_50_return', 'nifty_5d_return', 'nifty_10d_return', 'nifty_21d_return', 'nifty_126d_return', 'nifty_volatility_20d']]

    # 3. Fetch companies info (static covariates)
    print("Loading company static attributes...")
    companies_df = pd.read_sql_query("""
        SELECT ticker, sector, industry 
        FROM companies 
        WHERE ticker != 'NIFTY50'
    """, conn)
    print(f"Loaded static metadata for {len(companies_df)} companies.")

    # 4. Fetch daily sentiment aggregates
    print("Loading daily sentiment aggregates...")
    sentiment_df = pd.read_sql_query("""
        SELECT date, ticker, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed
        FROM daily_sentiment_aggregates
        ORDER BY ticker, date
    """, conn, parse_dates=['date'])
    print(f"Loaded {len(sentiment_df)} sentiment rows.")

    # Apply 1-day shift to eliminate lookahead bias
    sentiment_df['date'] = sentiment_df['date'] + pd.Timedelta(days=1)

    # 5. Check which tickers have any sentiment history
    tickers_with_sentiment = set(sentiment_df['ticker'].unique())
    print(f"Tickers with at least one sentiment record: {len(tickers_with_sentiment)}")

    conn.close()

    # 6. Merge static attributes
    print("Merging static metadata...")
    df = pd.merge(market_df, companies_df, on='ticker', how='left')

    # Add boolean indicator for sentiment history
    df['has_sentiment_history'] = df['ticker'].isin(tickers_with_sentiment).astype(int)

    # 7. Merge daily sentiment aggregates
    print("Merging sentiment aggregates...")
    df = pd.merge(df, sentiment_df, on=['date', 'ticker'], how='left')

    # Sort to ensure proper chronological calculations per ticker
    df.sort_values(by=['ticker', 'date'], inplace=True)

    # Calculate median daily dollar volume per ticker to tune sentiment decay per stock
    print("Calculating ticker-specific dollar volume quintiles...")
    df['dollar_volume'] = df['volume'] * df['close_price']
    dollar_volume = df.groupby('ticker')['dollar_volume'].median()
    q20 = dollar_volume.quantile(0.20)
    q80 = dollar_volume.quantile(0.80)
    
    half_life_map = {}
    for ticker, vol in dollar_volume.items():
        if vol >= q80:
            half_life_map[ticker] = 1.5  # Highly liquid / Large-cap (fast decay)
        elif vol <= q20:
            half_life_map[ticker] = 6.0  # Illiquid / Small-cap (slow decay)
        else:
            half_life_map[ticker] = 4.0  # Mid-cap (standard decay)

    # 8. Apply Exponential Decay Forward-Fill and indicator calculations per ticker
    print("Applying exponential decay and calculating technical indicators on adjusted prices...")
    decay_cols = ['net_sentiment_score', 'bullish_intensity', 'bearish_intensity', 'total_chunks_analyzed']
    
    # Process ticker by ticker to apply decay and compute technicals
    grouped = df.groupby('ticker')
    decayed_chunks = []
    
    for ticker, group in grouped:
        group = group.copy()
        
        # Calculate split and dividend adjustment factor
        adj_factor = group['adj_close'] / (group['close_price'] + 1e-8)
        group['adj_open'] = group['open_price'] * adj_factor
        group['adj_high'] = group['high_price'] * adj_factor
        group['adj_low'] = group['low_price'] * adj_factor
        
        # Fetch ticker-specific half-life
        hl = half_life_map.get(ticker, 4.0)
        for col in decay_cols:
            decayed_col_name = f'decayed_{col}' if col != 'total_chunks_analyzed' else 'decayed_announcement_volume'
            group[decayed_col_name] = decay_forward_fill(group[col], half_life=hl)
        
        # Calculate technical indicators on split/dividend adjusted price
        group['log_return_1d'] = np.log(group['adj_close'] / group['adj_close'].shift(1))
        group['log_return_3d'] = np.log(group['adj_close'] / group['adj_close'].shift(3))
        group['log_return_5d'] = np.log(group['adj_close'] / group['adj_close'].shift(5))
        group['log_return_10d'] = np.log(group['adj_close'] / group['adj_close'].shift(10))
        group['log_return_21d'] = np.log(group['adj_close'] / group['adj_close'].shift(21))
        group['log_return_126d'] = np.log(group['adj_close'] / group['adj_close'].shift(126))
        
        # RSI 14 (on adj_close)
        group['rsi_14'] = calculate_rsi(group['adj_close'], period=14)
        
        # Stochastic RSI
        rsi_min = group['rsi_14'].rolling(window=14).min()
        rsi_max = group['rsi_14'].rolling(window=14).max()
        group['stoch_rsi_14'] = (group['rsi_14'] - rsi_min) / (rsi_max - rsi_min + 1e-8)
        
        # MACD (on adj_close) and MACD Histogram
        macd_line, signal_line, macd_hist = calculate_macd(group['adj_close'])
        group['macd'] = macd_line
        group['macd_signal'] = signal_line
        group['macd_hist'] = macd_hist
        
        # Bollinger Bands Width (20d, 2 std dev)
        sma20 = group['adj_close'].rolling(window=20).mean()
        std20 = group['adj_close'].rolling(window=20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        group['bb_width'] = (bb_upper - bb_lower) / (sma20 + 1e-8)
        
        # Price vs 50-day SMA
        sma50 = group['adj_close'].rolling(window=50).mean()
        group['price_vs_sma50'] = (group['adj_close'] - sma50) / (sma50 + 1e-8)
        
        # Price vs 52-week High (252 trading days)
        high_52w = group['adj_high'].rolling(window=252, min_periods=20).max()
        group['price_vs_52w_high'] = (group['adj_close'] - high_52w) / (high_52w + 1e-8)
        
        # 10-day Momentum
        group['momentum_10d'] = np.log(group['adj_close'] / group['adj_close'].shift(10))
        
        # Volume Spike (Volume relative to 30-day moving average)
        volume_ma_30 = group['volume'].rolling(window=30).mean()
        group['volume_spike'] = group['volume'] / (volume_ma_30 + 1e-8)

        # On-Balance Volume (OBV) - normalized z-score over 20d
        close_diff = group['adj_close'].diff()
        direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
        obv = (direction * group['volume']).cumsum()
        obv_mean = obv.rolling(window=20).mean()
        obv_std = obv.rolling(window=20).std()
        group['obv_normalized'] = (obv - obv_mean) / (obv_std + 1e-8)

        # Price Volume Trend (PVT) Normalized
        pvt = (group['log_return_1d'] * group['volume']).cumsum()
        pvt_mean = pvt.rolling(window=20).mean()
        pvt_std = pvt.rolling(window=20).std()
        group['pvt_normalized'] = (pvt - pvt_mean) / (pvt_std + 1e-8)

        # RSI Divergence Proxy
        ret_14d = np.log(group['adj_close'] / group['adj_close'].shift(14))
        rsi_diff_14d = group['rsi_14'] - group['rsi_14'].shift(14)
        group['rsi_divergence'] = ret_14d - (rsi_diff_14d / 100.0)

        # Realized Volatility (20d rolling std of returns)
        group['realized_vol_20d'] = group['log_return_1d'].rolling(window=20).std()
        
        # Volatility Regime (5d std / 20d std) - Is volatility expanding?
        vol_5d = group['log_return_1d'].rolling(window=5).std()
        group['volatility_regime'] = vol_5d / (group['realized_vol_20d'] + 1e-8)
        
        # ATR (Average True Range, 14d) using adjusted high, low, close
        high_low = group['adj_high'] - group['adj_low']
        high_pc = (group['adj_high'] - group['adj_close'].shift(1)).abs()
        low_pc = (group['adj_low'] - group['adj_close'].shift(1)).abs()
        tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        group['atr_14'] = tr.rolling(window=14).mean()

        # Sentiment Volume (7d rolling sum of announcement chunks analyzed)
        group['sentiment_volume_7d'] = group['total_chunks_analyzed'].fillna(0.0).rolling(window=7).sum()
        
        decayed_chunks.append(group)

    df = pd.concat(decayed_chunks, ignore_index=True)

    # 9. Sector-Relative Return (captures outperformance vs peer group)
    print("Calculating sector-relative returns and volatility...")
    sector_daily = df.groupby(['date', 'sector'])['log_return_1d'].mean().reset_index()
    sector_daily.rename(columns={'log_return_1d': 'sector_1d_return'}, inplace=True)
    
    # Calculate rolling 20-day sector volatility
    sector_daily['sector_volatility_20d'] = sector_daily.groupby('sector')['sector_1d_return'].transform(lambda x: x.rolling(20).std().fillna(0.0))
    
    df = pd.merge(df, sector_daily, on=['date', 'sector'], how='left')
    df['sector_relative_return_1d'] = df['log_return_1d'] - df['sector_1d_return']

    # 10. Merge NIFTY 50 returns
    print("Merging NIFTY50 index returns...")
    df = pd.merge(df, nifty_df, on='date', how='left')
    for col in ['nifty_50_return', 'nifty_5d_return', 'nifty_10d_return', 'nifty_21d_return', 'nifty_126d_return', 'nifty_volatility_20d']:
        df[col] = df[col].fillna(0.0)

    # 10b. Calculate Alpha Targets
    print("Calculating Alpha targets...")
    df['target_5d'] = df.groupby('ticker')['log_return_5d'].shift(-5) - df.groupby('ticker')['nifty_5d_return'].shift(-5)
    df['target_10d'] = df.groupby('ticker')['log_return_10d'].shift(-10) - df.groupby('ticker')['nifty_10d_return'].shift(-10)
    df['target_21d'] = df.groupby('ticker')['log_return_21d'].shift(-21) - df.groupby('ticker')['nifty_21d_return'].shift(-21)
    df['target_126d'] = df.groupby('ticker')['log_return_126d'].shift(-126) - df.groupby('ticker')['nifty_126d_return'].shift(-126)
    
    # Keep the default 'target' mapping to 5d for backwards compatibility with TFT scripts if needed
    df['target'] = df['target_5d']

    # 11. Generate continuous time index based on trading days
    print("Creating time index based on trading calendar...")
    unique_dates = sorted(df['date'].unique())
    date_to_idx = {d: idx for idx, d in enumerate(unique_dates)}
    df['time_idx'] = df['date'].map(date_to_idx)

    # 12. Extract calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Quarter-end trading session flag
    df['quarter_end'] = (df['date'].dt.month.isin([3, 6, 9, 12]) & df['date'].dt.is_month_end).astype(int)

    # Mock is_earnings_week based on typical intervals (quarterly announcements)
    df['is_earnings_week'] = (df['decayed_announcement_volume'] > 0.1).astype(int)

    # 13. Cleanup NaNs in features
    df['rsi_14'] = df['rsi_14'].fillna(50.0)
    df['macd'] = df['macd'].fillna(0.0)
    df['macd_signal'] = df['macd_signal'].fillna(0.0)
    df['volume_spike'] = df['volume_spike'].fillna(1.0)
    
    # Fill remaining returns
    df['log_return_1d'] = df['log_return_1d'].fillna(0.0)
    df['log_return_3d'] = df['log_return_3d'].fillna(0.0)
    df['log_return_5d'] = df['log_return_5d'].fillna(0.0)
    df['log_return_10d'] = df['log_return_10d'].fillna(0.0)
    df['log_return_21d'] = df['log_return_21d'].fillna(0.0)
    df['log_return_126d'] = df['log_return_126d'].fillna(0.0)
    
    # Fill new features NaNs
    df['realized_vol_20d'] = df['realized_vol_20d'].fillna(0.0)
    df['atr_14'] = df['atr_14'].fillna(0.0)
    df['sector_relative_return_1d'] = df['sector_relative_return_1d'].fillna(0.0)
    df['sentiment_volume_7d'] = df['sentiment_volume_7d'].fillna(0.0)
    
    df['stoch_rsi_14'] = df['stoch_rsi_14'].fillna(0.5)
    df['macd_hist'] = df['macd_hist'].fillna(0.0)
    df['bb_width'] = df['bb_width'].fillna(0.0)
    df['price_vs_sma50'] = df['price_vs_sma50'].fillna(0.0)
    df['price_vs_52w_high'] = df['price_vs_52w_high'].fillna(0.0)
    df['momentum_10d'] = df['momentum_10d'].fillna(0.0)
    df['obv_normalized'] = df['obv_normalized'].fillna(0.0)
    df['volatility_regime'] = df['volatility_regime'].fillna(1.0)
    df['pvt_normalized'] = df['pvt_normalized'].fillna(0.0)
    df['rsi_divergence'] = df['rsi_divergence'].fillna(0.0)

    print(f"Data shape after feature engineering: {df.shape}")
    print(f"Number of distinct tickers: {df['ticker'].nunique()}")
    print("Checking for NaN values in features:")
    feature_cols = [
        'time_idx', 'adj_close', 'volume', 'sector', 'industry', 'has_sentiment_history',
        'decayed_net_sentiment_score', 'decayed_bullish_intensity', 'decayed_bearish_intensity',
        'decayed_announcement_volume', 'log_return_1d', 'log_return_3d', 'log_return_5d',
        'log_return_10d', 'log_return_21d', 'log_return_126d',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike', 'nifty_50_return', 
        'nifty_5d_return', 'nifty_10d_return', 'nifty_21d_return', 'nifty_126d_return',
        'day_of_week', 'month', 'is_earnings_week',
        'realized_vol_20d', 'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d', 'quarter_end',
        'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized', 'volatility_regime',
        'pvt_normalized', 'rsi_divergence'
    ]
    nan_counts = df[feature_cols].isna().sum()
    print(nan_counts[nan_counts > 0])

    # Save to parquet
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_features.parquet')
    
    df.to_parquet(output_file)
    print(f"Features materialization complete! Saved to {output_file}")

if __name__ == '__main__':
    main()
