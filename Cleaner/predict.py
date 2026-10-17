"""
predict.py — Live inference pipeline for the TFT stock prediction system.

Usage:
    python predict.py [--checkpoint PATH] [--output-table predictions] [--no-db-write]

Workflow:
    1. Load the most recent N rows from PostgreSQL (encoder_length + headroom days)
    2. Apply the SAME build_features transforms using SAVED scalers (no refit — avoids leakage)
    3. Load best checkpoint
    4. Generate q10 / q50 / q90 for each ticker across the next 5 trading days
    5. Write results to `predictions` table in PostgreSQL (or print to stdout)

Critical note on scalers:
    PyTorch Forecasting's TimeSeriesDataSet does its own internal normalization. The
    "scalers" we save here are used to detect distribution drift via KS test only.
    The actual scaling for the model is managed by the saved dataset parameters inside
    the checkpoint, so we never refit on live data.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from build_features import shift_sentiment_to_next_trading_day

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Load .env ────────────────────────────────────────────────────────────────
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

DB_HOST     = os.environ.get("DB_HOST",     "localhost")
DB_PORT     = os.environ.get("DB_PORT",     "5432")
DB_NAME     = os.environ.get("DB_NAME",     "stock_predict")
DB_USER     = os.environ.get("DB_USER",     "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'training_data')

# ─── Feature helpers (mirror build_features.py exactly) ───────────────────────

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).copy()
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


# ─── Drift detection ──────────────────────────────────────────────────────────

def run_ks_drift_check(live_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Compares live feature distributions against the training baseline stored
    in artifacts/feature_stats.json using the Kolmogorov-Smirnov test.
    Returns a dict mapping feature -> p-value (low p = drift detected).
    """
    stats_path = os.path.join(ARTIFACTS_DIR, 'feature_stats.json')
    if not os.path.exists(stats_path):
        print("[DRIFT] No baseline feature_stats.json found. Skipping drift check.")
        return {}

    from scipy.stats import ks_2samp

    with open(stats_path, 'r') as f:
        baseline_stats = json.load(f)

    drift_results = {}
    drifted_features = []

    for col in feature_cols:
        if col not in baseline_stats:
            continue
        baseline_samples = baseline_stats[col].get('samples', [])
        if not baseline_samples or col not in live_df.columns:
            continue
        live_vals = live_df[col].dropna().values
        if len(live_vals) < 10:
            continue
        stat, p_value = ks_2samp(baseline_samples, live_vals)
        drift_results[col] = {'ks_stat': float(stat), 'p_value': float(p_value)}
        if p_value < 0.05:
            drifted_features.append(col)

    if drifted_features:
        print(f"\n[DRIFT] WARNING: Distribution drift detected in {len(drifted_features)} feature(s):")
        for feat in drifted_features:
            r = drift_results[feat]
            print(f"    {feat:40s}  KS={r['ks_stat']:.4f}  p={r['p_value']:.4f}")
        print("[DRIFT] Consider retraining if drift is persistent across multiple runs.\n")
    else:
        print("[DRIFT] No significant feature drift detected.")

    return drift_results


# ─── Data loading ─────────────────────────────────────────────────────────────

def fetch_recent_data(conn, lookback_days: int = 90) -> pd.DataFrame:
    """
    Pulls the most recent `lookback_days` calendar days of market and sentiment
    data from PostgreSQL. We fetch extra history so the encoder (21d) has enough
    context and rolling windows (30d volume MA, 20d vol, etc.) are fully warm.
    """
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    print(f"Fetching market data since {cutoff}...")

    market_df = pd.read_sql_query(f"""
        SELECT date, ticker, open_price, high_price, low_price, close_price, adj_close, volume
        FROM market_data_daily
        WHERE ticker != 'NIFTY50' AND date >= '{cutoff}'
        ORDER BY ticker, date
    """, conn, parse_dates=['date'])

    nifty_df = pd.read_sql_query(f"""
        SELECT date, close_price AS nifty_close
        FROM market_data_daily
        WHERE ticker = 'NIFTY50' AND date >= '{cutoff}'
        ORDER BY date
    """, conn, parse_dates=['date'])

    companies_df = pd.read_sql_query("""
        SELECT ticker, sector, industry
        FROM companies
        WHERE ticker != 'NIFTY50'
    """, conn)

    sentiment_df = pd.read_sql_query(f"""
        SELECT date, ticker, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed
        FROM daily_sentiment_aggregates
        WHERE date >= '{cutoff}'
        ORDER BY ticker, date
    """, conn, parse_dates=['date'])

    return market_df, nifty_df, companies_df, sentiment_df


# ─── Feature engineering (stateless — no scaler refit) ────────────────────────

def build_live_features(market_df, nifty_df, companies_df, sentiment_df) -> pd.DataFrame:
    """
    Applies the SAME transformations as build_features.py, but uses only
    the data passed in (no DB access). Scalers are NOT refit — PyTorch
    Forecasting applies the saved scaler params from the checkpoint.
    """
    # Sentiment moves to the next trading session (anti-lookahead; same rule as training)
    sentiment_df = shift_sentiment_to_next_trading_day(sentiment_df, market_df['date'])

    tickers_with_sentiment = set(sentiment_df['ticker'].unique())

    # NIFTY returns, aligned to the stock trading calendar (same rule as training)
    calendar = pd.DatetimeIndex(sorted(market_df['date'].unique()), name='date')
    nifty_df = nifty_df.set_index('date').reindex(calendar).ffill().bfill().reset_index()
    nifty_df['nifty_50_return'] = np.log(
        nifty_df['nifty_close'] / nifty_df['nifty_close'].shift(1)
    ).fillna(0.0)
    nifty_df['nifty_volatility_20d'] = nifty_df['nifty_50_return'].rolling(window=20).std().fillna(0.0)
    nifty_df = nifty_df[['date', 'nifty_50_return', 'nifty_volatility_20d']]

    df = pd.merge(market_df, companies_df, on='ticker', how='left')
    df['has_sentiment_history'] = df['ticker'].isin(tickers_with_sentiment).astype(int)
    df = pd.merge(df, sentiment_df, on=['date', 'ticker'], how='left')
    df.sort_values(['ticker', 'date'], inplace=True)

    # Liquidity-based sentiment decay half-life (same logic as training)
    df['dollar_volume'] = df['volume'] * df['close_price']
    dollar_volume = df.groupby('ticker')['dollar_volume'].median()
    q20 = dollar_volume.quantile(0.20)
    q80 = dollar_volume.quantile(0.80)
    half_life_map = {}
    for ticker, vol in dollar_volume.items():
        if vol >= q80:
            half_life_map[ticker] = 1.5
        elif vol <= q20:
            half_life_map[ticker] = 6.0
        else:
            half_life_map[ticker] = 4.0

    decay_cols = ['net_sentiment_score', 'bullish_intensity', 'bearish_intensity', 'total_chunks_analyzed']
    chunks = []

    for ticker, group in df.groupby('ticker'):
        group = group.copy()

        adj_factor = group['adj_close'] / (group['close_price'] + 1e-8)
        group['adj_open']  = group['open_price']  * adj_factor
        group['adj_high']  = group['high_price']  * adj_factor
        group['adj_low']   = group['low_price']   * adj_factor

        hl = half_life_map.get(ticker, 4.0)
        for col in decay_cols:
            col_name = f'decayed_{col}' if col != 'total_chunks_analyzed' else 'decayed_announcement_volume'
            group[col_name] = decay_forward_fill(group[col], half_life=hl)

        group['log_return_1d'] = np.log(group['adj_close'] / group['adj_close'].shift(1)).fillna(0.0)
        group['log_return_3d'] = np.log(group['adj_close'] / group['adj_close'].shift(3)).fillna(0.0)
        group['log_return_5d'] = np.log(group['adj_close'] / group['adj_close'].shift(5)).fillna(0.0)

        # Target is unknown for live data — set 0.0 to pass validation check
        group['target'] = 0.0

        group['rsi_14'] = calculate_rsi(group['adj_close'], period=14).fillna(50.0)

        # Stochastic RSI
        rsi_min = group['rsi_14'].rolling(window=14).min()
        rsi_max = group['rsi_14'].rolling(window=14).max()
        group['stoch_rsi_14'] = ((group['rsi_14'] - rsi_min) / (rsi_max - rsi_min + 1e-8)).fillna(0.5)

        macd_line, signal_line, macd_hist = calculate_macd(group['adj_close'])
        group['macd']        = macd_line.fillna(0.0)
        group['macd_signal'] = signal_line.fillna(0.0)
        group['macd_hist']   = macd_hist.fillna(0.0)

        # Bollinger Bands Width
        sma20 = group['adj_close'].rolling(window=20).mean()
        std20 = group['adj_close'].rolling(window=20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        group['bb_width'] = ((bb_upper - bb_lower) / (sma20 + 1e-8)).fillna(0.0)

        # Price Volume Trend (PVT) Normalized
        pvt = (group['log_return_1d'] * group['volume']).cumsum()
        pvt_mean = pvt.rolling(window=20).mean()
        pvt_std = pvt.rolling(window=20).std()
        group['pvt_normalized'] = ((pvt - pvt_mean) / (pvt_std + 1e-8)).fillna(0.0)

        # RSI Divergence Proxy
        ret_14d = np.log(group['adj_close'] / group['adj_close'].shift(14))
        rsi_diff_14d = group['rsi_14'] - group['rsi_14'].shift(14)
        group['rsi_divergence'] = (ret_14d - (rsi_diff_14d / 100.0)).fillna(0.0)

        # Price vs 50-day SMA
        sma50 = group['adj_close'].rolling(window=50).mean()
        group['price_vs_sma50'] = ((group['adj_close'] - sma50) / (sma50 + 1e-8)).fillna(0.0)

        # Price vs 52-week High
        high_52w = group['adj_high'].rolling(window=252, min_periods=20).max()
        group['price_vs_52w_high'] = ((group['adj_close'] - high_52w) / (high_52w + 1e-8)).fillna(0.0)

        # 10-day Momentum
        group['momentum_10d'] = np.log(group['adj_close'] / group['adj_close'].shift(10)).fillna(0.0)

        # 12-1 month momentum (return from 12 months ago to 1 month ago); NaN without a year of history
        group['momentum_12_1'] = np.log(group['adj_close'].shift(21) / group['adj_close'].shift(252))

        volume_ma_30 = group['volume'].rolling(window=30).mean()
        group['volume_spike'] = (group['volume'] / (volume_ma_30 + 1e-8)).fillna(1.0)

        # On-Balance Volume (OBV)
        close_diff = group['adj_close'].diff()
        direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
        obv = (direction * group['volume']).cumsum()
        obv_mean = obv.rolling(window=20).mean()
        obv_std = obv.rolling(window=20).std()
        group['obv_normalized'] = ((obv - obv_mean) / (obv_std + 1e-8)).fillna(0.0)

        group['realized_vol_20d'] = group['log_return_1d'].rolling(window=20).std().fillna(0.0)

        # Volatility Regime
        vol_5d = group['log_return_1d'].rolling(window=5).std()
        group['volatility_regime'] = (vol_5d / (group['realized_vol_20d'] + 1e-8)).fillna(1.0)

        high_low  = group['adj_high'] - group['adj_low']
        high_pc   = (group['adj_high'] - group['adj_close'].shift(1)).abs()
        low_pc    = (group['adj_low']  - group['adj_close'].shift(1)).abs()
        tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        group['atr_14'] = tr.rolling(window=14).mean().fillna(0.0)

        group['sentiment_volume_7d'] = group['total_chunks_analyzed'].fillna(0.0).rolling(window=7).sum().fillna(0.0)

        chunks.append(group)

    df = pd.concat(chunks, ignore_index=True)

    # Sector-relative return
    sector_daily = df.groupby(['date', 'sector'])['log_return_1d'].mean().reset_index()
    sector_daily.rename(columns={'log_return_1d': 'sector_1d_return'}, inplace=True)
    sector_daily['sector_volatility_20d'] = sector_daily.groupby('sector')['sector_1d_return'].transform(lambda x: x.rolling(20).std().fillna(0.0))
    df = pd.merge(df, sector_daily, on=['date', 'sector'], how='left')
    df['sector_relative_return_1d'] = (df['log_return_1d'] - df['sector_1d_return']).fillna(0.0)

    # Merge NIFTY
    df = pd.merge(df, nifty_df, on='date', how='left')
    df['nifty_50_return'] = df['nifty_50_return'].fillna(0.0)
    df['nifty_volatility_20d'] = df['nifty_volatility_20d'].fillna(0.0)

    # Time index (must align with global training time_idx)
    # Load the saved date-to-idx mapping from artifacts
    date_map_path = os.path.join(ARTIFACTS_DIR, 'date_to_idx.json')
    if os.path.exists(date_map_path):
        with open(date_map_path, 'r') as f:
            raw_map = json.load(f)
        date_to_idx = {pd.Timestamp(k): v for k, v in raw_map.items()}
        max_known_idx = max(date_to_idx.values())
        def get_time_idx(d):
            if d in date_to_idx:
                return date_to_idx[d]
            # Extend forward from max known idx for new trading days
            return max_known_idx + (df['date'].unique().tolist().index(d)
                                    if d not in date_to_idx else 0)
        df['time_idx'] = df['date'].map(date_to_idx)
        new_dates_sorted = sorted(df[df['time_idx'].isna()]['date'].unique())
        for pos, d in enumerate(new_dates_sorted):
            df.loc[df['date'] == d, 'time_idx'] = max_known_idx + pos + 1
        df['time_idx'] = df['time_idx'].astype(int)
    else:
        print("[WARN] date_to_idx.json not found. Building local time_idx (may not align with training).")
        unique_dates = sorted(df['date'].unique())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        df['time_idx'] = df['date'].map(date_to_idx).astype(int)

    # Calendar features — must match dtype used in training
    df['day_of_week']    = df['date'].dt.dayofweek.astype(str)
    df['month']          = df['date'].dt.month.astype(str)
    df['quarter_end']    = (df['date'].dt.month.isin([3, 6, 9, 12]) & df['date'].dt.is_month_end).astype(int).astype(str)
    df['is_earnings_week'] = (df['decayed_announcement_volume'] > 0.1).astype(int).astype(str)
    df['has_sentiment_history'] = df['has_sentiment_history'].astype(str)

    df['ticker']   = df['ticker'].astype(str)
    df['sector']   = df['sector'].astype(str)
    df['industry'] = df['industry'].astype(str)

    real_cols = [
        'adj_close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike', 'nifty_50_return',
        'decayed_net_sentiment_score', 'decayed_bullish_intensity', 'decayed_bearish_intensity',
        'log_return_1d', 'log_return_3d', 'log_return_5d',
        'realized_vol_20d', 'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d',
        'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized', 'volatility_regime',
        'pvt_normalized', 'rsi_divergence', 'nifty_volatility_20d', 'sector_volatility_20d'
    ]
    for col in real_cols:
        df[col] = df[col].astype('float32')

    return df


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame, checkpoint_path: str) -> pd.DataFrame:
    """
    Loads the TFT checkpoint and generates q10/q50/q90 for the latest row
    of each ticker (the next 5 trading days).
    """
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    import pickle

    print(f"\nLoading TFT checkpoint: {checkpoint_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Inference device: {device}")

    tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    tft.eval()
    tft = tft.to(device)

    # Apply FP16 patch if running on GPU
    if device == 'cuda':
        for name, module in tft.named_modules():
            if hasattr(module, 'mask_bias') and hasattr(module.mask_bias, 'data'):
                module.mask_bias.data = torch.tensor(-65500.0,
                    dtype=module.mask_bias.dtype, device=module.mask_bias.device)
        print("Applied FP16 mask_bias patch for GPU inference.")

    training_dataset_path = os.path.join(ARTIFACTS_DIR, 'training_dataset.pkl')
    with open(training_dataset_path, 'rb') as f:
        training_dataset: TimeSeriesDataSet = pickle.load(f)

    horizons = ['5d', '10d', '21d', '126d']
    all_results = []
    generated_at = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # Base snapshot for LGBM
    tickers = list(df.groupby('ticker').tail(1)['ticker'])
    last_rows = df.groupby('ticker').tail(1).set_index('ticker').loc[tickers].reset_index()
    
    categorical_cols = ["ticker", "sector", "industry", "day_of_week", "month", "is_earnings_week", "has_sentiment_history", "quarter_end"]
    real_cols = [
        'adj_close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike', 'nifty_50_return',
        'decayed_net_sentiment_score', 'decayed_bullish_intensity', 'decayed_bearish_intensity',
        'log_return_1d', 'log_return_3d', 'log_return_5d',
        'realized_vol_20d', 'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d',
        'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized', 'volatility_regime',
        'pvt_normalized', 'rsi_divergence', 'nifty_volatility_20d', 'sector_volatility_20d'
    ]
    
    features = categorical_cols + real_cols
    for col in categorical_cols:
        last_rows[col] = last_rows[col].astype('category')
    for col in real_cols:
        last_rows[col] = last_rows[col].astype('float32')
    X_live = last_rows[features]

    # 12-1 month momentum and its rank across stocks (1 = strongest). In the walk-forward
    # evaluation it ranked stocks better than every model, so it is published beside the bands.
    momentum = last_rows['momentum_12_1'].to_numpy(dtype=float)
    momentum_rank = pd.Series(momentum).rank(ascending=False, method='min').to_numpy()

    for horizon in horizons:
        print(f"\n--- Predicting for Horizon: {horizon.upper()} ---")
        
        # --- LIGHTGBM INFERENCE ---
        import lightgbm as lgb
        lgbm_q10 = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, f'lgbm_model_{horizon}_q10.txt'))
        lgbm_q50 = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, f'lgbm_model_{horizon}_q50.txt'))
        lgbm_q90 = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, f'lgbm_model_{horizon}_q90.txt'))
        
        q10_lgbm = lgbm_q10.predict(X_live)
        q50_lgbm = lgbm_q50.predict(X_live)
        q90_lgbm = lgbm_q90.predict(X_live)

        if horizon == '5d':
            print("Generating TFT predictions for 5d...")
            tft_df = df.copy()

            # predict=True treats each stock's final row as the decoder step. Append one future
            # session per stock so that step lies AFTER the last known day; otherwise the
            # "forecast" covers a day whose prices are already known. Its unknown inputs are
            # masked by the TFT; only the calendar (known) inputs are used.
            future = tft_df.sort_values('time_idx').groupby('ticker').tail(1).copy()
            next_day = future['date'] + pd.offsets.BDay(1)
            future['date'] = next_day
            future['time_idx'] = future['time_idx'] + 1
            future['day_of_week'] = next_day.dt.dayofweek.astype(str)
            future['month'] = next_day.dt.month.astype(str)
            future['quarter_end'] = (next_day.dt.month.isin([3, 6, 9, 12]) & next_day.dt.is_month_end).astype(int).astype(str)
            future['target'] = 0.0
            tft_df = pd.concat([tft_df, future], ignore_index=True)

            live_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset, tft_df, stop_randomization=True, predict=True
            )
            live_loader = live_dataset.to_dataloader(batch_size=64, shuffle=False, num_workers=0)

            predict_kwargs = dict(accelerator="gpu") if device == 'cuda' else dict(accelerator="cpu")
            with torch.no_grad():
                out = tft.predict(live_loader, mode='quantiles', return_index=True, trainer_kwargs=predict_kwargs)

            pred_tensor = out.output  # [N, 1, 3] for q10/q50/q90

            # The TFT returns stocks in its dataset's order; align them to the LightGBM order
            tft_tickers = np.asarray(out.index['ticker']).astype(str)
            def tft_quantile(i):
                values = pred_tensor[..., i].mean(dim=1).cpu().numpy()
                return pd.Series(values, index=tft_tickers).reindex(tickers).to_numpy()
            q10_tft, q50_tft, q90_tft = tft_quantile(0), tft_quantile(1), tft_quantile(2)

            print("Blending TFT and LightGBM predictions (60/40)...")
            # Stocks the TFT could not score (too little history) fall back to LightGBM alone
            blend = lambda tft_q, lgbm_q: np.where(np.isnan(tft_q), lgbm_q, tft_q * 0.6 + lgbm_q * 0.4)
            q10 = blend(q10_tft, q10_lgbm)
            q50 = blend(q50_tft, q50_lgbm)
            q90 = blend(q90_tft, q90_lgbm)
        else:
            print("Using 100% LightGBM predictions...")
            q10, q50, q90 = q10_lgbm, q50_lgbm, q90_lgbm

        horizon_df = pd.DataFrame({
            'generated_at':  generated_at,
            'ticker':        tickers,
            'horizon':       horizon,
            'q10_return':    q10,
            'q50_return':    q50,
            'q90_return':    q90,
            'spread':        q90 - q10,
            'signal_strength': q50 / (q90 - q10 + 1e-8),
            'momentum_12_1': momentum,
            'momentum_rank': momentum_rank,
        })
        all_results.append(horizon_df)

    results = pd.concat(all_results, ignore_index=True)
    results = results.sort_values(['horizon', 'q50_return'], ascending=[True, False])
    return results


# ─── Final model: frozen 21-day blend and the day's confidence flag ───────────

def add_final_model(results: pd.DataFrame, record: bool) -> pd.DataFrame:
    """Adds final_model.py's blend score and rank to every row, plus the day's model agreement,
    threshold and high-confidence flag. Skipped with a warning if its artifacts are missing."""
    try:
        import final_model
        if not final_model.artifacts_ready():
            print("[WARN] Final-model artifacts missing; run Cleaner/final_model.py --train. Skipping the blend.")
            return results
        picks, day = final_model.score_latest(record=record)
    except Exception as e:
        print(f"[WARN] Final-model scoring failed, predictions are written without it: {e}")
        return results
    results = results.merge(picks, on='ticker', how='left')
    results['blend_date'] = pd.Timestamp(day['date']).date()
    results['model_agreement'] = day['agreement']
    results['agreement_threshold'] = day['threshold']
    results['high_confidence'] = day['high_confidence']
    label = 'HIGH CONFIDENCE' if day['high_confidence'] else 'normal confidence'
    print(f"Final model ({day['date']}): agreement {day['agreement']:.3f} vs threshold {day['threshold']:.3f} -> {label}")
    return results


# ─── DB write ─────────────────────────────────────────────────────────────────

def write_predictions_to_db(conn, results: pd.DataFrame):
    """
    Upserts prediction results into the `predictions` table.
    Creates the table if it does not exist.
    """
    import psycopg2.extras
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              SERIAL PRIMARY KEY,
            generated_at    TIMESTAMP NOT NULL,
            ticker          TEXT NOT NULL,
            horizon         TEXT NOT NULL,
            q10_return      FLOAT,
            q50_return      FLOAT,
            q90_return      FLOAT,
            spread          FLOAT,
            signal_strength FLOAT,
            UNIQUE (generated_at, ticker, horizon)
        );
    """)
    # Added after the table was first created
    cur.execute("""
        ALTER TABLE predictions
            ADD COLUMN IF NOT EXISTS momentum_12_1 FLOAT,
            ADD COLUMN IF NOT EXISTS momentum_rank INTEGER,
            ADD COLUMN IF NOT EXISTS blend_score FLOAT,
            ADD COLUMN IF NOT EXISTS blend_rank INTEGER,
            ADD COLUMN IF NOT EXISTS blend_date DATE,
            ADD COLUMN IF NOT EXISTS model_agreement FLOAT,
            ADD COLUMN IF NOT EXISTS agreement_threshold FLOAT,
            ADD COLUMN IF NOT EXISTS high_confidence BOOLEAN;
    """)

    def nullable(value, cast):
        return None if pd.isna(value) else cast(value)

    rows = [
        (
            row['generated_at'],
            row['ticker'],
            row['horizon'],
            row['q10_return'],
            row['q50_return'],
            row['q90_return'],
            row['spread'],
            row['signal_strength'],
            nullable(row['momentum_12_1'], float),
            nullable(row['momentum_rank'], int),
            nullable(row.get('blend_score'), float),
            nullable(row.get('blend_rank'), int),
            nullable(row.get('blend_date'), lambda v: v),
            nullable(row.get('model_agreement'), float),
            nullable(row.get('agreement_threshold'), float),
            nullable(row.get('high_confidence'), bool),
        )
        for _, row in results.iterrows()
    ]

    psycopg2.extras.execute_values(cur, """
        INSERT INTO predictions
            (generated_at, ticker, horizon, q10_return, q50_return, q90_return, spread, signal_strength,
             momentum_12_1, momentum_rank, blend_score, blend_rank, blend_date, model_agreement,
             agreement_threshold, high_confidence)
        VALUES %s
        ON CONFLICT (generated_at, ticker, horizon) DO UPDATE
            SET q10_return      = EXCLUDED.q10_return,
                q50_return      = EXCLUDED.q50_return,
                q90_return      = EXCLUDED.q90_return,
                spread          = EXCLUDED.spread,
                signal_strength = EXCLUDED.signal_strength,
                momentum_12_1   = EXCLUDED.momentum_12_1,
                momentum_rank   = EXCLUDED.momentum_rank,
                blend_score     = EXCLUDED.blend_score,
                blend_rank      = EXCLUDED.blend_rank,
                blend_date      = EXCLUDED.blend_date,
                model_agreement = EXCLUDED.model_agreement,
                agreement_threshold = EXCLUDED.agreement_threshold,
                high_confidence = EXCLUDED.high_confidence;
    """, rows)

    conn.commit()
    cur.close()
    print(f"Wrote {len(rows)} predictions to database.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TFT live inference — generates 5-day quantile forecasts for all tickers."
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to .ckpt file. If omitted, uses the latest checkpoint in lightning_logs/.'
    )
    parser.add_argument(
        '--lookback-days', type=int, default=400,
        help='Calendar days of history to pull from DB (default: 400, covers 60 encoder days + 252d rolling windows comfortably).'
    )
    parser.add_argument(
        '--no-db-write', action='store_true',
        help='Print predictions to stdout only; do not write to the predictions table.'
    )
    parser.add_argument(
        '--drift-check', action='store_true',
        help='Run KS drift detection on live features vs. training baseline.'
    )
    args = parser.parse_args()

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    checkpoint_path = args.checkpoint
    registry_path = os.path.join(ARTIFACTS_DIR, 'model_registry.json')
    if checkpoint_path is None and os.path.exists(registry_path):
        # The checkpoint train_tft.py recorded as its best model
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', registry['tft_checkpoint']))
        print(f"Using registered checkpoint: {checkpoint_path} (trained {registry.get('trained_at')})")
    if checkpoint_path is None:
        # Fallback: auto-discover latest checkpoint (look in workspace root, one level up)
        import glob
        pattern = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '**', '*.ckpt'))
        ckpts = glob.glob(pattern, recursive=True)
        # Filter and prioritize best checkpoints
        best_ckpts = [c for c in ckpts if 'best-tft' in os.path.basename(c)]
        if best_ckpts:
            checkpoint_path = max(best_ckpts, key=os.path.getmtime)
        elif ckpts:
            checkpoint_path = max(ckpts, key=os.path.getmtime)
        else:
            print("[ERROR] No checkpoint found. Train the model first with train_tft.py.")
            sys.exit(1)
        print(f"Auto-discovered checkpoint: {checkpoint_path}")

    # ── Connect to DB ────────────────────────────────────────────────────────
    import psycopg2
    print(f"Connecting to {DB_NAME}@{DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        database=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )

    # ── Fetch data ───────────────────────────────────────────────────────────
    market_df, nifty_df, companies_df, sentiment_df = fetch_recent_data(
        conn, lookback_days=args.lookback_days
    )

    # ── Build features ───────────────────────────────────────────────────────
    print("Building live feature matrix...")
    live_df = build_live_features(market_df, nifty_df, companies_df, sentiment_df)
    print(f"Live feature matrix: {live_df.shape}")

    # ── Drift check ──────────────────────────────────────────────────────────
    if args.drift_check:
        NUMERIC_FEATURES = [
            'adj_close', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_spike', 'nifty_50_return',
            'decayed_net_sentiment_score', 'log_return_1d', 'realized_vol_20d',
            'atr_14', 'sector_relative_return_1d', 'sentiment_volume_7d',
            'stoch_rsi_14', 'bb_width', 'price_vs_sma50', 'price_vs_52w_high', 'momentum_10d', 'obv_normalized', 'volatility_regime'
        ]
        run_ks_drift_check(live_df, NUMERIC_FEATURES)

    # ── Inference ────────────────────────────────────────────────────────────
    results = run_inference(live_df, checkpoint_path)
    results = add_final_model(results, record=not args.no_db_write)

    # ── Output ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  PREDICTIONS — Generated at {results['generated_at'].iloc[0]}")
    for horizon in ['5d', '10d', '21d', '126d']:
        print(f"\n  Top 5 tickers for horizon {horizon.upper()}:")
        print("=" * 70)
        display_cols = ['ticker', 'q10_return', 'q50_return', 'q90_return', 'spread', 'signal_strength']
        h_df = results[results['horizon'] == horizon].head(5)
        print(h_df[display_cols].to_string(index=False, float_format='{:.5f}'.format))
    print("=" * 70)

    # Save to CSV regardless of DB flag
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    csv_path = os.path.join(ARTIFACTS_DIR, f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nSaved predictions CSV to: {csv_path}")

    if not args.no_db_write:
        write_predictions_to_db(conn, results)

    conn.close()
    print("\nInference complete.")


if __name__ == '__main__':
    main()
