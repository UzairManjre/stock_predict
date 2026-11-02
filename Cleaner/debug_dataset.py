import os
import sys
import pickle
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
DB_PORT = os.environ.get("DB_PORT", "5433")
DB_NAME = os.environ.get("DB_NAME", "stock_predict")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

# Import helpers from predict.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import fetch_recent_data, build_live_features

def main():
    import psycopg2
    print(f"Connecting to {DB_NAME}@{DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        database=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )

    market_df, nifty_df, companies_df, sentiment_df = fetch_recent_data(conn, lookback_days=90)
    print(f"Loaded: market_df={len(market_df)}, nifty_df={len(nifty_df)}, companies_df={len(companies_df)}, sentiment_df={len(sentiment_df)}")

    live_df = build_live_features(market_df, nifty_df, companies_df, sentiment_df)
    print(f"Built live features: {live_df.shape}")

    # Load training dataset parameters
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    training_dataset_path = os.path.join(artifacts_dir, 'training_dataset.pkl')
    
    with open(training_dataset_path, 'rb') as f:
        training_dataset = pickle.load(f)

    print(f"Training dataset: max_encoder_length={training_dataset.max_encoder_length}, max_prediction_length={training_dataset.max_prediction_length}")
    print(f"Target: {training_dataset.target}")
    
    # 1. Print NaN counts for all variables in training_dataset
    print("\n--- Checking NaN counts in live_df ---")
    all_vars = training_dataset.flat_categoricals + training_dataset.reals
    print(f"Total variables from training dataset: {len(all_vars)}")
    
    nan_cols = {}
    for var in all_vars:
        if var in live_df.columns:
            nans = live_df[var].isna().sum()
            if nans > 0:
                nan_cols[var] = nans
        else:
            print(f"Warning: Variable missing from live_df: {var}")
            
    if nan_cols:
        print("NaNs found in columns:")
        for k, v in nan_cols.items():
            print(f"  {k}: {v} NaNs out of {len(live_df)}")
    else:
        print("No NaNs found in any training variables. Good.")

    # 2. Check length of data per ticker
    print("\n--- Checking row counts per ticker ---")
    ticker_counts = live_df['ticker'].value_counts()
    print(f"Tickers count: {len(ticker_counts)}")
    print(f"Tickers with >= 26 rows: {sum(ticker_counts >= 26)}")
    print(f"Tickers with < 26 rows: {sum(ticker_counts < 26)}")
    
    # Print a few small ones
    if sum(ticker_counts < 26) > 0:
        print("Sample tickers with < 26 rows:")
        print(ticker_counts[ticker_counts < 26].head(10))

    # 3. Test dataset creation for individual tickers
    print("\n--- Testing from_dataset on individual tickers ---")
    from pytorch_forecasting import TimeSeriesDataSet
    
    max_encoder_length = training_dataset.max_encoder_length
    max_prediction_length = training_dataset.max_prediction_length
    total_needed_length = max_encoder_length + max_prediction_length
    print(f"Total needed length for sequence construction: {total_needed_length}")
    
    # Try constructing live_df tail with total_needed_length
    live_df_tail = (
        live_df.groupby('ticker')
               .apply(lambda g: g.sort_values('time_idx').tail(total_needed_length))
               .reset_index(drop=True)
    )
    print(f"live_df_tail shape: {live_df_tail.shape}")

    success_tickers = []
    fail_tickers = {}

    for ticker in live_df_tail['ticker'].unique():
        ticker_data = live_df_tail[live_df_tail['ticker'] == ticker].copy()
        ticker_data['target'] = 0.0
        
        try:
            ds = TimeSeriesDataSet.from_dataset(
                training_dataset, ticker_data, stop_randomization=True, predict=True
            )
            success_tickers.append(ticker)
        except Exception as e:
            fail_tickers[ticker] = str(e)

    print(f"Success tickers: {len(success_tickers)}")
    print(f"Fail tickers: {len(fail_tickers)}")
    if fail_tickers:
        print("Sample failures:")
        for k, v in list(fail_tickers.items())[:5]:
            print(f"  {k}: {v}")

    conn.close()

if __name__ == '__main__':
    main()
