"""
restore_db_from_parquet.py — Rebuild the PostgreSQL tables from the saved feature matrix.

The original database and filing PDFs were lost, but training_data/processed_features.parquet
still holds every stock-day of prices plus the per-day sentiment aggregates. This script
reloads three tables from it so the rest of the pipeline (build_features, predict, dashboard)
can run again:

    companies                   ticker, sector, industry
    market_data_daily           OHLC, adj_close, volume (stocks only; NIFTY50 comes from fetch_nifty50.py)
    daily_sentiment_aggregates  net / bullish / bearish scores and chunk counts

Sentiment in the parquet was already shifted forward by one calendar day by build_features.py,
so it is shifted back here to recover the original filing date. Filings whose shifted date fell
on a non-trading day never reached the parquet and cannot be recovered from it.

Usage:
    python Cleaner/restore_db_from_parquet.py [--parquet PATH]
"""

import os
import sys
import argparse
import pandas as pd
from psycopg2.extras import execute_values

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Scaper.utils.db_connector import get_db_connection

DEFAULT_PARQUET = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'processed_features.parquet')

PRICE_COLS = ['date', 'ticker', 'open_price', 'high_price', 'low_price', 'close_price', 'adj_close', 'volume']
SENT_COLS = ['net_sentiment_score', 'bullish_intensity', 'bearish_intensity', 'total_chunks_analyzed']


def main():
    parser = argparse.ArgumentParser(description="Restore companies, prices and sentiment tables from the feature parquet.")
    parser.add_argument('--parquet', default=DEFAULT_PARQUET, help="Path to processed_features.parquet")
    args = parser.parse_args()

    print(f"Reading {args.parquet}...")
    df = pd.read_parquet(args.parquet, columns=PRICE_COLS + ['sector', 'industry'] + SENT_COLS)
    df['date'] = pd.to_datetime(df['date']).dt.date
    print(f"Loaded {len(df)} stock-days for {df['ticker'].nunique()} tickers.")

    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Companies
    companies = df.groupby('ticker')[['sector', 'industry']].first().reset_index()
    execute_values(cur, """
        INSERT INTO companies (ticker, sector, industry, is_active)
        VALUES %s
        ON CONFLICT (ticker) DO UPDATE SET sector = EXCLUDED.sector, industry = EXCLUDED.industry
    """, [(r.ticker, r.sector, r.industry, True) for r in companies.itertuples()])
    print(f"Upserted {len(companies)} companies.")

    # 2. Daily prices
    prices = df[PRICE_COLS].dropna(subset=['close_price'])
    price_rows = [
        (r.date, r.ticker, float(r.open_price), float(r.high_price), float(r.low_price),
         float(r.close_price), float(r.adj_close), int(r.volume))
        for r in prices.itertuples(index=False)
    ]
    execute_values(cur, """
        INSERT INTO market_data_daily (date, ticker, open_price, high_price, low_price, close_price, adj_close, volume)
        VALUES %s
        ON CONFLICT (date, ticker) DO NOTHING
    """, price_rows, page_size=10000)
    print(f"Inserted up to {len(price_rows)} price rows (existing rows kept).")

    # 3. Sentiment aggregates, shifted back to the filing date
    sent = df[df['total_chunks_analyzed'].notna()][['date', 'ticker'] + SENT_COLS].copy()
    sent['date'] = sent['date'] - pd.Timedelta(days=1)
    sent_rows = [
        (r.date, r.ticker, float(r.net_sentiment_score or 0.0), float(r.bullish_intensity or 0.0),
         float(r.bearish_intensity or 0.0), int(r.total_chunks_analyzed))
        for r in sent.itertuples(index=False)
    ]
    execute_values(cur, """
        INSERT INTO daily_sentiment_aggregates
            (date, ticker, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed)
        VALUES %s
        ON CONFLICT (date, ticker) DO NOTHING
    """, sent_rows, page_size=10000)
    print(f"Inserted up to {len(sent_rows)} sentiment rows.")

    conn.commit()

    for table in ('companies', 'market_data_daily', 'daily_sentiment_aggregates'):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]} rows")

    cur.close()
    conn.close()
    print("Restore complete.")


if __name__ == '__main__':
    main()
