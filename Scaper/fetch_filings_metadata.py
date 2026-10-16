"""
fetch_filings_metadata.py — Download corporate-announcement metadata (no PDFs) from NSE.

One request per stock returns its full announcement history from NSE's public JSON API, with:
    an_dt / exchdisstime   exact filing and dissemination timestamps (to the second)
    desc                   NSE's own filing category (e.g. "Financial Result Updates")
    attchmntText           a short text summary of the filing
    attchmntFile           link to the original PDF

This recovers the filing information lost with the PDF folder, and adds what the PDFs never
gave: exact timing and a category. Output: data/filings_meta.parquet (re-downloadable).

Usage:
    python Scaper/fetch_filings_metadata.py [--from-date 01-09-2020] [--delay 1.5] [--tickers RELIANCE TCS]
"""

import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Scaper.utils.db_connector import get_db_connection

API_URL = "https://www.nseindia.com/api/corporate-announcements"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/companies-listing/corporate-filings-announcements',
}
# Our ticker -> NSE symbols to try if the original returns nothing (renamed companies)
SYMBOL_FALLBACKS = {
    'TATAMOTORS': ['TMPV'],
    'LTIM': ['LTM'],
}
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'filings_meta.parquet')
KEEP_FIELDS = ['symbol', 'seq_id', 'an_dt', 'exchdisstime', 'sort_date', 'desc', 'attchmntText',
               'attchmntFile', 'smIndustry', 'sm_name']


def fetch_symbol(session, symbol, from_date, to_date, retries=3):
    params = {'index': 'equities', 'symbol': symbol, 'from_date': from_date, 'to_date': to_date}
    for attempt in range(retries):
        try:
            r = session.get(API_URL, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else []
            print(f"  {symbol}: HTTP {r.status_code}, retry {attempt + 1}/{retries}")
        except (requests.RequestException, ValueError) as e:
            print(f"  {symbol}: {type(e).__name__}, retry {attempt + 1}/{retries}")
        time.sleep(5 * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Download NSE announcement metadata for all active stocks.")
    parser.add_argument('--from-date', default='01-09-2020', help="DD-MM-YYYY (default: start of price history)")
    parser.add_argument('--delay', type=float, default=1.5, help="Seconds between requests (be polite to NSE)")
    parser.add_argument('--tickers', nargs='+', default=None)
    args = parser.parse_args()
    to_date = datetime.now().strftime('%d-%m-%Y')

    if args.tickers:
        tickers = args.tickers
    else:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT ticker FROM companies WHERE is_active = TRUE AND ticker <> 'NIFTY50' ORDER BY ticker")
        tickers = [r[0] for r in cur.fetchall()]
        conn.close()

    session = requests.Session()
    frames, failed, empty = [], [], []
    started = time.time()
    for i, ticker in enumerate(tickers, 1):
        records, used = [], None
        for symbol in [ticker] + SYMBOL_FALLBACKS.get(ticker, []):
            result = fetch_symbol(session, symbol, args.from_date, to_date)
            time.sleep(args.delay)
            if result is None:
                continue
            if result:
                records, used = result, symbol
                break
        if not used:
            (failed if result is None else empty).append(ticker)
            print(f"[{i}/{len(tickers)}] {ticker}: no data")
            continue
        frame = pd.DataFrame(records)
        frame = frame[[c for c in KEEP_FIELDS if c in frame.columns]].copy()
        frame.insert(0, 'ticker', ticker)
        frames.append(frame)
        print(f"[{i}/{len(tickers)}] {ticker}: {len(frame)} filings" + (f" (as {used})" if used != ticker else ""))

    if not frames:
        print("Nothing downloaded.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    for col in ['an_dt', 'exchdisstime']:
        df[col] = pd.to_datetime(df[col], format='%d-%b-%Y %H:%M:%S', errors='coerce')
    df['sort_date'] = pd.to_datetime(df['sort_date'], errors='coerce')
    df = df.drop_duplicates(subset=['ticker', 'seq_id'])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} filings for {df['ticker'].nunique()} stocks to {os.path.abspath(OUT_PATH)} "
          f"in {(time.time() - started) / 60:.1f} min")
    print(f"Date range: {df['an_dt'].min()} to {df['an_dt'].max()}")
    if empty:
        print(f"No filings returned for: {empty}")
    if failed:
        print(f"Requests failed for: {failed}")


if __name__ == '__main__':
    main()
