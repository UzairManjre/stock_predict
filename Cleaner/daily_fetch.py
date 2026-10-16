import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
import psycopg2.extras
import yfinance as yf

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

# First date of the training history; used for tickers with no stored rows
HISTORY_START = "2020-09-01"

# Yahoo symbols that differ from our ticker names. Tickers keep their original names in the
# database (the trained models know them); Yahoo moved the full history to the new symbols.
YF_SYMBOL_OVERRIDES = {
    "NIFTY50": "^NSEI",
    "TATAMOTORS": "TMPV.NS",  # Tata Motors Passenger Vehicles after the 2025 demerger
    "LTIM": "LTM.NS",         # LTIMindtree renamed
}

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def fetch_single_ticker(ticker: str, start_date: str, end_date: str):
    """
    Downloads daily data for a single ticker.
    Returns a list of records ready for database insert.
    """
    # Map ticker to yfinance symbol
    yf_symbol = YF_SYMBOL_OVERRIDES.get(ticker, f"{ticker}.NS")
    try:
        df = yf.download(
            tickers=yf_symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=True,
            rounding=True
        )
        if df.empty:
            return ticker, []

        # Handle yfinance MultiIndex or single index columns
        if isinstance(df.columns, list) or hasattr(df.columns, 'levels'):
            if 'Close' in df.columns:
                pass
            else:
                # Handle MultiIndex
                df.columns = [col[0] for col in df.columns]

        def get_scalar(val):
            if hasattr(val, 'iloc'):
                return val.iloc[0]
            return val

        records = []
        for index, row in df.iterrows():
            date_val = index.date()
            try:
                open_val = float(get_scalar(row['Open']))
                high_val = float(get_scalar(row['High']))
                low_val = float(get_scalar(row['Low']))
                close_val = float(get_scalar(row['Close']))
                adj_close_val = float(get_scalar(row.get('Adj Close', row['Close'])))
                volume_val = int(get_scalar(row['Volume']))
                
                # Check for NaN and skip
                if any(x != x for x in [open_val, high_val, low_val, close_val, adj_close_val]):
                    continue
                
                records.append((
                    date_val, ticker, open_val, high_val, low_val, close_val, adj_close_val, volume_val
                ))
            except Exception:
                continue

        return ticker, records
    except Exception as e:
        print(f"Warning: Error fetching {ticker} ({yf_symbol}): {e}")
        return ticker, []

def main():
    parser = argparse.ArgumentParser(description="Fetch daily stock prices and update market_data_daily.")
    parser.add_argument('--days', type=int, default=None,
                        help="Fetch this many trailing days for every ticker. Default: resume each ticker "
                             "from its last stored date, so gaps from missed runs are back-filled.")
    parser.add_argument('--workers', type=int, default=10, help="Number of parallel fetch threads (default: 10).")
    parser.add_argument('--tickers', nargs='+', default=None, help="Only fetch these tickers (default: all active).")
    args = parser.parse_args()

    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get active tickers
    cursor.execute("SELECT ticker FROM companies WHERE is_active = TRUE")
    tickers = [row[0] for row in cursor.fetchall()]

    # Ensure NIFTY50 is processed
    if "NIFTY50" not in tickers:
        tickers.append("NIFTY50")

    if args.tickers:
        tickers = [t for t in tickers if t in set(args.tickers)]

    # Per-ticker start date: last stored date minus a 5-day overlap (re-fetches recent revisions),
    # or the full history start for tickers with no rows yet.
    cursor.execute("SELECT ticker, MAX(date) FROM market_data_daily GROUP BY ticker")
    last_dates = dict(cursor.fetchall())

    def start_for(ticker):
        if args.days is not None:
            return (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        last = last_dates.get(ticker)
        if last is None:
            return HISTORY_START
        return (last - timedelta(days=5)).strftime('%Y-%m-%d')

    starts = {t: start_for(t) for t in tickers}
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Starting daily fetch up to {end_date}; "
          f"earliest start {min(starts.values())}, latest start {max(starts.values())}")

    print(f"Found {len(tickers)} active tickers to fetch.")

    all_records = []
    failed_tickers = []
    
    start_time = time.time()
    
    # Fetch data concurrently
    print(f"Fetching data using {args.workers} threads...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_single_ticker, ticker, starts[ticker], end_date): ticker for ticker in tickers}
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, records = future.result()
                if records:
                    all_records.extend(records)
                else:
                    failed_tickers.append(ticker)
            except Exception as exc:
                print(f"Warning: Ticker {ticker} generated an exception: {exc}")
                failed_tickers.append(ticker)

    fetch_duration = time.time() - start_time
    print(f"Fetched data in {fetch_duration:.2f}s. Total records collected: {len(all_records)}")
    if failed_tickers:
        print(f"Warning: Failed to fetch {len(failed_tickers)} tickers: {failed_tickers[:15]}...")

    # Upsert to Postgres
    if all_records:
        upsert_start = time.time()
        try:
            # Using execute_values for high-speed bulk upsert
            psycopg2.extras.execute_values(cursor, """
                INSERT INTO market_data_daily (
                    date, ticker, open_price, high_price, low_price, close_price, adj_close, volume
                )
                VALUES %s
                ON CONFLICT (date, ticker) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume
            """, all_records)
            conn.commit()
            upsert_duration = time.time() - upsert_start
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [SUCCESS] Bulk upserted {len(all_records)} records into market_data_daily in {upsert_duration:.2f}s.")
        except Exception as e:
            conn.rollback()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Database write failed: {e}")
            sys.exit(1)
    else:
        print("No market records to write.")

    cursor.close()
    conn.close()
    print("Daily price fetch complete.")

if __name__ == '__main__':
    main()
