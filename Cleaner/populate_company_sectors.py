import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

def fetch_company_info(ticker):
    # Clean ticker if needed (e.g. replacing '-' with '')
    # For Yahoo Finance, some tickers might have dash like BAJAJ-AUTO -> BAJAJAUTO or BAJAJ-AUTO
    ticker_variants = [ticker]
    if '-' in ticker:
        ticker_variants.append(ticker.replace('-', ''))
        
    for tok in ticker_variants:
        for suffix in [".NS", ".BO"]:
            yf_ticker = f"{tok}{suffix}"
            try:
                t = yf.Ticker(yf_ticker)
                info = t.info
                sector = info.get("sector")
                industry = info.get("industry")
                if sector:
                    return ticker, sector, industry
            except Exception:
                pass
    return ticker, None, None

def main():
    print(f"Connecting to database {DB_NAME} on {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Query tickers
    cursor.execute("SELECT ticker FROM companies ORDER BY ticker")
    rows = cursor.fetchall()
    tickers = [r['ticker'] for r in rows]
    print(f"Found {len(tickers)} companies in database.")
    
    results = []
    failed = []
    print("Fetching sector/industry information via yfinance (multi-threaded)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_company_info, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(futures)):
            ticker, sector, industry = future.result()
            if sector:
                results.append((sector, industry, ticker))
            else:
                failed.append(ticker)
                
    print(f"Successfully fetched details for {len(results)} / {len(tickers)} companies.")
    if failed:
        print(f"Failed to fetch info for: {failed[:15]}... ({len(failed)} total failed)")
    
    if results:
        print("Updating database...")
        try:
            cursor.executemany("""
                UPDATE companies 
                SET sector = %s, industry = %s 
                WHERE ticker = %s
            """, results)
            conn.commit()
            print("Database update committed successfully!")
        except Exception as e:
            conn.rollback()
            print(f"Database error: {e}")
            sys.exit(1)
            
    # Verify changes
    cursor.execute("SELECT sector, COUNT(*) as count FROM companies GROUP BY sector ORDER BY count DESC")
    print("\n--- Current Sector Distribution ---")
    for r in cursor.fetchall():
        print(f"  {r['sector'] or 'NULL'}: {r['count']} companies")
        
    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()
