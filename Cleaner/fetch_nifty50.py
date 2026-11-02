import os
import sys
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
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
DB_PORT = os.environ.get("DB_PORT", "5433")
DB_NAME = os.environ.get("DB_NAME", "stock_predict")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

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

    try:
        # 1. Ensure NIFTY50 exists in companies
        cursor.execute("SELECT ticker FROM companies WHERE ticker = 'NIFTY50'")
        nifty_exists = cursor.fetchone()
        if not nifty_exists:
            print("NIFTY50 company record not found. Creating mock record...")
            cursor.execute("""
                INSERT INTO companies (ticker, company_name, sector, industry, is_active)
                VALUES ('NIFTY50', 'NIFTY 50 Index', 'Index', 'Index', TRUE)
            """)
            conn.commit()
            print("NIFTY50 company record created.")

        # 2. Download ^NSEI historical data via yfinance
        print("Downloading NIFTY 50 index (^NSEI) from yfinance...")
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        df = yf.download("^NSEI", start="2020-09-01", end=end_date)
        if df.empty:
            print("❌ Error: Downloaded dataframe is empty. Please check internet connection or ticker.")
            sys.exit(1)

        print(f"Downloaded {len(df)} rows. Inserting into market_data_daily...")
        
        # Flatten MultiIndex columns if present (yfinance downloads sometimes return multi-index)
        if isinstance(df.columns, list) or hasattr(df.columns, 'levels'):
            if 'Close' in df.columns:
                pass
            else:
                # Handle MultiIndex
                df.columns = [col[0] for col in df.columns]

        records = []
        for index, row in df.iterrows():
            # index is Timestamp, row holds values
            date_val = index.date()
            open_val = float(row['Open'])
            high_val = float(row['High'])
            low_val = float(row['Low'])
            close_val = float(row['Close'])
            adj_close_val = float(row.get('Adj Close', row['Close']))
            volume_val = int(row['Volume'])
            
            # Check for NaN and skip or replace
            if any(x != x for x in [open_val, high_val, low_val, close_val, adj_close_val]):
                continue
                
            records.append((
                date_val, 'NIFTY50', open_val, high_val, low_val, close_val, adj_close_val, volume_val
            ))

        print(f"Prepared {len(records)} records for insert.")
        
        cursor.executemany("""
            INSERT INTO market_data_daily (
                date, ticker, open_price, high_price, low_price, close_price, adj_close, volume
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, ticker) DO UPDATE
            SET open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                adj_close = EXCLUDED.adj_close,
                volume = EXCLUDED.volume
        """, records)
        
        conn.commit()
        print("Successfully populated NIFTY50 market data in database!")
        
        # Verify NIFTY50 count
        cursor.execute("SELECT COUNT(*) FROM market_data_daily WHERE ticker = 'NIFTY50'")
        count = cursor.fetchone()['count']
        print(f"Total NIFTY50 records in database: {count}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error occurred: {e}")
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    main()
