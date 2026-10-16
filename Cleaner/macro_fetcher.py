import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_macro_data(db_conn):
    print("Fetching Global Macro data (USD/INR, Brent Crude)...")
    
    # INR=X is USD/INR, BZ=F is Brent Crude futures
    tickers = yf.Tickers("INR=X BZ=F")
    
    try:
        # Get last 5 days just in case we missed a day
        df = tickers.history(period="5d")
        
        # yfinance returns MultiIndex if multiple tickers
        close_prices = df['Close']
        
        # We take the latest available row for each
        latest_usd_inr = close_prices['INR=X'].dropna().iloc[-1]
        latest_brent = close_prices['BZ=F'].dropna().iloc[-1]
        
        print(f"  USD/INR: {latest_usd_inr:.2f}")
        print(f"  Brent Crude: {latest_brent:.2f}")
        
        # Store in db macro_daily
        cur = db_conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS macro_daily (
                date DATE PRIMARY KEY,
                nifty_pcr FLOAT,
                fii_flow_cr FLOAT,
                dii_flow_cr FLOAT,
                usd_inr FLOAT,
                brent_crude FLOAT
            )
        """)
        
        today = datetime.now().date()
        
        cur.execute("""
            INSERT INTO macro_daily (date, usd_inr, brent_crude)
            VALUES (%s, %s, %s)
            ON CONFLICT (date) DO UPDATE 
            SET usd_inr = EXCLUDED.usd_inr,
                brent_crude = EXCLUDED.brent_crude
        """, (today, float(latest_usd_inr), float(latest_brent)))
        
        db_conn.commit()
        print("Successfully updated macro data in DB.")
        
        return latest_usd_inr, latest_brent
        
    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return None, None

if __name__ == "__main__":
    # Test script standalone
    import sqlite3 # dummy for standalone test
    print("Testing yfinance Macro Fetcher...")
    conn = sqlite3.connect(':memory:')
    fetch_macro_data(conn)
