import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Scaper.utils.db_connector import get_db_connection

SQLITE_DB = os.path.join(os.path.dirname(__file__), 'stock_data.db')
STOCK_MAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stock_map.csv'))

def migrate():
    if not os.path.exists(SQLITE_DB):
        print(f"[ERROR] SQLite database not found at {SQLITE_DB}. Skipping data migration.")
        return
        
    print(f"Connecting to SQLite database: {SQLITE_DB}")
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_cursor = sqlite_conn.cursor()
    
    print("Connecting to PostgreSQL database...")
    pg_conn = get_db_connection()
    pg_cursor = pg_conn.cursor()
    
    # 1. Migrate Companies
    print("\n--- Migrating Companies ---")
    sqlite_cursor.execute("SELECT id, folder_name, bse_name_slug FROM stocks")
    stocks = sqlite_cursor.fetchall()
    
    # Also load stock_map.csv if exists
    stock_details = {}
    if os.path.exists(STOCK_MAP):
        try:
            df_map = pd.read_csv(STOCK_MAP)
            for _, row in df_map.iterrows():
                folder = str(row['FolderName']).strip()
                stock_details[folder] = {
                    'name': str(row['BseNameSlug']).replace('-', ' ').title(),
                    'ticker': folder
                }
        except Exception as e:
            print(f"[WARNING] Could not load stock_map.csv for company names: {e}")

    company_mapping = {}  # sqlite_id -> pg_ticker
    inserted_companies = 0
    
    for sid, folder, slug in stocks:
        ticker = folder.strip()
        company_name = stock_details.get(ticker, {}).get('name', slug.replace('-', ' ').title() if slug else ticker)
        
        # Insert or ignore in Postgres
        pg_cursor.execute("""
            INSERT INTO companies (ticker, company_name, is_active)
            VALUES (%s, %s, TRUE)
            ON CONFLICT (ticker) DO UPDATE 
            SET company_name = EXCLUDED.company_name
        """, (ticker, company_name))
        
        company_mapping[sid] = ticker
        inserted_companies += 1
        
    pg_conn.commit()
    print(f"[SUCCESS] Migrated/Updated {inserted_companies} companies.")
    
    # 2. Migrate Daily Market Data (OHLCV)
    print("\n--- Migrating Daily Market Data ---")
    sqlite_cursor.execute("SELECT COUNT(*) FROM ohlcv_data")
    total_ohlcv = sqlite_cursor.fetchone()[0]
    print(f"Found {total_ohlcv} OHLCV records in SQLite.")
    
    sqlite_cursor.execute("SELECT stock_id, date, open, high, low, close, volume FROM ohlcv_data")
    
    batch_size = 5000
    batch = []
    inserted_ohlcv = 0
    
    for row in sqlite_cursor:
        sid, date_str, open_p, high_p, low_p, close_p, vol = row
        ticker = company_mapping.get(sid)
        if not ticker:
            continue
            
        # Standardize date
        try:
            if ' ' in date_str:
                dt = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d').date()
            else:
                dt = datetime.strptime(date_str, '%Y-%m-%d').date()
        except Exception:
            continue
            
        batch.append((dt, ticker, open_p, high_p, low_p, close_p, close_p, vol))
        
        if len(batch) >= batch_size:
            pg_cursor.executemany("""
                INSERT INTO market_data_daily (date, ticker, open_price, high_price, low_price, close_price, adj_close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, ticker) DO NOTHING
            """, batch)
            inserted_ohlcv += len(batch)
            pg_conn.commit()
            print(f"   Inserted {inserted_ohlcv}/{total_ohlcv} OHLCV records...")
            batch = []
            
    if batch:
        pg_cursor.executemany("""
            INSERT INTO market_data_daily (date, ticker, open_price, high_price, low_price, close_price, adj_close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, ticker) DO NOTHING
        """, batch)
        inserted_ohlcv += len(batch)
        pg_conn.commit()
        
    print(f"[SUCCESS] Migrated {inserted_ohlcv} OHLCV records.")
    
    # 3. Migrate Announcements (to financial_reports)
    print("\n--- Migrating Corporate Announcements ---")
    sqlite_cursor.execute("SELECT COUNT(*) FROM announcements")
    total_ann = sqlite_cursor.fetchone()[0]
    print(f"Found {total_ann} announcements in SQLite.")
    
    sqlite_cursor.execute("SELECT stock_id, pdf_link, received_time, vader_score, finbert_score, hybrid_score, verdict FROM announcements")
    
    inserted_ann = 0
    for row in sqlite_cursor:
        sid, pdf_link, rec_time_str, vader, finbert, hybrid, verdict = row
        ticker = company_mapping.get(sid)
        if not ticker:
            continue
            
        # Parse date
        try:
            if ' ' in rec_time_str:
                rec_time = datetime.strptime(rec_time_str, '%d-%m-%Y %H:%M:%S')
            else:
                rec_time = datetime.strptime(rec_time_str, '%Y-%m-%d')
        except Exception:
            try:
                rec_time = datetime.strptime(rec_time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
            except Exception:
                rec_time = datetime.now()
                
        filing_date = rec_time.date()
        headline = os.path.basename(pdf_link) if pdf_link else "Announcement"
        
        # Insert announcement
        pg_cursor.execute("""
            INSERT INTO financial_reports (ticker, filing_date, report_type, source_url, headline, received_time)
            VALUES (%s, %s, 'Announcement', %s, %s, %s)
            RETURNING report_id
        """, (ticker, filing_date, pdf_link, headline, rec_time))
        
        report_id = pg_cursor.fetchone()[0]
        
        # If there are sentiment scores, populate sentiment chunks and aggregates
        if vader is not None or finbert is not None:
            # Insert a consolidated chunk for legacy data
            dominant_label = "neutral"
            if verdict and "BULLISH" in verdict:
                dominant_label = "positive"
            elif verdict and "BEARISH" in verdict:
                dominant_label = "negative"
                
            pg_cursor.execute("""
                INSERT INTO sentiment_chunks (report_id, chunk_index, chunk_text, score_positive, score_negative, score_neutral, dominant_label)
                VALUES (%s, 0, %s, %s, %s, %s, %s)
            """, (
                report_id,
                f"Legacy announcement PDF: {headline}",
                float(finbert) if finbert and finbert > 0 else 0.0,
                float(-finbert) if finbert and finbert < 0 else 0.0,
                float(1.0 - abs(finbert)) if finbert else 1.0,
                dominant_label
            ))
            
            # Update daily aggregates
            pg_cursor.execute("""
                INSERT INTO daily_sentiment_aggregates (date, ticker, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed)
                VALUES (%s, %s, %s, %s, %s, 1)
                ON CONFLICT (date, ticker) DO UPDATE SET
                    net_sentiment_score = (daily_sentiment_aggregates.net_sentiment_score * daily_sentiment_aggregates.total_chunks_analyzed + EXCLUDED.net_sentiment_score) / (daily_sentiment_aggregates.total_chunks_analyzed + 1),
                    bullish_intensity = (daily_sentiment_aggregates.bullish_intensity * daily_sentiment_aggregates.total_chunks_analyzed + EXCLUDED.bullish_intensity) / (daily_sentiment_aggregates.total_chunks_analyzed + 1),
                    bearish_intensity = (daily_sentiment_aggregates.bearish_intensity * daily_sentiment_aggregates.total_chunks_analyzed + EXCLUDED.bearish_intensity) / (daily_sentiment_aggregates.total_chunks_analyzed + 1),
                    total_chunks_analyzed = daily_sentiment_aggregates.total_chunks_analyzed + 1
            """, (
                filing_date,
                ticker,
                float(hybrid) if hybrid is not None else 0.0,
                float(finbert) if finbert and finbert > 0 else 0.0,
                float(-finbert) if finbert and finbert < 0 else 0.0
            ))
            
        inserted_ann += 1
        if inserted_ann % 500 == 0:
            pg_conn.commit()
            print(f"   Inserted {inserted_ann}/{total_ann} announcements...")
            
    pg_conn.commit()
    print(f"[SUCCESS] Migrated {inserted_ann} announcements, chunks, and aggregates.")
    
    sqlite_conn.close()
    pg_conn.close()
    print("\n[SUCCESS] Migration completed successfully!")

if __name__ == '__main__':
    migrate()
