import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import re
from datetime import datetime
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
DB_PORT = os.environ.get("DB_PORT", "5433")
DB_NAME = os.environ.get("DB_NAME", "stock_predict")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

def parse_filename(filename):
    basename = os.path.splitext(filename)[0]
    
    # 1. Match YYYY-MM-DD prefix
    match_iso = re.match(r'^(\d{4}-\d{2}-\d{2})_(.*)$', basename)
    if match_iso:
        date_str = match_iso.group(1)
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except Exception:
            pass
            
    # 2. Match DD-Mmm-YYYY prefix (e.g. 22-Sep-2023)
    match_mmm = re.match(r'^(\d{1,2})-([A-Za-z]{3})-(\d{4})_(.*)$', basename)
    if match_mmm:
        date_str = f"{match_mmm.group(1)}-{match_mmm.group(2)}-{match_mmm.group(3)}"
        try:
            return datetime.strptime(date_str, '%d-%b-%Y').date()
        except Exception:
            pass
            
    return None

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
    
    # Fetch all reports to examine their URLs
    print("Fetching reports from database...")
    cursor.execute("SELECT report_id, source_url, filing_date FROM financial_reports")
    reports = cursor.fetchall()
    print(f"Found {len(reports)} total reports.")
    
    updated_reports = []
    
    print("Analyzing and parsing dates from source_url...")
    for r in tqdm(reports):
        url = r['source_url']
        if not url:
            continue
            
        filename = os.path.basename(url)
        parsed_date = parse_filename(filename)
        
        if parsed_date and parsed_date != r['filing_date']:
            updated_reports.append((parsed_date, parsed_date, r['report_id']))
            
    print(f"Identified {len(updated_reports)} reports needing date correction.")
    
    if updated_reports:
        # Perform updates in batches
        batch_size = 5000
        print(f"Updating reports in database in batches of {batch_size}...")
        for i in range(0, len(updated_reports), batch_size):
            batch = updated_reports[i:i+batch_size]
            # Batch update query
            # We can use execute_values or standard executemany
            try:
                # Update query
                cursor.executemany("""
                    UPDATE financial_reports 
                    SET filing_date = %s, received_time = %s 
                    WHERE report_id = %s
                """, batch)
                conn.commit()
                print(f"  Processed updates {i} to {i+len(batch)}")
            except Exception as e:
                conn.rollback()
                print(f"Error updating batch starting at index {i}: {e}")
                sys.exit(1)
        print("Successfully updated financial_reports dates.")
    else:
        print("No reports needed updates.")

    # Rebuild daily_sentiment_aggregates
    print("\nRebuilding daily_sentiment_aggregates table...")
    try:
        print("  Truncating old aggregates...")
        cursor.execute("TRUNCATE TABLE daily_sentiment_aggregates;")
        
        print("  Calculating and inserting new aggregates...")
        cursor.execute("""
            INSERT INTO daily_sentiment_aggregates (date, ticker, net_sentiment_score, bullish_intensity, bearish_intensity, total_chunks_analyzed)
            SELECT 
                fr.filing_date AS date,
                fr.ticker,
                COALESCE(AVG(sc.score_positive - sc.score_negative), 0.0) AS net_sentiment_score,
                COALESCE(AVG(CASE WHEN sc.dominant_label = 'positive' THEN sc.score_positive END), 0.0) AS bullish_intensity,
                COALESCE(AVG(CASE WHEN sc.dominant_label = 'negative' THEN sc.score_negative END), 0.0) AS bearish_intensity,
                COUNT(sc.chunk_id) AS total_chunks_analyzed
            FROM sentiment_chunks sc
            JOIN financial_reports fr ON sc.report_id = fr.report_id
            GROUP BY fr.filing_date, fr.ticker;
        """)
        conn.commit()
        print("Successfully rebuilt daily_sentiment_aggregates!")
    except Exception as e:
        conn.rollback()
        print(f"Error rebuilding daily_sentiment_aggregates: {e}")
        sys.exit(1)
        
    # Check final stats
    cursor.execute("SELECT COUNT(*) FROM daily_sentiment_aggregates")
    dsa_count = cursor.fetchone()['count']
    cursor.execute("SELECT MIN(date), MAX(date) FROM daily_sentiment_aggregates")
    dsa_dates = cursor.fetchone()
    
    # Overlap with price data
    cursor.execute("""
        SELECT COUNT(*) as match_count 
        FROM daily_sentiment_aggregates dsa
        JOIN market_data_daily mdd ON dsa.date = mdd.date AND dsa.ticker = mdd.ticker
    """)
    overlaps = cursor.fetchone()['match_count']
    
    print("\n--- Migration Complete Summary ---")
    print(f"Daily Sentiment Aggregates count: {dsa_count} rows")
    print(f"Sentiment Aggregates Date Range: {dsa_dates['min']} to {dsa_dates['max']}")
    print(f"Overlap with market_data_daily (ready for model training): {overlaps} rows")
    
    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()
