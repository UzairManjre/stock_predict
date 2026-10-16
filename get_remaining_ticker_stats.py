import sys, os
sys.path.append(os.path.abspath('d:\\IMP\\NEED MONEY THINGS\\stock_predict'))
from Scaper.utils.db_connector import get_db_connection

conn = get_db_connection()
cursor = conn.cursor()

# Get unique tickers count for remaining
cursor.execute("""
    SELECT count(distinct ticker) 
    FROM financial_reports 
    WHERE report_id NOT IN (SELECT DISTINCT report_id FROM sentiment_chunks)
      AND source_url IS NOT NULL 
      AND source_url != '' 
      AND source_url != '-'
""")
unique_tickers_rem = cursor.fetchone()[0]

# Get top 10 tickers for remaining
cursor.execute("""
    SELECT ticker, count(1) as cnt 
    FROM financial_reports 
    WHERE report_id NOT IN (SELECT DISTINCT report_id FROM sentiment_chunks)
      AND source_url IS NOT NULL 
      AND source_url != '' 
      AND source_url != '-'
    GROUP BY ticker 
    ORDER BY cnt DESC 
    LIMIT 10
""")
top_tickers_rem = cursor.fetchall()

print(f"UNIQUE_TICKERS_REMAINING:{unique_tickers_rem}")
print("TOP_TICKERS_REMAINING:")
for t, c in top_tickers_rem:
    print(f"  {t}: {c}")
