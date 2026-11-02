import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Scaper.utils.db_connector import get_db_connection

conn = get_db_connection()
cursor = conn.cursor()

# Get unique tickers count
cursor.execute("SELECT count(distinct ticker) FROM financial_reports")
unique_tickers = cursor.fetchone()[0]

# Get top 10 tickers by document count
cursor.execute("""
    SELECT ticker, count(1) as cnt 
    FROM financial_reports 
    GROUP BY ticker 
    ORDER BY cnt DESC 
    LIMIT 10
""")
top_tickers = cursor.fetchall()

print(f"UNIQUE_TICKERS:{unique_tickers}")
print("TOP_TICKERS:")
for t, c in top_tickers:
    print(f"  {t}: {c}")
