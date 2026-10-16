import sys, os
sys.path.append(os.path.abspath('d:\\IMP\\NEED MONEY THINGS\\stock_predict'))
from Scaper.utils.db_connector import get_db_connection

conn = get_db_connection()
cursor = conn.cursor()

# Count processed
cursor.execute("""
    SELECT count(1) 
    FROM financial_reports 
    WHERE report_id IN (SELECT DISTINCT report_id FROM sentiment_chunks)
""")
processed = cursor.fetchone()[0]

# Count remaining
cursor.execute("""
    SELECT count(1) 
    FROM financial_reports 
    WHERE report_id NOT IN (SELECT DISTINCT report_id FROM sentiment_chunks)
      AND source_url IS NOT NULL 
      AND source_url != '' 
      AND source_url != '-'
""")
remaining = cursor.fetchone()[0]

# Count total
cursor.execute("""
    SELECT count(1) 
    FROM financial_reports
""")
total = cursor.fetchone()[0]

print(f"PROCESSED:{processed}")
print(f"REMAINING:{remaining}")
print(f"TOTAL:{total}")
