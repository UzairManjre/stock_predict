import sys, os
sys.path.append(os.path.abspath('d:\\IMP\\NEED MONEY THINGS\\stock_predict'))
from Scaper.utils.db_connector import get_db_connection

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("""
    SELECT count(1) 
    FROM financial_reports 
    WHERE report_id NOT IN (SELECT DISTINCT report_id FROM sentiment_chunks)
      AND source_url IS NOT NULL 
      AND source_url != '' 
      AND source_url != '-'
""")
print(cursor.fetchone()[0])
