
import os
import pandas as pd
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')
ANN_CSV = os.path.join(os.path.dirname(__file__), 'all_announcements_with_sentiment.csv')

DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'market_data_20250920_225539', 'individual_data')

# Helper to parse date
parse_date = lambda d: datetime.strptime(d, '%Y-%m-%d') if '-' in d else d


def get_stock_id(cursor, conn, folder_name):
    cursor.execute('SELECT id FROM stocks WHERE folder_name=?', (folder_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute('INSERT INTO stocks (folder_name) VALUES (?)', (folder_name,))
    conn.commit()
    return cursor.lastrowid

def safe_commit(conn, retries=5, delay=0.5):
    return conn.commit()


def insert_announcements_from_csv():
    print(f"Reading announcements from {ANN_CSV}")
    df = pd.read_csv(ANN_CSV)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    count = 0
    for _, row in df.iterrows():
        stock_folder = row.get('stock_folder')
        pdf_link = row.get('pdf_link')
        received_time = row.get('received_time')
        vader_score = row.get('vader_score')
        finbert_score = row.get('finbert_score')
        hybrid_score = row.get('hybrid_score')
        verdict = row.get('verdict')
        # Ensure stock exists
        stock_id = get_stock_id(cursor, conn, stock_folder)
        cursor.execute('''INSERT INTO announcements (stock_id, pdf_link, received_time, vader_score, finbert_score, hybrid_score, verdict) VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (stock_id, pdf_link, received_time, vader_score, finbert_score, hybrid_score, verdict))
        count += 1
        if count % 100 == 0:
            print(f"Inserted {count} announcements...")
            conn.commit()
    conn.commit()
    conn.close()
    print(f"Inserted {count} announcements into the database.")

if __name__ == '__main__':
    insert_announcements_from_csv()
