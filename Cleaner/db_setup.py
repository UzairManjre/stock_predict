import pandas as pd
import sqlite3
import os

def setup_database():
    """
    Sets up the SQLite database and creates the necessary tables for storing
    stock data, including OHLCV prices and corporate announcements.
    The database file will be created in the same directory as this script.
    """
    # The database will be named 'stock_data.db' and located in the 'Cleaner' directory.
    db_path = os.path.join(os.path.dirname(__file__), 'stock_data.db')
    print(f"Setting up database at: {db_path}")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create a 'stocks' table to hold basic information about each stock.
        # This table will link all other data to a specific stock.
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_name TEXT NOT NULL UNIQUE,
            bse_name_slug TEXT,
            bse_ticker TEXT,
            bse_security_code TEXT
        )
        ''')
        print("Table 'stocks' created or already exists.")

        # Create 'ohlcv_data' table for daily Open, High, Low, Close, Volume information.
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            FOREIGN KEY (stock_id) REFERENCES stocks (id)
        )
        ''')
        print("Table 'ohlcv_data' created or already exists.")

        # Create 'announcements' table to store corporate announcement details with sentiment fields.
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            pdf_link TEXT,
            received_time DATETIME,

            -- Sentiment fields
            vader_score REAL,
            finbert_score REAL,
            hybrid_score REAL,
            verdict TEXT,

            FOREIGN KEY (stock_id) REFERENCES stocks (id)
        )
        ''')
        print("Table 'announcements' created or already exists with sentiment fields.")

        conn.commit()
        print("Database setup complete. You can now run this script to create the 'stock_data.db' file.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    setup_database()
