
import sqlite3
import pandas as pd
import os
import sys

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    script_dir = os.path.dirname(__file__)
    db_path = os.path.join(script_dir, 'stock_data.db')
    print(f"Connecting to database at: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        sys.exit(1)

def prepare_and_save_data(conn):
    """
    Reads data from the database, prepares it for model training by joining
    OHLCV and announcement data, and saves it to a new directory.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    training_data_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(training_data_dir, exist_ok=True)
    print(f"Prepared training data will be saved in: {training_data_dir}")

    # Get all stocks from the database
    stocks_df = pd.read_sql_query("SELECT id, folder_name FROM stocks", conn)

    for _, stock in stocks_df.iterrows():
        stock_id = stock['id']
        stock_name = stock['folder_name']
        print(f"\nProcessing data for {stock_name} (ID: {stock_id})...")

        # 1. Fetch OHLCV data
        ohlcv_query = f"SELECT date, open, high, low, close, volume FROM ohlcv_data WHERE stock_id = {stock_id}"
        ohlcv_df = pd.read_sql_query(ohlcv_query, conn, parse_dates=['date'])
        
        if ohlcv_df.empty:
            print(f" -> No OHLCV data found for {stock_name}. Skipping.")
            continue
        
        ohlcv_df.set_index('date', inplace=True)

        # 2. Fetch and process announcement data
        announcements_query = f"SELECT received_time FROM announcements WHERE stock_id = {stock_id}"
        announcements_df = pd.read_sql_query(announcements_query, conn, parse_dates=['received_time'])
        
        if not announcements_df.empty:
            # Extract date part and count announcements per day
            announcements_df['date'] = announcements_df['received_time'].dt.date
            daily_announcements = announcements_df.groupby('date').size().reset_index(name='announcement_count')
            daily_announcements['date'] = pd.to_datetime(daily_announcements['date'])
            daily_announcements.set_index('date', inplace=True)
        else:
            daily_announcements = pd.DataFrame(columns=['announcement_count']) # Empty dataframe

        # 3. Join OHLCV data with announcement counts
        # We use a left join to keep all trading days, even those without announcements
        merged_df = ohlcv_df.join(daily_announcements, how='left')

        # Fill days with no announcements with 0
        merged_df['announcement_count'].fillna(0, inplace=True)
        merged_df['announcement_count'] = merged_df['announcement_count'].astype(int)

        # 4. Save the prepared data for this stock
        output_path = os.path.join(training_data_dir, f"{stock_name}_training.parquet")
        merged_df.to_parquet(output_path)
        print(f" -> Saved prepared data for {stock_name} to {output_path}")

def main():
    """Main function to run the data preparation process."""
    conn = get_db_connection()
    prepare_and_save_data(conn)
    conn.close()
    print("\nData preparation process finished. Connection closed.")

if __name__ == '__main__':
    main()
