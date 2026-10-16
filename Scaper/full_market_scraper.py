import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import portalocker

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from utils.cookie_manager import CookieManager
from utils.nse_fetcher import NSEDataFetcher
from utils.market_scraper import MarketDataScraper
# Top-level safe parquet saving function
def save_market_data_parquet(new_data: pd.DataFrame, file_path: str):
    required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    new_data = new_data[required_cols]
    new_data = new_data.dropna()
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], errors='coerce')
    new_data['symbol'] = new_data['symbol'].astype(str)
    for col in ['open', 'high', 'low', 'close']:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    new_data['volume'] = pd.to_numeric(new_data['volume'], errors='coerce').astype('Int64')
    new_data = new_data.dropna()

    with portalocker.Lock(file_path + '.lock', timeout=60):
        if os.path.exists(file_path):
            existing = pd.read_parquet(file_path)
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data
        combined = combined.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(file_path)) as tmp:
            table = pa.Table.from_pandas(combined[required_cols])
            pq.write_table(table, tmp.name)
            tmp_path = tmp.name
        os.replace(tmp_path, file_path)


if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    """Main execution function"""
    print("🚀 Starting NSE Market Data Scraper...")
    
    # Initialize components
    nse_fetcher = NSEDataFetcher()
    market_scraper = MarketDataScraper()
    
    # Check if we need to refresh cookies
    refresh = "--refresh" in sys.argv
    
    # 1. Get the full list of tickers from NSE
    all_tickers = nse_fetcher.get_all_tickers(force_refresh_cookies=refresh)
    
    if not all_tickers:
        print("⚠️ Failed to fetch tickers from NSE. Attempting fallback to local stock_map.csv...")
        # Resolve path to stock_map.csv in the root folder
        stock_map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stock_map.csv"))
        if os.path.exists(stock_map_path):
            try:
                df_map = pd.read_csv(stock_map_path)
                all_tickers = [str(symbol).strip() + ".NS" for symbol in df_map["FolderName"].dropna()]
                print(f"✅ Loaded {len(all_tickers)} tickers from local stock_map.csv fallback.")
            except Exception as e:
                print(f"❌ Failed to load local stock_map.csv: {e}")
        else:
            print(f"❌ Local stock_map.csv not found at {stock_map_path}")
    
    if all_tickers:
        # For testing, you might want to scrape a smaller sample first
        # To scrape all, comment out the next line:
        # all_tickers = all_tickers[:50] # Scrapes the first 50 stocks

        # 2. Define the date range (e.g., 5 years of data)
        end_date_str = datetime.now().strftime('%Y-%m-%d')
        start_date_str = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        print(f"Fetching 5 years of daily data from {start_date_str} to {end_date_str}...")

        # 3. Call the scraper function
        scraped_data_list = market_scraper.fetch_market_data(
            ticker_list=all_tickers,
            start_date=start_date_str,
            end_date=end_date_str,
            interval='1d'
        )

        # 4. Load the combined scraped data
        metadata = scraped_data_list # fetch_market_data returns the metadata dictionary
        if not metadata or metadata.get('total_records', 0) == 0:
            print("\nNo data was scraped in this run.")
            return
            
        try:
            # Use a single folder and update files
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)

            # Use fixed filenames
            csv_path = os.path.join(data_dir, "full_nse_market_data.csv")
            excel_path = os.path.join(data_dir, "full_nse_market_data.xlsx")
            parquet_path = os.path.join(data_dir, "full_nse_market_data.parquet")
            parquet_dir = os.path.join(data_dir, "by_ticker")
            os.makedirs(parquet_dir, exist_ok=True)

            # Load the complete market data from the timestamped run directory
            run_dir = os.path.join(data_dir, f"market_data_{metadata['scraping_date']}")
            new_parquet_path = os.path.join(run_dir, "summary", "complete_market_data.parquet")
            
            if not os.path.exists(new_parquet_path):
                print(f"❌ Scraped parquet file not found at {new_parquet_path}")
                return
                
            master_dataframe = pd.read_parquet(new_parquet_path)
            if master_dataframe.empty:
                print("\nNo data was collected after combining results.")
                return

            # Save CSV
            master_dataframe.to_csv(csv_path, index=False)

            # Save Excel with metadata
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                master_dataframe.to_excel(writer, sheet_name='Market_Data', index=False)
                metadata = pd.DataFrame([
                    ['Data Range Start', master_dataframe['timestamp'].min()],
                    ['Data Range End', master_dataframe['timestamp'].max()],
                    ['Total Tickers', master_dataframe['symbol'].nunique()],
                    ['Total Records', len(master_dataframe)],
                    ['Generated On', datetime.now()],
                    ['Columns Description', 'timestamp: Date and time of the data point'],
                    ['', 'symbol: Stock symbol (e.g., RELIANCE.NS)'],
                    ['', 'open: Opening price'],
                    ['', 'high: Highest price of the day'],
                    ['', 'low: Lowest price of the day'],
                    ['', 'close: Closing price'],
                    ['', 'volume: Trading volume']
                ])
                metadata.to_excel(writer, sheet_name='Metadata', header=False, index=False)

            # Save main Parquet file safely
            save_market_data_parquet(master_dataframe, parquet_path)

            # Save ticker-wise parquet files for efficient reading
            for ticker in master_dataframe['symbol'].unique():
                ticker_data = master_dataframe[master_dataframe['symbol'] == ticker]
                safe_ticker = ticker.replace('.', '_')
                ticker_parquet_path = os.path.join(parquet_dir, f"{safe_ticker}.parquet")
                save_market_data_parquet(ticker_data, ticker_parquet_path)

            print("\n" + "="*50)
            print("      SCRAPING COMPLETE & DATA UPDATED")
            print("="*50)
            print(f"Total tickers processed: {metadata.get('total_tickers_processed', 0)}")
            print(f"Total rows collected: {len(master_dataframe)}")
            print(f"Date range: {master_dataframe['timestamp'].min()} to {master_dataframe['timestamp'].max()}")
            print("\nData saved in multiple formats:")
            print(f"1. CSV file: {csv_path}")
            print(f"2. Excel file with metadata: {excel_path}")
            print(f"3. Parquet files by ticker: {parquet_dir}")
            print("\nSample data (first 5 rows):")
            print(master_dataframe.head())
            
        except Exception as e:
            print(f"\nError while saving data: {str(e)}")
    else:
        print("\nCould not retrieve ticker list. Aborting data fetch.")

if __name__ == "__main__":
    main()