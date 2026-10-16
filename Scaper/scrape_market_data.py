import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from utils.cookie_manager import CookieManager
from utils.nse_fetcher import NSEDataFetcher
from utils.market_scraper import MarketDataScraper

def main():
    """Main execution function"""
    print("🚀 Starting NSE Market Data Scraper...")
    
    # Setup output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    nse_fetcher = NSEDataFetcher()
    market_scraper = MarketDataScraper(output_dir=output_dir)
    
    # Parse command line arguments
    refresh = "--refresh" in sys.argv
    test_mode = "--test" in sys.argv
    
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
        # In test mode, only use a few tickers
        if test_mode:
            all_tickers = all_tickers[:5]
            print(f"TEST MODE: Using only {len(all_tickers)} tickers")

        # 2. Define the date range (e.g., 5 years of data)
        end_date_str = datetime.now().strftime('%Y-%m-%d')
        start_date_str = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        print(f"Fetching 5 years of daily data from {start_date_str} to {end_date_str}...")

        # 3. Call the scraper function with batch processing
        metadata = market_scraper.fetch_market_data(
            ticker_list=all_tickers,
            start_date=start_date_str,
            end_date=end_date_str,
            interval='1d',
            batch_size=50  # Process 50 tickers at a time
        )

        # 4. Print summary
        print("\n" + "="*50)
        print("      SCRAPING COMPLETE")
        print("="*50)
        print(f"Start Date: {metadata['start_date']}")
        print(f"End Date: {metadata['end_date']}")
        print(f"Total Tickers Processed: {metadata['total_tickers_processed']}")
        print(f"Successful Tickers: {metadata['successful_tickers']}")
        print(f"Failed Tickers: {metadata['failed_tickers']}")
        print(f"Total Records: {metadata['total_records']}")
        print(f"Number of Batches: {metadata['batch_count']}")
        
        # Print path to the data
        data_path = os.path.join(output_dir, f"market_data_{metadata['scraping_date']}")
        print(f"\nData is saved in: {data_path}")
        print("Directory Structure:")
        print("1. individual_data/")
        print("   └── [STOCK_NAME]/")
        print("       ├── data_STARTDATE_to_ENDDATE.parquet  # Stock's historical data")
        print("       └── metadata.csv                       # Stock-specific metadata")
        print("2. summary/")
        print("   ├── complete_market_data.parquet  # Complete dataset")
        print("   └── complete_market_data.csv      # Complete dataset in CSV")
        print("3. metadata.csv    - Scraping session details")
        print("4. failed_tickers.csv - List of tickers that failed (if any)")
    else:
        print("\nCould not retrieve ticker list. Aborting data fetch.")

if __name__ == "__main__":
    main()
