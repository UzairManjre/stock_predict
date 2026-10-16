import yfinance as yf
import pandas as pd
import time
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

class MarketDataScraper:
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the MarketDataScraper with an output directory.
        
        Args:
            output_dir: Directory where data will be stored
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def fetch_market_data(self, ticker_list: List[str], start_date: str, end_date: str, 
                         interval: str = '1d', batch_size: int = 50) -> Dict[str, Any]:
        """
        Fetches historical market data for a list of stock tickers from Yahoo Finance.
        
        Args:
            ticker_list: List of stock tickers with .NS suffix
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1d' for daily, '1wk' for weekly, etc.)
            batch_size: Number of tickers to process before saving a checkpoint
            
        Returns:
            Dictionary containing metadata about the scraping process
        """
        all_data = []
        successful_tickers = []
        failed_tickers = []
        total_records = 0
        
        # Create a timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(self.output_dir, f"market_data_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"--- Starting Market Data Scraper for {len(ticker_list)} tickers ---")
        print(f"Data will be saved in: {run_dir}")

        batch_count = 0
        current_batch = []
        
        for i, ticker in enumerate(ticker_list):
            print(f"Fetching data for: {ticker} ({i+1}/{len(ticker_list)})...")
            try:
                # Download data
                data = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    rounding=True
                )

                if data.empty:
                    print(f"  > WARNING: No data found for {ticker}.")
                    failed_tickers.append({"ticker": ticker, "reason": "No data found"})
                    continue

                # Ensure we have all required columns
                required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
                if not all(col in data.columns for col in required_columns):
                    print(f"  > WARNING: Missing required columns for {ticker}.")
                    failed_tickers.append({"ticker": ticker, "reason": "Missing columns"})
                    continue

                # Process the data
                data = data.reset_index()
                
                # Rename columns to lowercase
                column_mapping = {
                    'Date': 'timestamp',
                    'Datetime': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                data = data.rename(columns=column_mapping)
                
                # Add symbol column
                data['symbol'] = ticker
                
                # Ensure timestamp is datetime
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Select and order columns
                final_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                data = data[final_columns]
                
                # Create directory structure for individual ticker
                ticker_name = ticker.replace('.NS', '')
                ticker_dir = os.path.join(run_dir, "individual_data", ticker_name)
                os.makedirs(ticker_dir, exist_ok=True)
                
                # Save individual ticker data with date in filename
                ticker_file = os.path.join(ticker_dir, f"data_{start_date}_to_{end_date}.parquet")
                data.to_parquet(ticker_file, index=False)
                
                # Save additional ticker-specific metadata
                ticker_metadata = {
                    "symbol": ticker,
                    "data_points": len(data),
                    "date_range_start": data['timestamp'].min(),
                    "date_range_end": data['timestamp'].max(),
                    "average_volume": data['volume'].mean(),
                    "max_price": data[['high']].max().values[0],
                    "min_price": data[['low']].min().values[0],
                    "first_price": data.iloc[0]['open'],
                    "last_price": data.iloc[-1]['close']
                }
                
                # Save ticker metadata
                pd.DataFrame([ticker_metadata]).to_csv(
                    os.path.join(ticker_dir, "metadata.csv"), 
                    index=False
                )
                
                # Add to current batch
                current_batch.append(data)
                successful_tickers.append(ticker)
                total_records += len(data)
                
                # If batch size reached, save batch
                if len(current_batch) >= batch_size:
                    batch_count += 1
                    batch_df = pd.concat(current_batch, ignore_index=True)
                    batch_file = os.path.join(run_dir, f"batch_{batch_count}.parquet")
                    batch_df.to_parquet(batch_file, index=False)
                    current_batch = []
                    
                print(f"  > Successfully saved data for {ticker}")
                
            except Exception as e:
                print(f"  > ERROR: Could not fetch data for {ticker}. Reason: {e}")
                failed_tickers.append({"ticker": ticker, "reason": str(e)})
            
            # Respectful delay between requests
            time.sleep(0.5)
        
        # Save any remaining batch
        if current_batch:
            batch_count += 1
            batch_df = pd.concat(current_batch, ignore_index=True)
            batch_file = os.path.join(run_dir, f"batch_{batch_count}.parquet")
            batch_df.to_parquet(batch_file, index=False)
        
        # Save metadata
        metadata = {
            "scraping_date": timestamp,
            "start_date": start_date,
            "end_date": end_date,
            "total_tickers_processed": len(ticker_list),
            "successful_tickers": len(successful_tickers),
            "failed_tickers": len(failed_tickers),
            "total_records": total_records,
            "batch_count": batch_count,
            "interval": interval
        }
        
        # Save metadata and failed tickers info
        pd.DataFrame(metadata.items(), columns=['Key', 'Value']).to_csv(
            os.path.join(run_dir, "metadata.csv"), index=False
        )
        
        if failed_tickers:
            pd.DataFrame(failed_tickers).to_csv(
                os.path.join(run_dir, "failed_tickers.csv"), index=False
            )
        
        # Create summary directories
        print("\nCreating combined dataset...")
        summary_dir = os.path.join(run_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        combined_file = os.path.join(summary_dir, "complete_market_data.parquet")
        combined_csv = os.path.join(summary_dir, "complete_market_data.csv")
        
        # Read and combine all batch files
        all_batches = []
        for i in range(1, batch_count + 1):
            batch_file = os.path.join(run_dir, f"batch_{i}.parquet")
            all_batches.append(pd.read_parquet(batch_file))
        
        if all_batches:
            combined_data = pd.concat(all_batches, ignore_index=True)
            combined_data.to_parquet(combined_file, index=False)
            combined_data.to_csv(combined_csv, index=False)
            
            print(f"\nData saved successfully to local directory!")
            print(f"Total records: {total_records}")
            print(f"Successful tickers: {len(successful_tickers)}")
            print(f"Failed tickers: {len(failed_tickers)}")
            print(f"\nFiles saved in: {run_dir}")
            
            # Save to PostgreSQL
            try:
                from .db_connector import get_db_connection
                print("\nSaving combined market data to PostgreSQL...")
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # 1. Ensure all tickers exist in the companies table
                unique_symbols = combined_data['symbol'].unique()
                for sym in unique_symbols:
                    ticker = sym.replace('.NS', '').replace('.BO', '').strip()
                    cursor.execute("""
                        INSERT INTO companies (ticker, company_name, is_active)
                        VALUES (%s, %s, TRUE)
                        ON CONFLICT (ticker) DO NOTHING
                    """, (ticker, ticker))
                conn.commit()
                
                # 2. Build rows for bulk insert
                records = []
                for _, row in combined_data.iterrows():
                    date_val = pd.to_datetime(row['timestamp']).date()
                    ticker = row['symbol'].replace('.NS', '').replace('.BO', '').strip()
                    open_val = float(row['open']) if pd.notna(row['open']) else None
                    high_val = float(row['high']) if pd.notna(row['high']) else None
                    low_val = float(row['low']) if pd.notna(row['low']) else None
                    close_val = float(row['close']) if pd.notna(row['close']) else None
                    volume_val = int(row['volume']) if pd.notna(row['volume']) else None
                    
                    records.append((date_val, ticker, open_val, high_val, low_val, close_val, close_val, volume_val))
                
                # Bulk upsert using ON CONFLICT to update Close/Volume if they change
                batch_size = 5000
                for idx_b in range(0, len(records), batch_size):
                    batch = records[idx_b:idx_b+batch_size]
                    cursor.executemany("""
                        INSERT INTO market_data_daily (date, ticker, open_price, high_price, low_price, close_price, adj_close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, ticker) DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            adj_close = EXCLUDED.adj_close,
                            volume = EXCLUDED.volume
                    """, batch)
                conn.commit()
                cursor.close()
                conn.close()
                print(f"✅ Successfully saved {len(records)} records to PostgreSQL database.")
            except Exception as pg_err:
                print(f"⚠️ Warning: Could not save data to PostgreSQL: {pg_err}")
            
            print("\nMain data files:")
            print(f"1. Parquet: {combined_file}")
            print(f"2. CSV: {combined_csv}")
            print(f"3. Individual ticker files in the same directory")
        
        return metadata
