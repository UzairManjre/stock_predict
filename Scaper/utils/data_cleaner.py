import pandas as pd
import os
from datetime import datetime
import numpy as np

class DataCleaner:
    @staticmethod
    def clean_market_data(input_file: str, output_dir: str) -> None:
        """
        Cleans and restructures the market data into a proper format.
        
        Args:
            input_file: Path to the input CSV file
            output_dir: Directory to save cleaned data
        """
        print("Reading raw data file...")
        # Read the CSV file with low_memory=False to handle mixed types
        df = pd.read_csv(input_file, low_memory=False, header=[0, 1])
        
        # Print column structure info
        print("Column levels:", df.columns.levels)
        print("Column structure:", df.columns)
        
        # Get unique tickers from open price columns
        unique_tickers = sorted(list(set([col[1] for col in df.columns if col[0] == 'open' and isinstance(col[1], str) and '.NS' in col[1]])))
        
        print(f"Found {len(unique_tickers)} unique tickers")
        
        # Create cleaned dataframes for each data type
        clean_dfs = []
        
        for ticker in unique_tickers:
            print(f"Processing {ticker}...")
            
            try:
                # Get all columns for this ticker
                open_col = ('open', ticker)
                high_col = ('high', ticker)
                low_col = ('low', ticker)
                close_col = ('close', ticker)
                volume_col = ('volume', ticker)
                
                # Create DataFrame for this ticker
                ticker_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(df[('timestamp', 'Unnamed: 0_level_1')]),
                    'symbol': ticker,
                    'open': pd.to_numeric(df[open_col], errors='coerce'),
                    'high': pd.to_numeric(df[high_col], errors='coerce'),
                    'low': pd.to_numeric(df[low_col], errors='coerce'),
                    'close': pd.to_numeric(df[close_col], errors='coerce'),
                    'volume': pd.to_numeric(df[volume_col], errors='coerce')
                })
                
                # Remove rows where all price data is NaN
                ticker_df = ticker_df.dropna(subset=['open', 'high', 'low', 'close'], how='all')
                
                if not ticker_df.empty:
                    clean_dfs.append(ticker_df)
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Combine all clean data
        print("Combining cleaned data...")
        clean_data = pd.concat(clean_dfs, ignore_index=True)
        
        # Sort by timestamp and symbol
        clean_data = clean_data.sort_values(['symbol', 'timestamp'])
        
        # Remove any remaining NaN values
        clean_data = clean_data.dropna()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get date range for filenames
        start_date = clean_data['timestamp'].min().strftime('%Y%m%d')
        end_date = clean_data['timestamp'].max().strftime('%Y%m%d')
        base_filename = f"nse_market_data_{start_date}_to_{end_date}"
        
        # Save as CSV
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        clean_data.to_csv(csv_path, index=False)
        
        # Save as Excel with metadata
        excel_path = os.path.join(output_dir, f"{base_filename}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write main data
            clean_data.to_excel(writer, sheet_name='Market_Data', index=False)
            
            # Create metadata
            metadata = pd.DataFrame([
                ['Data Range Start', clean_data['timestamp'].min()],
                ['Data Range End', clean_data['timestamp'].max()],
                ['Total Tickers', clean_data['symbol'].nunique()],
                ['Total Records', len(clean_data)],
                ['Generated On', datetime.now()],
                ['Data Statistics', ''],
                ['Min Price', clean_data[['open', 'high', 'low', 'close']].min().min()],
                ['Max Price', clean_data[['open', 'high', 'low', 'close']].max().max()],
                ['Avg Volume', clean_data['volume'].mean()],
                ['', ''],
                ['Columns Description', ''],
                ['timestamp', 'Date and time of the data point'],
                ['symbol', 'Stock symbol (e.g., RELIANCE.NS)'],
                ['open', 'Opening price'],
                ['high', 'Highest price of the day'],
                ['low', 'Lowest price of the day'],
                ['close', 'Closing price'],
                ['volume', 'Trading volume']
            ])
            metadata.to_excel(writer, sheet_name='Metadata', header=False, index=False)
            
            # Add summary statistics
            stats = clean_data.groupby('symbol').agg({
                'close': ['count', 'mean', 'std', 'min', 'max'],
                'volume': ['mean', 'max']
            }).round(2)
            stats.columns = ['Days', 'Avg Price', 'Std Dev', 'Min Price', 'Max Price', 'Avg Volume', 'Max Volume']
            stats.to_excel(writer, sheet_name='Statistics')
        
        # Save individual ticker files as parquet
        parquet_dir = os.path.join(output_dir, f"{base_filename}_by_ticker")
        os.makedirs(parquet_dir, exist_ok=True)
        
        for ticker in clean_data['symbol'].unique():
            ticker_data = clean_data[clean_data['symbol'] == ticker]
            safe_ticker = ticker.replace('.', '_')
            parquet_path = os.path.join(parquet_dir, f"{safe_ticker}.parquet")
            ticker_data.to_parquet(parquet_path, index=False)
        
        # Print summary
        print("\n" + "="*50)
        print("      DATA CLEANING COMPLETE")
        print("="*50)
        print(f"Total tickers processed: {clean_data['symbol'].nunique()}")
        print(f"Total records: {len(clean_data)}")
        print(f"Date range: {clean_data['timestamp'].min()} to {clean_data['timestamp'].max()}")
        print("\nFiles saved:")
        print(f"1. CSV file: {csv_path}")
        print(f"2. Excel file with metadata: {excel_path}")
        print(f"3. Individual ticker files: {parquet_dir}")
        print("\nSample of cleaned data:")
        print(clean_data.head())
