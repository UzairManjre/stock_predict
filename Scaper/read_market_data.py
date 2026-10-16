import os
import pandas as pd
import glob
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
from datetime import datetime

class MarketDataReader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.current_session = self._get_latest_session()
        
    def _get_latest_session(self) -> Optional[str]:
        sessions = glob.glob(os.path.join(self.data_dir, "market_data_*"))
        if not sessions:
            return None
        return max(sessions)
    
    def list_available_stocks(self) -> List[str]:
        if not self.current_session:
            return []
        stock_dirs = glob.glob(os.path.join(self.current_session, "individual_data", "*"))
        return [os.path.basename(d) for d in stock_dirs]

    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.current_session:
            print("No data sessions found")
            return None

        stock_dir = os.path.join(self.current_session, "individual_data", symbol)
        if not os.path.exists(stock_dir):
            print(f"No data found for stock {symbol}")
            return None

        parquet_files = glob.glob(os.path.join(stock_dir, "data_*.parquet"))
        if not parquet_files:
            print(f"No parquet file found for stock {symbol}")
            return None

        df = pd.read_parquet(parquet_files[0])

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

        rename_map = {
            "('timestamp', '')": "timestamp",
            "timestamp": "timestamp",
            "timestamp_": "timestamp",
            "('symbol', '')": "symbol",
            "symbol": "symbol",
            "symbol_": "symbol",
        }
        df.rename(columns=rename_map, inplace=True)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            raise KeyError("No 'timestamp' column found after cleaning")

        print("Final columns:", df.columns.tolist())
        return df

    def get_stock_metadata(self, symbol: str) -> Optional[Dict]:
        if not self.current_session:
            return None
        metadata_file = os.path.join(self.current_session, "individual_data", symbol, "metadata.csv")
        if not os.path.exists(metadata_file):
            return None
        metadata = pd.read_csv(metadata_file)
        clean_metadata = {}
        for key, value in metadata.iloc[0].items():
            clean_metadata[key] = value
        return clean_metadata

    def get_all_stocks_data(self) -> pd.DataFrame:
        if not self.current_session:
            return pd.DataFrame()
        combined_file = os.path.join(self.current_session, "summary", "complete_market_data.parquet")
        if not os.path.exists(combined_file):
            return pd.DataFrame()
        return pd.read_parquet(combined_file)

    def get_session_info(self) -> Dict:
        if not self.current_session:
            return {}
        metadata_file = os.path.join(self.current_session, "metadata.csv")
        if not os.path.exists(metadata_file):
            return {}
        metadata = pd.read_csv(metadata_file)
        return dict(zip(metadata['Key'], metadata['Value']))

    def plot_stock_prices(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        df = self.get_stock_data(symbol)
        if df is None:
            return

        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]

        close_col = [c for c in df.columns if 'close' in c][0]
        high_col = [c for c in df.columns if 'high' in c][0]
        low_col = [c for c in df.columns if 'low' in c][0]
        volume_col = [c for c in df.columns if 'volume' in c][0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1], sharex=True)
        ax1.plot(df['timestamp'], df[close_col], label='Close Price', color='blue')
        ax1.set_title(f'{symbol} Stock Price and Volume')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2.bar(df['timestamp'], df[volume_col], label='Volume', color='gray', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        ax2.legend(loc='upper left')

        plt.xticks(rotation=45)

        stats_text = f'Current: {df[close_col].iloc[-1]:.2f}\n'
        stats_text += f'High: {df[high_col].max():.2f}\n'
        stats_text += f'Low: {df[low_col].min():.2f}\n'
        stats_text += f'Avg Vol: {df[volume_col].mean():,.0f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

def main():
    reader = MarketDataReader()
    stocks = reader.list_available_stocks()
    print(f"Found {len(stocks)} stocks in the latest session")
    print("\nAvailable stocks:")
    print(", ".join(stocks))

    session_info = reader.get_session_info()
    print("\nSession Information:")
    for key, value in session_info.items():
        print(f"{key}: {value}")

    if stocks:
        symbol = stocks[0]
        print(f"\nExample data for {symbol}:")
        data = reader.get_stock_data(symbol)
        if data is not None:
            print("\nFirst 5 rows:")
            print(data.head())

            print("\nMetadata:")
            metadata = reader.get_stock_metadata(symbol)
            for key, value in metadata.items():
                print(f"{key}: {value}")

            print("\nPlotting stock prices...")
            reader.plot_stock_prices(symbol)

if __name__ == "__main__":
    main()